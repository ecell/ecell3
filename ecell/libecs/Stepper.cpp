//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include <functional>
#include <algorithm>
#include <limits>

#include "Util.hpp"
#include "Substance.hpp"
#include "Reactor.hpp"
#include "Model.hpp"
#include "FullID.hpp"
#include "PropertySlotMaker.hpp"

#include "Stepper.hpp"


namespace libecs
{


  ////////////////////////// Stepper

  
  void Stepper::makeSlots()
  {

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ID", *this,
				      Type2Type<String>(),
				      NULLPTR,
				      &Stepper::getID ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SystemList", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getSystemList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "CurrentTime", *this,
				      Type2Type<Real>(),
				      NULLPTR,
				      &Stepper::getCurrentTime ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "StepInterval", *this,
				      Type2Type<Real>(),
				      &Stepper::setStepInterval,
				      &Stepper::getStepInterval ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "UserMaxInterval", *this,
				      Type2Type<Real>(),
				      &Stepper::setUserMaxInterval,
				      &Stepper::getUserMaxInterval ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "UserMinInterval", *this,
				      Type2Type<Real>(),
				      &Stepper::setUserMinInterval,
				      &Stepper::getUserMinInterval ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "MaxInterval", *this,
				      Type2Type<Real>(),
				      NULLPTR,
				      &Stepper::getMaxInterval ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "MinInterval", *this,
				      Type2Type<Real>(),
				      NULLPTR,
				      &Stepper::getMinInterval ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "StepIntervalConstraint", *this,
				      Type2Type<Polymorph>(),
				      &Stepper::setStepIntervalConstraint,
				      &Stepper::getStepIntervalConstraint ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SlaveStepper", *this,
				      Type2Type<String>(),
				      &Stepper::setSlaveStepperID,
				      &Stepper::getSlaveStepperID ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SubstanceCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getSubstanceCache ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ReactorCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getReactorCache ) );

  }

  Stepper::Stepper() 
    :
    theModel( NULLPTR ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theTolerantStepInterval( 0.001 ),
    theNextStepInterval( 0.001 ),
    theUserMinInterval( 0.0 ),
    theUserMaxInterval( std::numeric_limits<Real>::max() ),
    theSlaveStepper( NULLPTR )
  {
    makeSlots();
  }

  void Stepper::initialize()
  {
    FOR_ALL( SystemVector, theSystemVector, initialize );

    Int aSize( theSubstanceCache.size() );

    theQuantityBuffer.resize( aSize );
    theVelocityBuffer.resize( aSize );

    //    if( isEntityListChanged() )
    //      {

    //
    // update theReactorCache
    //
    theReactorCache.clear();
    for( SystemVectorConstIterator i( theSystemVector.begin() );
	 i != theSystemVector.end() ; ++i )
	{
	  const SystemCptr aSystem( *i );

	  for( ReactorMapConstIterator 
		 j( aSystem->getReactorMap().begin() );
	       j != aSystem->getReactorMap().end(); j++ )
	    {
	      ReactorPtr aReactorPtr( (*j).second );

	      theReactorCache.push_back( aReactorPtr );

	      aReactorPtr->initialize();
	    }
	}

    // sort by Reactor priority
    std::sort( theReactorCache.begin(), theReactorCache.end(),
	       Reactor::PriorityCompare() );


    //
    // Update theSubstanceCache
    //

    // get all the substances which are reactants of the Reactors
    theSubstanceCache.clear();
    // for all the reactors
    for( ReactorCache::const_iterator i( theReactorCache.begin());
	 i != theReactorCache.end() ; ++i )
      {
	ReactantMapCref aReactantMap( (*i)->getReactantMap() );

	// for all the reactants
	for( ReactantMapConstIterator j( aReactantMap.begin() );
	     j != aReactantMap.end(); ++j )
	  {
	    SubstancePtr aSubstancePtr( j->second.getSubstance() );

	    // prevent duplication
	    if( std::find( theSubstanceCache.begin(), theSubstanceCache.end(),
			   aSubstancePtr ) == theSubstanceCache.end() )
	      {
		theSubstanceCache.push_back( aSubstancePtr );
		aSubstancePtr->registerStepper( this );
		aSubstancePtr->initialize();
	      }
	  }
      }

    //    clearEntityListChanged();
    //      }
    
  }


  const Polymorph Stepper::getSystemList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theSystemVector.size() );

    for( SystemVectorConstIterator i( getSystemVector().begin() );
	 i != getSystemVector().end() ; ++i )
      {
	SystemCptr aSystemPtr( *i );
	FullIDCref aFullID( aSystemPtr->getFullID() );
	const String aFullIDString( aFullID.getString() );

	aVector.push_back( aFullIDString );
      }

    return aVector;
  }


  void Stepper::registerSystem( SystemPtr aSystem )
  { 
    if( std::find( theSystemVector.begin(), theSystemVector.end(), aSystem ) 
   	== theSystemVector.end() )
      {
   	theSystemVector.push_back( aSystem );
      }
  }

  void Stepper::removeSystem( SystemPtr aSystem )
  { 
    SystemVectorIterator i( find( theSystemVector.begin(), 
				  theSystemVector.end(),
				  aSystem ) );
    
    if( i == theSystemVector.end() )
      {
	THROW_EXCEPTION( NotFound,
			 getClassName() + String( ": " ) 
			 + getID() + ": " + aSystem->getFullID().getString() 
			 + " not found in this stepper." );
      }

    theSystemVector.erase( i );
  }

  void Stepper::registerLoggedPropertySlot( PropertySlotPtr aPropertySlotPtr )
  {
    theLoggedPropertySlotVector.push_back( aPropertySlotPtr );
  }

  void Stepper::setSlaveStepperID( StringCref aStepperID )
  {
    if( aStepperID == "" )
      {
	setSlaveStepper( NULLPTR );
      }
    else
      {
	setSlaveStepper( getModel()->getStepper( aStepperID ) );
      }
  }

  const String Stepper::getSlaveStepperID() const
  {
    StepperPtr aStepperPtr( getSlaveStepper() );
    if( aStepperPtr == NULLPTR )
      {
	return String();
      }
    else
      {
	return aStepperPtr->getID();
      }
  }

  void Stepper::setStepIntervalConstraint( PolymorphCref aValue )
  {
    PolymorphVector aVector( aValue.asPolymorphVector() );
    checkSequenceSize( aVector, 2 );

    const StepperPtr aStepperPtr( getModel()->
				  getStepper( aVector[0].asString() ) );
    const Real aFactor( aVector[1].asReal() );

    setStepIntervalConstraint( aStepperPtr, aFactor );
  }

  const Polymorph Stepper::getStepIntervalConstraint() const
  {
    PolymorphVector aVector;
    aVector.reserve( theStepIntervalConstraintMap.size() );

    for( StepIntervalConstraintMapConstIterator 
	   i( theStepIntervalConstraintMap.begin() ); 
	      i != theStepIntervalConstraintMap.end() ; ++i )
      {
	PolymorphVector anInnerVector;
	anInnerVector.push_back( (*i).first->getID() );
	anInnerVector.push_back( (*i).second );

	aVector.push_back( anInnerVector );
      }

    return aVector;
  }

  void Stepper::setStepIntervalConstraint( StepperPtr aStepperPtr,
					   RealCref aFactor )
  {
    theStepIntervalConstraintMap.erase( aStepperPtr );

    if( aFactor != 0.0 )
      {
	theStepIntervalConstraintMap.
	  insert( std::make_pair( aStepperPtr, aFactor ) );
      }
  }

  const Real Stepper::getMaxInterval() const
  {
    Real aMaxInterval( getUserMaxInterval() );

    for( StepIntervalConstraintMapConstIterator 
	   i( theStepIntervalConstraintMap.begin() ); 
	      i != theStepIntervalConstraintMap.end() ; ++i )
      {
	const StepperPtr aStepperPtr( (*i).first );
	Real aConstraint( aStepperPtr->getStepInterval() * (*i).second );

	if( aMaxInterval > aConstraint )
	  {
	    aMaxInterval = aConstraint;
	  }
      }

    return aMaxInterval;
  }
  
  void Stepper::log()
  {
    // update loggers
    FOR_ALL( PropertySlotVector, theLoggedPropertySlotVector, updateLogger );
  }



  const Polymorph Stepper::getSubstanceCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theSubstanceCache.size() );
    
    for( SubstanceCache::const_iterator i( theSubstanceCache.begin() );
	 i != theSubstanceCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }
  
  const Polymorph Stepper::getReactorCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theReactorCache.size() );
    
    for( ReactorCache::const_iterator i( theReactorCache.begin() );
	 i != theReactorCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }
  

  const UnsignedInt Stepper::getSubstanceCacheIndex( SubstancePtr aSubstance )
  {
    SubstanceCache::const_iterator 
      anIterator( std::find( theSubstanceCache.begin(), 
			     theSubstanceCache.end(), aSubstance ) );

    return anIterator - theSubstanceCache.begin();
  }

  inline void Stepper::clear()
  {
    //
    // Substance::clear()
    //
    const UnsignedInt aSize( theSubstanceCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	// save original quantity values
	theQuantityBuffer[ c ] = aSubstance->saveQuantity();

	// clear phase is here!
	aSubstance->clear();
      }


    //    FOR_ALL( SubstanceCache, theSubstanceCache, clear );
      
    //
    // Reactor::clear() ?
    //
    //FOR_ALL( ,, clear );
      
    //
    // System::clear() ?
    //
    //FOR_ALL( ,, clear );
  }

  inline void Stepper::react()
  {
    //
    // Reactor::react()
    //
    FOR_ALL( ReactorCache, theReactorCache, react );

  }

  inline void Stepper::integrate()
  {
    //
    // Substance::integrate()
    //
    const UnsignedInt aSize( theSubstanceCache.size() );

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	aSubstance->integrate( getCurrentTime() );
      }
  }

  inline void Stepper::slave()
  {
    // call slave
    StepperPtr aSlaveStepperPtr( getSlaveStepper() );
    if( aSlaveStepperPtr != NULLPTR )
      {
	aSlaveStepperPtr->step();
	aSlaveStepperPtr->setCurrentTime( getCurrentTime() );
      }
  }

  inline void Stepper::reset()
  {
    const UnsignedInt aSize( theSubstanceCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	// restore x (original value)
	aSubstance->setQuantity( theQuantityBuffer[ c ] );

	// clear velocity
	theVelocityBuffer[ c ] = 0.0;
	aSubstance->setVelocity( 0.0 );
      }
  }

  inline void Stepper::updateVelocityBuffer()
  {
    setStepInterval( getNextStepInterval() );

    const UnsignedInt aSize( theSubstanceCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	theVelocityBuffer[ c ] = aSubstance->getVelocity();

	// avoid negative value
	while( aSubstance->checkRange( getStepInterval() ) == false )
	  {
	    // don't use setStepInterval()
	    loadStepInterval( getStepInterval()*0.5 );
	  }
      }

    if( getStepInterval() < getTolerantStepInterval() )
      {
  	setNextStepInterval( getStepInterval() * 2.0 );
      }
    else setNextStepInterval( getStepInterval() );
  }



  ////////////////////////// Euler1Stepper

  Euler1Stepper::Euler1Stepper()
  {
    ; // do nothing
  }

  void Euler1Stepper::step()
  {
    integrate();
    slave();
    log();
    clear();
    react();

    updateVelocityBuffer();
  }


  ////////////////////////// RungeKutta4Stepper

  RungeKutta4Stepper::RungeKutta4Stepper()
  {
    ; // do nothing
  }

  void RungeKutta4Stepper::step()
  {
    // integrate phase first
    integrate();

    slave();

    log();

    // clear
    clear();

    // ========= 1 ===========
    react();

    const UnsignedInt aSize( theSubstanceCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	// get k1
	Real aVelocity( aSubstance->getVelocity() );

	// restore k1 / 2 + x
	aSubstance->loadQuantity( aVelocity * .5 * getStepInterval()
				  + theQuantityBuffer[ c ] );

	theVelocityBuffer[ c ] = aVelocity;

	// clear velocity
	aSubstance->setVelocity( 0 );
      }

    // ========= 2 ===========
    react();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	const Real aVelocity( aSubstance->getVelocity() );
	theVelocityBuffer[ c ] += aVelocity + aVelocity;

	// restore k2 / 2 + x
	aSubstance->loadQuantity( aVelocity * .5 * getStepInterval()
				  + theQuantityBuffer[ c ] );


	// clear velocity
	aSubstance->setVelocity( 0 );
      }


    // ========= 3 ===========
    react();
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	const Real aVelocity( aSubstance->getVelocity() );
	theVelocityBuffer[ c ] += aVelocity + aVelocity;

	// restore k3 + x
	aSubstance->loadQuantity( aVelocity * getStepInterval()
				  + theQuantityBuffer[ c ] );

	// clear velocity
	aSubstance->setVelocity( 0 );
      }


    // ========= 4 ===========
    react();

    // restore theQuantityBuffer
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	const Real aVelocity( aSubstance->getVelocity() );

	// restore x (original value)
	aSubstance->loadQuantity( theQuantityBuffer[ c ] );

	//// x(n+1) = x(n) + 1/6 * (k1 + k4 + 2 * (k2 + k3)) + O(h^5)

	theVelocityBuffer[ c ] += aVelocity;
	theVelocityBuffer[ c ] *= ( 1.0 / 6.0 );
	aSubstance->setVelocity( theVelocityBuffer[ c ] );
      }
    
    // don't call updateVelocityBuffer() -- it is already updated by
    // the algorithm.
  }


  ////////////////////////// AdaptiveStepsizeEuler1Stepper

  AdaptiveStepsizeEuler1Stepper::AdaptiveStepsizeEuler1Stepper()
  {
    ; // do nothing
  }

  void AdaptiveStepsizeEuler1Stepper::initialize()
  {
    Stepper::initialize();

    setUserMaxInterval( 0.001 );
  }

  void AdaptiveStepsizeEuler1Stepper::step()
  {
    Real delta, delta_max;
    Real aTolerance, anError;
    Real aStepInterval;

    const Real eps_abs( 1.0e-16 );
    const Real eps_rel( 1.0e-4 );
    const Real a_y( 0.0 );
    const Real a_dydt( 1.0 );

    const Real safety( 0.9 );

    const UnsignedInt aSize( theSubstanceCache.size() );

    integrate();
    slave();
    log();
    clear();

    setStepInterval( getNextStepInterval() );

    while( 1 )
      {

	// ========= 1 ===========
	react();

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	    // get k1
	    const Real aVelocity( aSubstance->getVelocity() );
	    theVelocityBuffer[ c ] = aVelocity;

	    aSubstance->loadQuantity( aVelocity * .5 * getStepInterval() 
				      + theQuantityBuffer[ c ] );
	    aSubstance->setVelocity( 0 );
	  }

	// ========= 2 ===========
	react();

	delta_max = 0.0;

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	    // get k2 = f(x+h/2, y+k1*h/2)
	    const Real aVelocity( aSubstance->getVelocity() );

  	    aTolerance = eps_abs 
	      + eps_rel * ( a_y * fabs(theQuantityBuffer[ c ]) 
			    + a_dydt * getStepInterval() * fabs(theVelocityBuffer[ c ]) );

	    anError = theVelocityBuffer[ c ] - aVelocity;
	    delta = fabs(anError / aTolerance);

	    getchar();

	    if( delta > delta_max )
	      {
		delta_max = delta;
		
		// shrink it if the error exceeds 110%
		if( delta_max > 1.1 )
		  {
               	    setStepInterval( getStepInterval() * 0.5 );
//      		    setStepInterval( getStepInterval() * pow(delta_max, -1.0) *  safety );

		    reset();
		    continue;
		  }
	      }

	    // restore x and dx/dt (original value)
	    aSubstance->loadQuantity( theQuantityBuffer[ c ] );
	    aSubstance->setVelocity( theVelocityBuffer[ c ] );
	  }


	if( delta_max <= 0.5 )
	  {
	    // grow it if error is 50% less than desired
    	    aStepInterval = getStepInterval() * 2.0;
//    	    aStepInterval = getStepInterval() * pow(delta_max, -0.5) * safety;

	    if( aStepInterval >= getUserMaxInterval() )
	      {
		aStepInterval = getUserMaxInterval();
	      }

	    setNextStepInterval( aStepInterval );
	  }
	else
	  setNextStepInterval( getStepInterval() );

	break;
      }

    // don't call updateVelocityBuffer() -- it is already updated by
    // the algorithm.
  }


 ////////////////////////// AdaptiveStepsizeMidpoint2Stepper

  AdaptiveStepsizeMidpoint2Stepper::AdaptiveStepsizeMidpoint2Stepper()
  {
    ; // do nothing
  }

  void AdaptiveStepsizeMidpoint2Stepper::initialize()
  {
    Stepper::initialize();

    const UnsignedInt aSize( theSubstanceCache.size() );

    theK1.resize( aSize );
    theErrorEstimate.resize( aSize );
  }

  void AdaptiveStepsizeMidpoint2Stepper::step()
  {
    Real maxError( 0.0 );
    Real desiredError, tmpError;
    Real aStepInterval;

    const UnsignedInt aSize( theSubstanceCache.size() );

    const Real eps_rel( 1.0e-8 );
    const Real eps_abs( 1.0e-6 );
    const Real a_y( 0.0 );
    const Real a_dydt( 0.0 );

    const Real safety( 0.9 );

    // integrate phase first
    integrate();

    slave();

    log();

    // clear
    clear();

    setStepInterval( getNextStepInterval() );

    while( 1 )
      {

	// ========= 1 ===========
	react();

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    // get k1
	    theK1[ c ] = aSubstance->getVelocity();

	    // restore k1/2 + x
	    aSubstance->loadQuantity( theK1[ c ] * .5  * getStepInterval()
				      + theQuantityBuffer[ c ] );

	    // k1 * for ~Yn+1
	    theErrorEstimate[ c ] = theK1[ c ];

	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }

	// ========= 2 ===========
	react();

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    // get k2
	    theVelocityBuffer[ c ] = aSubstance->getVelocity();
	    
	    // restore -k1+ k2 * 2 + x
	    aSubstance->loadQuantity( (-theK1[ c ] + theVelocityBuffer[ c ] * 2.0) * getStepInterval()
				      + theQuantityBuffer[ c ] );
	    
	    // k2 * 4 for ~Yn+1
  	    theErrorEstimate[ c ] += theVelocityBuffer[ c ] * 4.0;

	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }
	
	// ========= 3 ===========
	react();
	
	maxError = 0.0;

	// restore theQuantityBuffer
	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	    const Real aVelocity( aSubstance->getVelocity() );

	    // k3 for ~Yn+1
	    theErrorEstimate[ c ] += aVelocity;
	    theErrorEstimate[ c ] /= 6.0;

//  	    desiredError = eps_rel * ( a_y * fabs(theQuantityBuffer[ c ]) + a_dydt * getStepInterval() * fabs(theVelocityBuffer[ c ]) ) + eps_abs;
//  	    tmpError = fabs(theVelocityBuffer[ c ] - theErrorEstimate[ c ]) / desiredError;
	    
//  	    if( tmpError > maxError )
//  	      {
//  		maxError = tmpError;
		
//  		if( maxError > 1.1 )
//  		  {
//  		    // shrink it if the error exceeds 110%
//  		    setStepInterval( getStepInterval() * pow(maxError , -0.5) *  safety );

//  		    reset();
//  		    continue;
//  		  }
//  	      }

	    // restore x (original value)
	    aSubstance->loadQuantity( theQuantityBuffer[ c ] );

	    //// x(n+1) = x(n) + k2 * aStepInterval + O(h^3)
	    aSubstance->setVelocity( theVelocityBuffer[ c ] );
	  }

	// grow it if error is 50% less than desired
//  	if (maxError <= 0.5)
//  	  {
//  	    aStepInterval = getStepInterval() * pow(maxError , -1.0/3.0) * safety;
//  	    if( aStepInterval >= getUserMaxInterval() )
//  	      {
//  		aStepInterval = getStepInterval();
//  	      }

//  	    setNextStepInterval( aStepInterval );
//  	  }
//  	else setNextStepInterval( getStepInterval() );

	setNextStepInterval( getStepInterval() );

	break;
      }

    // don't call updateVelocityBuffer() -- it is already updated by
    // the algorithm.
  }


  ////////////////////////// CashKarp4Stepper

  CashKarp4Stepper::CashKarp4Stepper()
  {
    ; // do nothing
  }

  void CashKarp4Stepper::initialize()
  {
    Stepper::initialize();

    const UnsignedInt aSize( theSubstanceCache.size() );

    theK1.resize( aSize );
    theK2.resize( aSize );
    theK3.resize( aSize );
    theK4.resize( aSize );
    theK5.resize( aSize );
    theK6.resize( aSize );

    theErrorEstimate.resize( aSize );
  }
  
  void CashKarp4Stepper::step()
  {
    Real maxError( 0.0 );
    Real desiredError, tmpError;
    Real aStepInterval;

    const UnsignedInt aSize( theSubstanceCache.size() );

    const Real eps_rel( 1.0e-8 );
    const Real eps_abs( 1.0e-10 );
    const Real a_y( 0.0 );
    const Real a_dydt( 0.0 );
    const Real safety( 0.9 );

    // integrate phase first
    integrate();

    slave();

    log();

    // clear
    clear();

    setStepInterval( getNextStepInterval() );

    while( 1 )
      {

	// ========= 1 ===========
	react();

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    // get k1
	    theK1[ c ] = aSubstance->getVelocity();

	    // restore k1 / 5 + x
	    aSubstance->loadQuantity( theK1[ c ] * .2  * getStepInterval()
				      + theQuantityBuffer[ c ] );

	    // k1 * 37/378 for Yn+1
	    theVelocityBuffer[ c ] = theK1[ c ] * ( 37.0 / 378.0 );
	    // k1 * 2825/27648 for ~Yn+1
	    theErrorEstimate[ c ] = theK1[ c ] * ( 2825.0 / 27648.0 );

	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }

	// ========= 2 ===========
	react();

	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    theK2[ c ] = aSubstance->getVelocity();
	    
	    // restore k1 * 3/40+ k2 * 9/40 + x
	    aSubstance->loadQuantity( theK1[ c ] * ( 3.0 / 40.0 ) * getStepInterval()
				      + theK2[ c ] * ( 9.0 / 40.0 ) * getStepInterval()
				      + theQuantityBuffer[ c ] );
	    
	    // k2 * 0 for Yn+1 (do nothing)
//  	    theVelocityBuffer[ c ] = theK2[ c ] * 0;
	    // k2 * 0 for ~Yn+1 (do nothing)
//  	    theErrorEstimate[ c ] = theK2[ c ] * 0;
	    
	    
	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }
	
	// ========= 3 ===========
	react();
	
	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    theK3[ c ] = aSubstance->getVelocity();
	    
	    // restore k1 * 3/10 - k2 * 9/10 + k3 * 6/5 + x
	    aSubstance->loadQuantity( theK1[ c ] * ( 3.0 / 10.0 ) * getStepInterval()
				      - theK2[ c ] * ( 9.0 / 10.0 ) * getStepInterval()
				      + theK3[ c ] * ( 6.0 / 5.0 ) * getStepInterval()
				      + theQuantityBuffer[ c ] );
	    
	    // k3 * 250/621 for Yn+1
	    theVelocityBuffer[ c ] += theK3[ c ] * ( 250.0 / 621.0 );
	    // k3 * 18575/48384 for ~Yn+1
	    theErrorEstimate[ c ] += theK3[ c ] * ( 18575.0 / 48384.0 );
	    
	    
	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }
	
	// ========= 4 ===========
	react();
	
	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    theK4[ c ] = aSubstance->getVelocity();
	    
	    // restore k2 * 5/2 - k1 * 11/54 - k3 * 70/27 + k4 * 35/27 + x
	    aSubstance->loadQuantity( theK2[ c ] * ( 5.0 / 2.0 ) * getStepInterval()
				      - theK1[ c ] * ( 11.0 / 54.0 ) * getStepInterval()
				      - theK3[ c ] * ( 70.0 / 27.0 ) * getStepInterval()
				      + theK4[ c ] * ( 35.0 / 27.0 ) * getStepInterval()
				      + theQuantityBuffer[ c ] );
	    
	    // k4 * 125/594 for Yn+1
	    theVelocityBuffer[ c ] += theK4[ c ] * ( 125.0 / 594.0 );
	    // k4 * 13525/55296 for ~Yn+1
	    theErrorEstimate[ c ] += theK4[ c ] * ( 13525.0 / 55296.0 );
	    
	    
	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }
	
	
	// ========= 5 ===========
	react();
	
	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	    
	    theK5[ c ] = aSubstance->getVelocity();
	    
	    // restore k1 * 1631/55296 + k2 * 175/512 + k3 * 575/13824 + k4 * 44275/110592 + k5 * 253/4096 + x
	    aSubstance->loadQuantity( theK1[ c ] * ( 1631.0 / 55296.0 ) * getStepInterval()
				      + theK2[ c ] * ( 175.0 / 512.0 ) * getStepInterval()
				      + theK3[ c ] * ( 575.0 / 13824.0 ) * getStepInterval()
				      + theK4[ c ] * ( 44275.0 / 110592.0 ) * getStepInterval()
				      + theK5[ c ] * ( 253.0 / 4096.0 ) * getStepInterval()
				      + theQuantityBuffer[ c ] );
	    
	    // k5 * 0 for Yn+1(do nothing)
//  	    theVelocityBuffer[ c ] += theK5[ c ] * 0;
	    // k5 * 277/14336 for ~Yn+1
	    theErrorEstimate[ c ] += theK5[ c ] * ( 277.0 / 14336.0 );
	    
	    
	    // clear velocity
	    aSubstance->setVelocity( 0 );
	  }
		
	// ========= 6 ===========
	react();
	
	maxError = 0.0;

	// restore theQuantityBuffer
	for( UnsignedInt c( 0 ); c < aSize; ++c )
	  {
	    SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	    theK6[ c ] = aSubstance->getVelocity();

	    // k6 * 512/1771 for Yn+1
	    theVelocityBuffer[ c ] += theK6[ c ] * ( 512.0 / 1771.0 );
	    // k6 * 1/4 for ~Yn+1
	    theErrorEstimate[ c ] += theK6[ c ] * .25;

//  	    desiredError = eps_rel * ( a_y * fabs(theQuantityBuffer[ c ]) + a_dydt * getStepInterval() * fabs(theVelocityBuffer[ c ]) ) + eps_abs;
//  	    tmpError = fabs(theVelocityBuffer[ c ] - theErrorEstimate[ c ]) / desiredError;
	    
//  	    if( tmpError > maxError )
//  	      {
//  		maxError = tmpError;
		
//  		if( maxError > 1.1 )
//  		  {
//  		    // shrink it if the error exceeds 110%
//  		    setStepInterval( getStepInterval() * pow(maxError , -0.20) *  safety );

//    		    printf("Error: %e\nInterval: %lf\n", theVelocityBuffer[ c ]-theErrorEstimate[ c ], getStepInterval());
//  		    reset();
//  		    continue;
//  		  }
//  	      }

	    // restore x (original value)
	    aSubstance->loadQuantity( theQuantityBuffer[ c ] );

	    //// x(n+1) = x(n) + (k1 * 37/378 + k3 * 250/621 + k4 * 125/594 + k6 * 512/1771) + O(h^5)
	    aSubstance->setVelocity( theVelocityBuffer[ c ] );
	  }

	// grow it if error is 50% less than desired
//  	if (maxError <= 0.5)
//  	  {
//  	    aStepInterval = getStepInterval() * pow(maxError , -0.25) * safety;

//  	    if( aStepInterval >= getUserMaxInterval() )
//  	      {
//  		aStepInterval = getStepInterval();
//  	      }

//  	    printf("Error: %e\nInterval: %lf\n", maxError, aStepInterval);
	    
//  	    setNextStepInterval( aStepInterval );
//  	  }
//  	else setNextStepInterval( getStepInterval() );
	
	setNextStepInterval( getStepInterval() );

	break;
      }

    // don't call updateVelocityBuffer() -- it is already updated by
    // the algorithm.
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

