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


  inline void Stepper::updateVelocityBuffer()
  {
    const UnsignedInt aSize( theSubstanceCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	theVelocityBuffer[ c ] = aSubstance->getVelocity();
      }
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



} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
