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

#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>


#include "Util.hpp"
#include "Substance.hpp"
#include "Reactor.hpp"
#include "Model.hpp"
#include "FullID.hpp"
#include "PropertySlotMaker.hpp"

// to be removed (for SRM)
#include "Integrators.hpp"

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
		  createPropertySlot( "StepsPerSecond", *this,
				      Type2Type<Real>(),
				      NULLPTR,
				      &Stepper::getStepsPerSecond ) );

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


  }

  Stepper::Stepper() 
    :
    theModel( NULLPTR ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theUserMinInterval( 0.0 ),
    theUserMaxInterval( std::numeric_limits<Real>::max() ),
    theEntityListChanged( true )
  {
    makeSlots();
    calculateStepsPerSecond();
  }

  void Stepper::initialize()
  {
    FOR_ALL( SystemVector, theSystemVector, initialize );
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

  void Stepper::setStepInterval( RealCref aStepInterval )
  {
    theStepInterval = aStepInterval;
    calculateStepsPerSecond();
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
  
  void Stepper::push()
  {
    // update loggers
    FOR_ALL( PropertySlotVector, theLoggedPropertySlotVector, updateLogger );
  }


  ////////////////////////// SRMStepper

  SRMStepper::SRMStepper()
  {
    makeSlots();
  }

  
  void SRMStepper::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SubstanceCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &SRMStepper::getSubstanceCache ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ReactorCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &SRMStepper::getReactorCache ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "RuleReactorCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &SRMStepper::getRuleReactorCache ) );
  }


  const Polymorph SRMStepper::getSubstanceCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theSubstanceCache.size() );
    
    for( SubstanceCache::const_iterator i( theSubstanceCache.begin() );
	 i != theSubstanceCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getID() );
      }
    
    return aVector;
  }
  
  const Polymorph SRMStepper::getReactorCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theReactorCache.size() );
    
    for( ReactorCache::const_iterator i( theReactorCache.begin() );
	 i != theReactorCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getID() );
      }
    
    return aVector;
  }
  
  const Polymorph SRMStepper::getRuleReactorCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theReactorCache.size() );
    
    for( ReactorCache::const_iterator i( theRuleReactorCache.begin() );
	 i != theRuleReactorCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getID() );
      }
    
    return aVector;
  }
  

  void SRMStepper::initialize()
  {
    Stepper::initialize();

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

	      SRMReactorPtr aSRMReactorPtr( dynamic_cast<SRMReactorPtr>
					    ( aReactorPtr ) );

	      if( aSRMReactorPtr != NULLPTR )
		{
		  theReactorCache.push_back( aSRMReactorPtr );
		}
	    }
	}

    // sort by Reactor priority
    std::sort( theReactorCache.begin(), theReactorCache.end(),
	       SRMReactor::PriorityCompare() );


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
	      }
	  }
      }


    //
    // Update theRuleReactorCache
    //

    // move RuleReactors from theReactorCache to theRuleReactorCache
    theRuleReactorCache.clear();
    std::remove_copy_if( theReactorCache.begin(), theReactorCache.end(),
			 std::back_inserter( theRuleReactorCache ), 
			 not1( RuleSRMReactor::IsRuleReactor() ) );
    std::remove_if( theReactorCache.begin(), theReactorCache.end(),
		    RuleSRMReactor::IsRuleReactor() );

    
    //    clearEntityListChanged();
    //      }
    
  }


  inline void SRMStepper::clear()
  {
    //
    // Substance::clear()
    //
    FOR_ALL( SubstanceCache, theSubstanceCache, clear );
      
    //
    // Reactor::clear() ?
    //
    //FOR_ALL( ,, clear );
      
    //
    // System::clear() ?
    //
    //FOR_ALL( ,, clear );
  }

  inline void SRMStepper::react()
  {
    //
    // Reactor::react()
    //
    FOR_ALL( ReactorCache, theReactorCache, react );

  }

  inline void SRMStepper::integrate()
  {
    //
    // Substance::integrate()
    //
    FOR_ALL( SubstanceCache, theSubstanceCache, integrate );
  }

  inline void SRMStepper::rule()
  {
    //
    // Reactor::react() of RuleReactors
    //
    FOR_ALL( ReactorCache, theRuleReactorCache, react );
  }


  void SRMStepper::step()
  {
    clear();
    react();
    integrate();
    rule();
    
    Stepper::step();
  }


  ////////////////////////// Euler1SRMStepper

  Euler1SRMStepper::Euler1SRMStepper()
  {
    ; // do nothing
  }

  ////////////////////////// RungeKutta4SRMStepper

  RungeKutta4SRMStepper::RungeKutta4SRMStepper()
  {
    ; // do nothing
  }

  void RungeKutta4SRMStepper::initialize()
  {
    SRMStepper::initialize();

    Int aSize( theSubstanceCache.size() );

    theQuantityBuffer.resize( aSize );
    theK.resize( aSize );
  }
  

  void RungeKutta4SRMStepper::step()
  {
    const UnsignedInt aSize( theSubstanceCache.size() );

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	// save original quantity values
	theQuantityBuffer[ c ] = aSubstance->saveQuantity();

	// clear phase is here!
	aSubstance->clear();
      }

    // ========= 1 ===========
    react();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );

	// get k1
	Real aVelocity( aSubstance->getVelocity() );

	// restore k1 / 2 + x
	aSubstance->loadQuantity( aVelocity * .5 + theQuantityBuffer[ c ] );

	theK[ c ] = aVelocity;

	// clear velocity
	aSubstance->setVelocity( 0 );
      }

    // ========= 2 ===========
    react();


    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	const Real aVelocity( aSubstance->getVelocity() );
	theK[ c ] += aVelocity + aVelocity;

	// restore k2 / 2 + x
	aSubstance->loadQuantity( aVelocity * .5 + theQuantityBuffer[ c ] );


	// clear velocity
	aSubstance->setVelocity( 0 );
      }


    // ========= 3 ===========
    react();

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	SubstancePtr const aSubstance( theSubstanceCache[ c ] );
	const Real aVelocity( aSubstance->getVelocity() );
	theK[ c ] += aVelocity + aVelocity;

	// restore k3 + x
	aSubstance->loadQuantity( aVelocity + theQuantityBuffer[ c ] );


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

	aSubstance->setVelocity( ( theK[ c ] + aVelocity ) * ( 1.0 / 6.0 ) );

	// integrate phase here!!
	aSubstance->integrate();
      }
    
    rule();
    
    Stepper::step();
  }



} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
