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
#include "Variable.hpp"
#include "Process.hpp"
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
		  createPropertySlot( "VariableCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getVariableCache ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ProcessCache", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getProcessCache ) );

  }

  Stepper::Stepper() 
    :
    theFirstNormalProcess( theProcessCache.begin() ),
    theModel( NULLPTR ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theUserMinInterval( std::numeric_limits<Real>::min() * 10 ),
    theUserMaxInterval( std::numeric_limits<Real>::max() * .1 ),
    theSlaveStepper( NULLPTR )
  {
    makeSlots();
  }

  void Stepper::initialize()
  {
    FOR_ALL( SystemVector, theSystemVector, initialize );

    //    if( isEntityListChanged() )
    //      {

    //
    // update theProcessCache
    //
    theProcessCache.clear();
    for( SystemVectorConstIterator i( theSystemVector.begin() );
	 i != theSystemVector.end() ; ++i )
	{
	  const SystemCptr aSystem( *i );

	  for( ProcessMapConstIterator 
		 j( aSystem->getProcessMap().begin() );
	       j != aSystem->getProcessMap().end(); j++ )
	    {
	      ProcessPtr aProcessPtr( (*j).second );

	      theProcessCache.push_back( aProcessPtr );

	      aProcessPtr->initialize();
	    }
	}

    // sort by Process priority
    std::sort( theProcessCache.begin(), theProcessCache.end(),
	       Process::PriorityCompare() );

    // find boundary of negative and zero priority processes
    theFirstNormalProcess = 
      std::lower_bound( theProcessCache.begin(), theProcessCache.end(),	0,
			Process::PriorityCompare() );

    //
    // Update theVariableCache
    //

    // get all the variables which are included in the VariableReferenceVector
    // of the Processs
    theVariableCache.clear();
    // for all the processs
    for( ProcessVectorConstIterator i( theProcessCache.begin());
	 i != theProcessCache.end() ; ++i )
      {
	VariableReferenceVectorCref 
	  aVariableReferenceVector( (*i)->getVariableReferenceVector() );

	// for all the VariableReferences
	for( VariableReferenceVectorConstIterator 
	       j( aVariableReferenceVector.begin() );
	     j != aVariableReferenceVector.end(); ++j )
	  {
	    VariablePtr aVariablePtr( j->getVariable() );

	    // prevent duplication
	    if( std::find( theVariableCache.begin(), theVariableCache.end(),
			   aVariablePtr ) == theVariableCache.end() )
	      {
		theVariableCache.push_back( aVariablePtr );
		aVariablePtr->registerStepper( this );
		aVariablePtr->initialize();
	      }
	  }
      }

    //    clearEntityListChanged();
    //      }

    Int aSize( theVariableCache.size() );

    theValueBuffer.resize( aSize );
    theVelocityBuffer.resize( aSize );
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

  const Polymorph Stepper::getVariableCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableCache.size() );
    
    for( VariableVectorConstIterator i( theVariableCache.begin() );
	 i != theVariableCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }
  
  const Polymorph Stepper::getProcessCache() const
  {
    PolymorphVector aVector;
    aVector.reserve( theProcessCache.size() );
    
    for( ProcessVectorConstIterator i( theProcessCache.begin() );
	 i != theProcessCache.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }
  
  const UnsignedInt Stepper::findInVariableCache( VariablePtr aVariable )
  {
    VariableVectorConstIterator
      anIterator( std::find( theVariableCache.begin(), 
			     theVariableCache.end(), aVariable ) );

    return anIterator - theVariableCache.begin();
  }

  void Stepper::clear()
  {
    //
    // Variable::clear()
    //
    const UnsignedInt aSize( theVariableCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableCache[ c ] );

	// save original value values
	theValueBuffer[ c ] = aVariable->saveValue();

	// clear phase is here!
	aVariable->clear();
      }


    //    FOR_ALL( VariableCache, theVariableCache, clear );
      
    //
    // Process::clear() ?
    //
    //    FOR_ALL( ,, clear );
      
    //
    // System::clear() ?
    //
    //    FOR_ALL( ,, clear );
  }

  void Stepper::process()
  {
    //
    // Process::process()
    //
    FOR_ALL( ProcessVector, theProcessCache, process );
  }

  void Stepper::processNegative()
  {
    std::for_each( theProcessCache.begin(), theFirstNormalProcess, 
		   std::mem_fun( &Process::process ) );
  }

  void Stepper::processNormal()
  {
    std::for_each( theFirstNormalProcess, theProcessCache.end(),
		   std::mem_fun( &Process::process ) );
  }

  void Stepper::integrate()
  {
    //
    // Variable::integrate()
    //
    const UnsignedInt aSize( theVariableCache.size() );

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableCache[ c ] );

	aVariable->integrate( getCurrentTime() );
      }
  }

  void Stepper::slave()
  {
    // call slave
    StepperPtr aSlaveStepperPtr( getSlaveStepper() );
    if( aSlaveStepperPtr != NULLPTR )
      {
	aSlaveStepperPtr->step();
	aSlaveStepperPtr->setCurrentTime( getCurrentTime() );
      }
  }

  void Stepper::reset()
  {
    const UnsignedInt aSize( theVariableCache.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableCache[ c ] );

	// restore x (original value)
	aVariable->setValue( theValueBuffer[ c ] );

	// clear velocity
	theVelocityBuffer[ c ] = 0.0;
	aVariable->setVelocity( 0.0 );
      }
  }


  ////////////////////////// DifferentialStepper


  DifferentialStepper::DifferentialStepper()
    :
    theRelativeTorelance( 1.0e-6 ),
    theAbsoluteTorelance( 1.0e-6 ),
    theStateScalingFactor( 1.0 ),
    theDerivativeScalingFactor( 1.0 ),
    safety( 0.9 ),
    theTolerantStepInterval( 0.001 ),
    theNextStepInterval( 0.001 )
  {
    makeSlots();
  }

  void DifferentialStepper::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "StepInterval", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::initializeStepInterval,
				      &DifferentialStepper::getStepInterval 
				      ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "RelativeTorelance", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setRelativeTorelance,
				      &DifferentialStepper::getRelativeTorelance
				      ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "AbsoluteTorelance", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setAbsoluteTorelance,
				      &DifferentialStepper::getAbsoluteTorelance
				      ) );
 
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "StateScalingFactor", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setStateScalingFactor,
				      &DifferentialStepper::getStateScalingFactor
				      ) ); 

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "DerivativeScalingFactor", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setDerivativeScalingFactor,
				      &DifferentialStepper::getDerivativeScalingFactor
				      ) ); 
  }

  void DifferentialStepper::initialize()
  {
    Stepper::initialize();

    // should create another method for property slot ?
    //    setNextStepInterval( getStepInterval() );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

