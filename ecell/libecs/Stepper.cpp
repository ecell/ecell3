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
		  createPropertySlot( "ReadVariableList", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getReadVariableList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "WriteVariableList", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getWriteVariableList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ProcessList", *this,
				      Type2Type<Polymorph>(),
				      NULLPTR,
				      &Stepper::getProcessList ) );

  }

  Stepper::Stepper() 
    :
    theFirstNormalProcess( theProcessVector.begin() ),
    theModel( NULLPTR ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theUserMinInterval( std::numeric_limits<Real>::min() * 10 ),
    theUserMaxInterval( std::numeric_limits<Real>::max() * .1 )
  {
    makeSlots();
  }

  void Stepper::initialize()
  {
    FOR_ALL( SystemVector, theSystemVector, initialize );

    //    if( isEntityListChanged() )
    //      {

    //
    // update theProcessVector
    //
    theProcessVector.clear();
    for( SystemVectorConstIterator i( theSystemVector.begin() );
	 i != theSystemVector.end() ; ++i )
	{
	  const SystemCptr aSystem( *i );

	  for( ProcessMapConstIterator 
		 j( aSystem->getProcessMap().begin() );
	       j != aSystem->getProcessMap().end(); j++ )
	    {
	      ProcessPtr aProcessPtr( (*j).second );

	      theProcessVector.push_back( aProcessPtr );

	      aProcessPtr->initialize();
	    }
	}

    // sort by Process priority
    std::sort( theProcessVector.begin(), theProcessVector.end(),
	       Process::PriorityCompare() );

    // find boundary of negative and zero priority processes
    theFirstNormalProcess = 
      std::lower_bound( theProcessVector.begin(), theProcessVector.end(),	0,
			Process::PriorityCompare() );

    //
    // Update theWriteVariableVector
    //

    // (1) for each Variable which is included in the VariableReferenceVector
    //     of the Processes of this Stepper,
    // (2) if the Variable is mutable, 
    //           put it into theWriteVariableVector, 
    //       and register this Stepper to the StepperList of the Variable.
    // (3) if the Variable is accessible,
    //           puto it into theReadVariableVector
    theWriteVariableVector.clear();
    theReadVariableVector.clear();
    // for all the processs
    for( ProcessVectorConstIterator i( theProcessVector.begin());
	 i != theProcessVector.end() ; ++i )
      {
	VariableReferenceVectorCref 
	  aVariableReferenceVector( (*i)->getVariableReferenceVector() );

	// for all the VariableReferences
	for( VariableReferenceVectorConstIterator 
	       j( aVariableReferenceVector.begin() );
	     j != aVariableReferenceVector.end(); ++j )
	  {
	    VariableReferenceCref aVariableReference( *j );
	    VariablePtr aVariablePtr( aVariableReference.getVariable() );

	    if( aVariableReference.isMutator() )
	      {
		// prevent duplication

		if( std::find( theWriteVariableVector.begin(), 
			       theWriteVariableVector.end(),
			       aVariablePtr ) == theWriteVariableVector.end() )
		  {
		    
		    theWriteVariableVector.push_back( aVariablePtr );
		    
		    aVariablePtr->registerStepper( this );
		    // FIXME: this this needed?
		    aVariablePtr->initialize();
		  }
	      }

	    if( aVariableReference.isAccessor() )
	      {
		// prevent duplication

		if( std::find( theReadVariableVector.begin(), 
			       theReadVariableVector.end(),
			       aVariablePtr ) == theReadVariableVector.end() )
		  {
		    theReadVariableVector.push_back( aVariablePtr );
		  }
	      }

	  }
      }

    //    clearEntityListChanged();
    //      }

    Int aSize( theWriteVariableVector.size() );

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

  const Polymorph Stepper::getWriteVariableList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theWriteVariableVector.size() );
    
    for( VariableVectorConstIterator i( theWriteVariableVector.begin() );
	 i != theWriteVariableVector.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }

  const Polymorph Stepper::getReadVariableList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theReadVariableVector.size() );
    
    for( VariableVectorConstIterator i( theReadVariableVector.begin() );
	 i != theReadVariableVector.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }
  
  const Polymorph Stepper::getProcessList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theProcessVector.size() );
    
    for( ProcessVectorConstIterator i( theProcessVector.begin() );
	 i != theProcessVector.end() ; ++i )
      {
	aVector.push_back( (*i)->getFullID().getString() );
      }
    
    return aVector;
  }
  
  const UnsignedInt Stepper::findInWriteVariableVector( VariablePtr aVariable )
  {
    VariableVectorConstIterator
      anIterator( std::find( theWriteVariableVector.begin(), 
			     theWriteVariableVector.end(), aVariable ) );

    return anIterator - theWriteVariableVector.begin();
  }

  void Stepper::clear()
  {
    //
    // Variable::clear()
    //
    const UnsignedInt aSize( theWriteVariableVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theWriteVariableVector[ c ] );

	// save original value values
	theValueBuffer[ c ] = aVariable->saveValue();

	// clear phase is here!
	aVariable->clear();
      }


    //    FOR_ALL( WriteVariableVector, theWriteVariableVector, clear );
      
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
    FOR_ALL( ProcessVector, theProcessVector, process );
  }

  void Stepper::processNegative()
  {
    std::for_each( theProcessVector.begin(), theFirstNormalProcess, 
		   std::mem_fun( &Process::process ) );
  }

  void Stepper::processNormal()
  {
    std::for_each( theFirstNormalProcess, theProcessVector.end(),
		   std::mem_fun( &Process::process ) );
  }

  void Stepper::integrate()
  {
    //
    // Variable::integrate()
    //
    const UnsignedInt aSize( theWriteVariableVector.size() );

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theWriteVariableVector[ c ] );

	aVariable->integrate( getCurrentTime() );
      }
  }


  void Stepper::reset()
  {
    const UnsignedInt aSize( theWriteVariableVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theWriteVariableVector[ c ] );

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
    theTolerance( 1.0e-6 ),
    theAbsoluteToleranceFactor( 1.0 ),
    theStateToleranceFactor( 1.0 ),
    theDerivativeToleranceFactor( 1.0 ),
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
		  createPropertySlot( "Tolerance", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setTolerance,
				      &DifferentialStepper::getTolerance
				      ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "AbsoluteToleranceFactor", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setAbsoluteToleranceFactor,
				      &DifferentialStepper::getAbsoluteToleranceFactor
				      ) );
 
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "StateToleranceFactor", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setStateToleranceFactor,
				      &DifferentialStepper::getStateToleranceFactor
				      ) ); 

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "DerivativeToleranceFactor", *this,
				      Type2Type<Real>(),
				      &DifferentialStepper::setDerivativeToleranceFactor,
				      &DifferentialStepper::getDerivativeToleranceFactor
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

