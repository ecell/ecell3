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
#include "VariableProxy.hpp"
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

    DEFINE_PROPERTYSLOT( "ID", String,
			 NULLPTR,
			 &Stepper::getID );

    DEFINE_PROPERTYSLOT( "CurrentTime", Real,
			 NULLPTR,
			 &Stepper::getCurrentTime );

    DEFINE_PROPERTYSLOT( "StepInterval", Real,
			 &Stepper::setStepInterval,
			 &Stepper::getStepInterval );

    DEFINE_PROPERTYSLOT( "UserMaxInterval", Real,
			 &Stepper::setUserMaxInterval,
			 &Stepper::getUserMaxInterval );

    DEFINE_PROPERTYSLOT( "UserMinInterval", Real,
			 &Stepper::setUserMinInterval,
			 &Stepper::getUserMinInterval );

    DEFINE_PROPERTYSLOT( "MaxInterval", Real,
			 NULLPTR,
			 &Stepper::getMaxInterval );

    DEFINE_PROPERTYSLOT( "MinInterval", Real,
			 NULLPTR,
			 &Stepper::getMinInterval );

    //    DEFINE_PROPERTYSLOT( "StepIntervalConstraint", Polymorph,
    //			 &Stepper::setStepIntervalConstraint,
    //			 &Stepper::getStepIntervalConstraint );

    DEFINE_PROPERTYSLOT( "ReadVariableList", Polymorph,
			 NULLPTR,
			 &Stepper::getReadVariableList );

    DEFINE_PROPERTYSLOT( "WriteVariableList", Polymorph,
			 NULLPTR,
			 &Stepper::getWriteVariableList );

    DEFINE_PROPERTYSLOT( "ProcessList", Polymorph,
			 NULLPTR,
			 &Stepper::getProcessList );

    DEFINE_PROPERTYSLOT( "SystemList", Polymorph,
			 NULLPTR,
			 &Stepper::getSystemList );

    DEFINE_PROPERTYSLOT( "DependentStepperList", Polymorph,
			 NULLPTR,
			 &Stepper::getDependentStepperList );

  }

  Stepper::Stepper() 
    :
    theModel( NULLPTR ),
    theSchedulerIndex( -1 ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theUserMinInterval( std::numeric_limits<Real>::min() * 10 ),
    theUserMaxInterval( std::numeric_limits<Real>::max() * .1 )
  {
    makeSlots();
  }

  void Stepper::initialize()
  {
    //    if( isEntityListChanged() )
    //      {

    //
    // update theProcessVector
    //
    //    updateProcessVector();


    //
    // Update theWriteVariableVector and theReadVariableVector
    //
    updateVariableVectors();


    //    clearEntityListChanged();
    //      }

    const Int aSize( theVariableProxyVector.size() );

    theValueBuffer.resize( aSize );

    updateLoggedPropertySlotVector();
  }

 
  void Stepper::updateProcessVector()
  {
    /*
    theProcessVector.clear();
    for( SystemVectorConstIterator i( theSystemVector.begin() );
	 i != theSystemVector.end() ; ++i )
	{
	  const SystemCptr aSystem( *i );

	  for( ProcessMapConstIterator 
		 j( aSystem->getProcessMap().begin() );
	       j != aSystem->getProcessMap().end(); ++j )
	    {
	      ProcessPtr aProcessPtr( (*j).second );

	      theProcessVector.push_back( aProcessPtr );
	    }
	}
    */

    // sort by memory address. this is an optimization.
    std::sort( theProcessVector.begin(), theProcessVector.end() );

    // sort by Process priority
    std::stable_sort( theProcessVector.begin(), theProcessVector.end(),
		      Process::PriorityCompare() );

  }

  void Stepper::updateVariableVectors()
  {
    // (1) for each Variable which is included in the VariableReferenceVector
    //     of the Processes of this Stepper,
    // (2) if the Variable is mutable, 
    //           put it into theVariableProxyVector, 
    //       and register this Stepper to the StepperList of the Variable.
    // (3) if the Variable is accessible,
    //           puto it into theReadVariableVector
    // (4) sort theReadVariableVector and theProxyVariableVector by 
    //     memory address.
    theReadVariableVector.clear();
    // for all the processs

    VariableVector aVariableVector;
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

		if( std::find( aVariableVector.begin(), 
			       aVariableVector.end(),
			       aVariablePtr ) == aVariableVector.end() )
		  {
		    aVariableVector.push_back( aVariablePtr );
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

    // sort by memory address. this is an optimization.
    std::sort( theReadVariableVector.begin(), theReadVariableVector.end() );
    std::sort( aVariableVector.begin(), aVariableVector.end() );


    theVariableProxyVector.clear();
    for( VariableProxyVectorIterator i( theVariableProxyVector.begin() );
	 i != theVariableProxyVector.end(); ++i )
      {
	VariableProxyPtr anVariableProxyPtr( *i );
	delete anVariableProxyPtr;
      }

    // create VariableProxies.
    for( VariableVectorIterator i( aVariableVector.begin() );
	 i != aVariableVector.end(); ++i )
      {
	VariablePtr aVariablePtr( *i );
	theVariableProxyVector.
	  push_back( createVariableProxy( aVariablePtr ) );
      }

  }


  void Stepper::updateLoggedPropertySlotVector()
  {
    theLoggedPropertySlotVector.clear();

    EntityVector anEntityVector;
    anEntityVector.reserve( theProcessVector.size() + 
			    theVariableProxyVector.size() +
			    theSystemVector.size() );

    std::copy( theProcessVector.begin(), theProcessVector.end(),
	       std::back_inserter( anEntityVector ) );

    for( VariableProxyVectorConstIterator i( theVariableProxyVector.begin() );
	 i != theVariableProxyVector.end(); ++i )
      {
	anEntityVector.push_back( (*i)->getVariable() );
      }
		   
    for( EntityVectorConstIterator i( anEntityVector.begin() );
	 i != anEntityVector.end() ; ++i )
      {
	PropertySlotMapCref aPropertySlotMap( (*i)->getPropertySlotMap() );
	for( PropertySlotMapConstIterator j( aPropertySlotMap.begin() );
	     j != aPropertySlotMap.end(); ++j )
	  {
	    PropertySlotPtr aPropertySlotPtr( j->second );
	    if( aPropertySlotPtr->isLogged() )
	      {
		theLoggedPropertySlotVector.push_back( aPropertySlotPtr );
	      }
	  }

      }


  }

  void Stepper::updateDependentStepperVector()
  {
    theDependentStepperVector.clear();

    StepperMapCref aStepperMap( getModel()->getStepperMap() );

    for( StepperMapConstIterator i( aStepperMap.begin() );
	 i != aStepperMap.end(); ++i )
      {
	StepperPtr aStepperPtr( i->second );

	// exclude this
	if( aStepperPtr == this )
	  {
	    continue;
	  }

	VariableVectorCref aTargetVector( aStepperPtr->
					  getReadVariableVector() );

	// This search assumes both vectors are sorted by pointer addresses.
	//
	// For efficiency, binary_search should be done for possibly longer
	// vector, and linear iteration for possibly shorter vector.
	//
	for( VariableProxyVectorConstIterator j( theVariableProxyVector.begin() );
	     j != theVariableProxyVector.end(); ++j )
	  {
	    VariablePtr aVariablePtr( (*j)->getVariable() );

	    if( std::binary_search( aTargetVector.begin(), aTargetVector.end(),
				    aVariablePtr ) )
	      {
		theDependentStepperVector.push_back( aStepperPtr );
		break;
	      }
	  }
      }
    
    // optimization: sort by memory address.
    std::sort( theDependentStepperVector.begin(), 
	       theDependentStepperVector.end() );

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


  const Polymorph Stepper::getDependentStepperList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theDependentStepperVector.size() );

    for( StepperVectorConstIterator i( getDependentStepperVector().begin() );
	 i != getDependentStepperVector().end() ; ++i )
      {
	StepperCptr aStepperPtr( *i );

	aVector.push_back( aStepperPtr->getID() );
      }

    return aVector;
  }


  void Stepper::registerSystem( SystemPtr aSystemPtr )
  { 
    if( std::find( theSystemVector.begin(), theSystemVector.end(), aSystemPtr )
	== theSystemVector.end() )
      {
   	theSystemVector.push_back( aSystemPtr );
      }
  }

  void Stepper::removeSystem( SystemPtr aSystemPtr )
  { 
    SystemVectorIterator i( find( theSystemVector.begin(), 
				  theSystemVector.end(),
				  aSystemPtr ) );
    
    if( i == theSystemVector.end() )
      {
	THROW_EXCEPTION( NotFound,
			 getClassName() + String( ": " ) 
			 + getID() + ": " 
			 + aSystemPtr->getFullID().getString() 
			 + " not found in this stepper. Can't remove." );
      }

    theSystemVector.erase( i );
  }


  void Stepper::registerProcess( ProcessPtr aProcessPtr )
  { 
    if( std::find( theProcessVector.begin(), theProcessVector.end(), 
		   aProcessPtr ) == theProcessVector.end() )
      {
   	theProcessVector.push_back( aProcessPtr );
      }

    updateProcessVector();
  }

  void Stepper::removeProcess( ProcessPtr aProcessPtr )
  { 
    ProcessVectorIterator i( find( theProcessVector.begin(), 
				   theProcessVector.end(),
				   aProcessPtr ) );
    
    if( i == theProcessVector.end() )
      {
	THROW_EXCEPTION( NotFound,
			 getClassName() + String( ": " ) 
			 + getID() + ": " 
			 + aProcessPtr->getFullID().getString() 
			 + " not found in this stepper. Can't remove." );
      }

    theProcessVector.erase( i );
  }


  void Stepper::registerLoggedPropertySlot( PropertySlotPtr aPropertySlotPtr )
  {
    theLoggedPropertySlotVector.push_back( aPropertySlotPtr );
  }


  /*
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
  */

  const Real Stepper::getMaxInterval() const
  {
    Real aMaxInterval( getUserMaxInterval() );

    /*
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
    */

    return aMaxInterval;
  }
  
  void Stepper::log()
  {
    // update loggers
    std::for_each( theLoggedPropertySlotVector.begin(), 
		   theLoggedPropertySlotVector.end(),
		   std::bind2nd( std::mem_fun( &PropertySlot::log ), 
				 getCurrentTime() ) );
  }

  const Polymorph Stepper::getWriteVariableList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableProxyVector.size() );
    
    for( VariableProxyVectorConstIterator i( theVariableProxyVector.begin() );
	 i != theVariableProxyVector.end() ; ++i )
      {
	aVector.push_back( (*i)->getVariable()->getFullID().getString() );
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
  
  const UnsignedInt 
  Stepper::getVariableProxyIndex( VariableCptr const aVariable )
  {
    VariableProxyVectorConstIterator
      anIterator( std::lower_bound( theVariableProxyVector.begin(), 
				    theVariableProxyVector.end(), 
				    aVariable, 
				    VariableProxy::VariablePtrCompare() ) );

    // This may be called before push_back() function.
//     DEBUG_EXCEPTION( (*anIterator)->getVariable() == aVariable , NotFound, 
// 		     "This should not occur.  Must be a bug." );

    return anIterator - theVariableProxyVector.begin();
  }


  void Stepper::clear()
  {
    //
    // Variable::clear()
    //
    const UnsignedInt aSize( theVariableProxyVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableProxyVector[ c ]->getVariable() );

	// save original value values
	theValueBuffer[ c ] = aVariable->getValue();

	// clear phase is here!
	aVariable->clear();
      }

    //    FOR_ALL( VariableProxyVector, theVariableProxyVector, clear );
  }

  void Stepper::process()
  {
    std::for_each( theProcessVector.begin(), theProcessVector.end(),
		   std::mem_fun( &Process::process ) );
  }


  void Stepper::integrate()
  {
    //
    // Variable::integrate()
    //
    std::for_each( theVariableProxyVector.begin(),
		   theVariableProxyVector.end(), 
		   std::bind2nd( std::mem_fun( &VariableProxy::integrate ),
				 getCurrentTime() ) );
  }


  void Stepper::reset()
  {
    const UnsignedInt aSize( theVariableProxyVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableProxyVector[ c ]->
				     getVariable() );

	// restore x (original value) and clear velocity
	//FIXME: should go into VariableProxy:::reset()?
	aVariable->setValue( theValueBuffer[ c ] );
	aVariable->setVelocity( 0.0 );
      }
  }


  void Stepper::dispatchInterruptions()
  {
    std::for_each( theDependentStepperVector.begin(),
		   theDependentStepperVector.end(),
		   std::bind2nd( std::mem_fun( &Stepper::interrupt ), this ) );
  }

  void Stepper::interrupt( StepperPtr const )
  {
    ; // do nothing
  }



  ////////////////////////// DifferentialStepper


  DifferentialStepper::DifferentialStepper()
    :
    theTolerance( 1.0e-6 ),
    theAbsoluteToleranceFactor( 1.0 ),
    theStateToleranceFactor( 1.0 ),
    theDerivativeToleranceFactor( 1.0 ),
    safety( 0.9 ),
    theMaxErrorRatio( 1.0 ),
    theTolerantStepInterval( 0.001 ),
    theNextStepInterval( 0.001 )
  {
    makeSlots();
  }

  void DifferentialStepper::makeSlots()
  {
    DEFINE_PROPERTYSLOT( "StepInterval", Real, 
			 &DifferentialStepper::initializeStepInterval,
			 &DifferentialStepper::getStepInterval );

    DEFINE_PROPERTYSLOT( "Tolerance", Real,
			 &DifferentialStepper::setTolerance,
			 &DifferentialStepper::getTolerance );

    DEFINE_PROPERTYSLOT( "AbsoluteToleranceFactor", Real,
			 &DifferentialStepper::setAbsoluteToleranceFactor,
			 &DifferentialStepper::getAbsoluteToleranceFactor );
 
    DEFINE_PROPERTYSLOT( "StateToleranceFactor", Real,
			 &DifferentialStepper::setStateToleranceFactor,
			 &DifferentialStepper::getStateToleranceFactor );

    DEFINE_PROPERTYSLOT( "DerivativeToleranceFactor", Real,
			 &DifferentialStepper::setDerivativeToleranceFactor,
			 &DifferentialStepper::getDerivativeToleranceFactor );

    DEFINE_PROPERTYSLOT( "MaxErrorRatio", Real,
			 NULLPTR,
			 &DifferentialStepper::getMaxErrorRatio );
  }

  void DifferentialStepper::initialize()
  {
    Stepper::initialize();

    theVelocityBuffer.resize( theVariableProxyVector.size() );

    // should create another method for property slot ?
    //    setNextStepInterval( getStepInterval() );
  }


  void DifferentialStepper::reset()
  {
    // clear velocity buffer
    theVelocityBuffer.assign( theVelocityBuffer.size(), 0.0 );

    Stepper::reset();
  }



  #if 0
  const bool DifferentialStepper::checkExternalError() const
  {
    const Real aSize( theValueVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {

	// no!! theValueVector and theReadVariableVector holds completely
	// different sets of Variables
	const Real anOriginalValue( theValueVector[ c ] + 
				    numeric_limit<Real>::epsilon() );

	const Real aCurrentValue( theReadVariableVector[ c ]->getValue() );

	const Real anRelativeError( fabs( aCurrentValue - anOriginalValue ) 
				    / anOriginalValue );
	if( anRelativeError <= aTolerance )
	  {
	    continue;
	  }

	return false;
      }

    return true;
  }
#endif /* 0 */

  void DifferentialStepper::interrupt( StepperPtr const aCaller )
  {
    const Real aCallerTimeScale( aCaller->getTimeScale() );
    const Real aStepInterval   ( getStepInterval() );

    // If the step size of this is less than caller's,
    // ignore this interruption.
    if( aCallerTimeScale >= aStepInterval )
      {
	return;
      }

    // if all Variables didn't change its value more than 10%,
    // ignore this interruption.
    /*  !!! currently this cannot be used
    if( checkExternalError() )
      {
	return;
      }
    */

    // Shrink the next step size to that of caller's
    setNextStepInterval( aCallerTimeScale );

    const Real aCurrentTime      ( getCurrentTime() );
    const Real aNextStep         ( aCurrentTime + aStepInterval );
    const Real aCallerCurrentTime( aCaller->getCurrentTime() );
    const Real aCallerNextStep   ( aCallerCurrentTime + aCallerTimeScale );

    // If the next step of this occurs *before* the next step of the caller,
    // just shrink step size of this Stepper.
    if( aNextStep < aCallerNextStep )
      {
	return;
      }

    // If the next step of this will occur *after* the caller,
    // reschedule this Stepper, as well as shrinking the next step size.
    //    setStepInterval( aCallerCurrentTime + ( aCallerTimeScale * 0.5 ) 
    //		     - aCurrentTime );
    setStepInterval( aCallerCurrentTime - aCurrentTime );

    getModel()->reschedule( this );
  }


  //////////////////// DiscreteEventStepper

  void DiscreteEventStepper::dispatchInterruptions()
  {
    Stepper::dispatchInterruptions();
  }


  //////////////////// DiscreteTimeStepper

  DiscreteTimeStepper::DiscreteTimeStepper()
  {
    ; // do nothing -- no additional property slots
  }

  void DiscreteTimeStepper::step()
  {
    process();
  }


  ////////////////////// SlaveStepper

  SlaveStepper::SlaveStepper()
  {
    // gcc3 doesn't currently support numeric_limits::infinity.
    // using max() instead.
    const Real anInfinity( std::numeric_limits<Real>::max() *
			   std::numeric_limits<Real>::max() );
    setStepInterval( anInfinity );
  }

  void SlaveStepper::initialize()
  {
    Stepper::initialize();

    // Make sure this never scheduled by the scheduler.
    getModel()->reschedule( this );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

