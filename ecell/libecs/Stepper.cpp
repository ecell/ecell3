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

    DEFINE_PROPERTYSLOT( "CurrentTime", Real,
			 NULLPTR,
			 &Stepper::getCurrentTime );

    DEFINE_PROPERTYSLOT( "StepInterval", Real,
			 &Stepper::setStepInterval,
			 &Stepper::getStepInterval );

    DEFINE_PROPERTYSLOT( "OriginalStepInterval", Real,
			 &Stepper::setOriginalStepInterval,
			 &Stepper::getOriginalStepInterval );

    DEFINE_PROPERTYSLOT( "MaxInterval", Real,
			 &Stepper::setMaxInterval,
			 &Stepper::getMaxInterval );

    DEFINE_PROPERTYSLOT( "MinInterval", Real,
			 &Stepper::setMinInterval,
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
    theReadWriteVariableOffset( 0 ),
    theReadOnlyVariableOffset( 0 ),
    theModel( NULLPTR ),
    theSchedulerIndex( -1 ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theOriginalStepInterval( 0.001 ),
    theMinInterval( std::numeric_limits<Real>::min() * 10 ),
    theMaxInterval( std::numeric_limits<Real>::max() * .1 )
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
    // Update theVariableVector.  This also calls updateVariableProxyVector.
    //
    updateVariableVector();

    createVariableProxies();


    //    clearEntityListChanged();
    //      }

    // size of the value buffer == the number of *all* variables.
    // (not just read or write variables)
    theValueBuffer.resize( theVariableVector.size() );

    updateLoggedPropertySlotVector();
  }

 
  void Stepper::updateProcessVector()
  {
    // sort by memory address. this is an optimization.
    std::sort( theProcessVector.begin(), theProcessVector.end() );

    // sort by Process priority, conserving the partial order in memory
    std::stable_sort( theProcessVector.begin(), theProcessVector.end(),
		      Process::PriorityCompare() );

  }

  void Stepper::updateVariableVector()
  {

    DECLARE_MAP( VariablePtr, VariableReference, std::less<VariablePtr>,
		 PtrVariableReferenceMap );

    PtrVariableReferenceMap aPtrVariableReferenceMap;

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
	    VariableReferenceCref aNewVariableReference( *j );
	    VariablePtr aVariablePtr( aNewVariableReference.getVariable() );

	    PtrVariableReferenceMapIterator 
	      anIterator( aPtrVariableReferenceMap.find( aVariablePtr ) );

	    if( anIterator == aPtrVariableReferenceMap.end() )
	      {
		aPtrVariableReferenceMap.
		  insert( PtrVariableReferenceMap::
			  value_type( aVariablePtr, aNewVariableReference ) );
	      }
	    else
	      {
		VariableReferenceRef aVariableReference( anIterator->second );

		aVariableReference.
		  setIsAccessor( aVariableReference.isAccessor() ||
				 aNewVariableReference.isAccessor() );
		
		aVariableReference.
		  setCoefficient( abs( aVariableReference.getCoefficient() )
				  + abs( aNewVariableReference.
					 getCoefficient() ) );
	      }
	  }
      }

    VariableReferenceVector aVariableReferenceVector;
    aVariableReferenceVector.reserve( aPtrVariableReferenceMap.size() );

    // I want select2nd... but use a for loop for portability.
    for( PtrVariableReferenceMapConstIterator 
	   i( aPtrVariableReferenceMap.begin() );
	 i != aPtrVariableReferenceMap.end() ; ++i )
      {
	aVariableReferenceVector.push_back( i->second );
      }
    
    VariableReferenceVectorIterator aReadOnlyVariableReferenceIterator = 
      std::partition( aVariableReferenceVector.begin(),
		      aVariableReferenceVector.end(),
		      std::mem_fun_ref( &VariableReference::isMutator ) );

    VariableReferenceVectorIterator aReadWriteVariableReferenceIterator = 
      std::partition( aVariableReferenceVector.begin(),
		      aReadOnlyVariableReferenceIterator,
		      std::not1
		      ( std::mem_fun_ref( &VariableReference::isAccessor ) )
		      );

    theVariableVector.clear();
    theVariableVector.reserve( aVariableReferenceVector.size() );

    std::transform( aVariableReferenceVector.begin(),
		    aVariableReferenceVector.end(),
		    std::back_inserter( theVariableVector ),
		    std::mem_fun_ref( &VariableReference::getVariable ) );

    theReadWriteVariableOffset = aReadWriteVariableReferenceIterator 
      - aVariableReferenceVector.begin();

    theReadOnlyVariableOffset = aReadOnlyVariableReferenceIterator 
      - aVariableReferenceVector.begin();

    // For each part of the vector, sort by memory address. 
    // This is an optimization.
    VariableVectorIterator aReadWriteVariableIterator = 
      theVariableVector.begin() + theReadWriteVariableOffset;
    VariableVectorIterator aReadOnlyVariableIterator = 
      theVariableVector.begin() + theReadOnlyVariableOffset;

    std::sort( theVariableVector.begin(),  aReadWriteVariableIterator );
    std::sort( aReadWriteVariableIterator, aReadOnlyVariableIterator );
    std::sort( aReadOnlyVariableIterator,  theVariableVector.end() );
  }


  void Stepper::createVariableProxies()
  {
    // create VariableProxies.
    for( UnsignedInt c( 0 );  c != theReadOnlyVariableOffset; ++c )
      {
	VariablePtr aVariablePtr( theVariableVector[ c ] );
	aVariablePtr->registerProxy( createVariableProxy( aVariablePtr ) );
      }
  }


  void Stepper::updateLoggedPropertySlotVector()
  {
    theLoggedPropertySlotVector.clear();

    EntityVector anEntityVector;
    anEntityVector.reserve( theProcessVector.size() + 
			    getReadOnlyVariableOffset() +
			    theSystemVector.size() );

    // copy theProcessVector
    std::copy( theProcessVector.begin(), theProcessVector.end(),
	       std::back_inserter( anEntityVector ) );

    // append theVariableVector
    std::copy( theVariableVector.begin(), 
	       theVariableVector.begin() + theReadOnlyVariableOffset,
	       std::back_inserter( anEntityVector ) );

    // append theSystemVector
    std::copy( theSystemVector.begin(), theSystemVector.end(),
	       std::back_inserter( anEntityVector ) );
		   

    // Scan all the relevant Entities, and find logged PropertySlots.
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

	VariableVectorCref aTargetVector( aStepperPtr->getVariableVector() );

	VariableVectorConstIterator aReadWriteTargetVariableIterator
	  ( aTargetVector.begin() + 
	    aStepperPtr->getReadWriteVariableOffset() );

	VariableVectorConstIterator aReadOnlyTargetVariableIterator
	  ( aTargetVector.begin() +
	    aStepperPtr->getReadOnlyVariableOffset() );

	// For efficiency, binary_search should be done for possibly longer
	// vector, and linear iteration for possibly shorter vector.
	//

	// if one Variable in this::readlist appears in the target::write list
	for( UnsignedInt c( 0 ); c != theReadOnlyVariableOffset; ++c )
	  {
	    VariablePtr const aVariablePtr( theVariableVector[ c ] );

	    // search in target::readwrite and target::read list.
	    if( std::binary_search( aReadWriteTargetVariableIterator,
				    aReadOnlyTargetVariableIterator,
				    aVariablePtr ) ||
		std::binary_search( aReadOnlyTargetVariableIterator,
				    aTargetVector.end(),
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
    aVector.reserve( theVariableVector.size() );

    for( UnsignedInt c( 0 ); c != theReadOnlyVariableOffset; ++c )
      {
	aVector.push_back( theVariableVector[c]->getFullID().getString() );
      }
    
    return aVector;
  }

  const Polymorph Stepper::getReadVariableList() const
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableVector.size() );
    
    for( UnsignedInt c( theReadWriteVariableOffset ); 
	 c != theVariableVector.size(); ++c )
      {
	aVector.push_back( theVariableVector[c]->getFullID().getString() );
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
  
  const UnsignedInt Stepper::getVariableIndex( VariableCptr const aVariable )
  {
    VariableVectorConstIterator
      anIterator( std::find( theVariableVector.begin(), 
			     theVariableVector.end(), 
			     aVariable ) ); 

    DEBUG_EXCEPTION( anIterator != theVariableVector.end() , NotFound, 
 		     "Variable not found." );

    return anIterator - theVariableVector.begin();
  }


  void Stepper::clear()
  {
    //
    // Variable::clear()
    //
    const UnsignedInt aSize( theVariableVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// save original value values
	theValueBuffer[ c ] = aVariable->getValue();

	// clear phase is here!
	aVariable->clear();
      }

  }

  void Stepper::process()
  {
    std::for_each( theProcessVector.begin(), theProcessVector.end(),
    		   std::mem_fun( &Process::process ) );
  }


  void Stepper::integrate( const Real aTime )
  {
    //
    // Variable::integrate()
    //
    std::for_each( theVariableVector.begin(), theVariableVector.end(), 
		   std::bind2nd( std::mem_fun( &Variable::integrate ),
				 aTime ) );

    setCurrentTime( aTime );
  }


  void Stepper::reset()
  {
    // restore original values and clear velocity of all the *write* variables.
    for( UnsignedInt c( 0 ); c < getReadOnlyVariableOffset(); ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// restore x (original value) and clear velocity
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

    DEFINE_PROPERTYSLOT( "NextStepInterval", Real, 
			 NULLPTR,
			 &DifferentialStepper::getNextStepInterval );
  }

  void DifferentialStepper::initialize()
  {
    Stepper::initialize();

    // size of the velocity buffer == the number of *write* variables.
    theVelocityBuffer.resize( getReadOnlyVariableOffset() );

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

	// no!! theValueVector and theVariableVector holds completely
	// different sets of Variables
	const Real anOriginalValue( theValueVector[ c ] + 
				    numeric_limit<Real>::epsilon() );

	const Real aCurrentValue( theVariableVector[ c ]->getValue() );

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

    // If the step size of this is less than caller's timescale,
    // ignore this interruption.
    if( aCallerTimeScale >= aStepInterval )
      {
	return;
      }

    // if all Variables didn't change its value more than 10%,
    // ignore this interruption.
    /*  !!! currently this is disabled
    if( checkExternalError() )
      {
	return;
      }
    */
	
    const Real aCurrentTime      ( getCurrentTime() );
    const Real aCallerCurrentTime( aCaller->getCurrentTime() );

    // aCallerTimeScale == 0 implies need for immediate reset
    if( aCallerTimeScale != 0.0 )
      {
	// Shrink the next step size to that of caller's
	setNextStepInterval( aCallerTimeScale );

	const Real aNextStep      ( aCurrentTime + aStepInterval );
	const Real aCallerNextStep( aCallerCurrentTime + aCallerTimeScale );

	// If the next step of this occurs *before* the next step 
	// of the caller, just shrink step size of this Stepper.
	if( aNextStep <= aCallerNextStep )
	  {
	    //std::cerr << aCurrentTime << " return" << std::endl;

	    return;
	  }

	//	std::cerr << aCurrentTime << " noreset" << std::endl;

	
	// If the next step of this will occur *after* the caller,
	// reschedule this Stepper, as well as shrinking the next step size.
	//    setStepInterval( aCallerCurrentTime + ( aCallerTimeScale * 0.5 ) 
	//		     - aCurrentTime );
      }
    else
      {
	// reset step interval to the default
	//	std::cerr << aCurrentTime << " reset" << std::endl;

	setNextStepInterval( 0.001 );
      }
      
    setStepInterval( aCallerCurrentTime - aCurrentTime );
    getModel()->reschedule( this );
  }


  ////////////////////////// AdaptiveDifferentialStepper

  void AdaptiveDifferentialStepper::makeSlots()
  {
    DEFINE_PROPERTYSLOT( "Tolerance", Real,
			 &AdaptiveDifferentialStepper::setTolerance,
			 &AdaptiveDifferentialStepper::getTolerance );

    DEFINE_PROPERTYSLOT( "AbsoluteToleranceFactor", Real,
			 &AdaptiveDifferentialStepper::setAbsoluteToleranceFactor,
			 &AdaptiveDifferentialStepper::getAbsoluteToleranceFactor );
 
    DEFINE_PROPERTYSLOT( "StateToleranceFactor", Real,
			 &AdaptiveDifferentialStepper::setStateToleranceFactor,
			 &AdaptiveDifferentialStepper::getStateToleranceFactor );

    DEFINE_PROPERTYSLOT( "DerivativeToleranceFactor", Real,
			 &AdaptiveDifferentialStepper::setDerivativeToleranceFactor,
			 &AdaptiveDifferentialStepper::getDerivativeToleranceFactor );

    DEFINE_PROPERTYSLOT( "MaxErrorRatio", Real,
			 NULLPTR,
			 &AdaptiveDifferentialStepper::getMaxErrorRatio );

    DEFINE_PROPERTYSLOT( "AbsoluteEpsilon", Real,
			 &AdaptiveDifferentialStepper::setAbsoluteEpsilon,
			 &AdaptiveDifferentialStepper::getAbsoluteEpsilon );

    DEFINE_PROPERTYSLOT( "RelativeEpsilon", Real,
			 &AdaptiveDifferentialStepper::setRelativeEpsilon,
			 &AdaptiveDifferentialStepper::getRelativeEpsilon );

    DEFINE_PROPERTYSLOT( "Order", Int, 
			 NULLPTR,
			 &AdaptiveDifferentialStepper::getOrder );
  }

  AdaptiveDifferentialStepper::AdaptiveDifferentialStepper()
    :
    theTolerance( 1.0e-6 ),
    theAbsoluteToleranceFactor( 1.0 ),
    theStateToleranceFactor( 1.0 ),
    theDerivativeToleranceFactor( 1.0 ),
    theAbsoluteEpsilon( 0.1 ),
    theRelativeEpsilon( 0.1 ),
    safety( 0.9 ),
    theMaxErrorRatio( 1.0 )
  {
    makeSlots();
  }

  void AdaptiveDifferentialStepper::initialize()
  {
    DifferentialStepper::initialize();
  }

  void AdaptiveDifferentialStepper::step()
  {
    // clear
    clear();

    setStepInterval( getNextStepInterval() );

    while( !calculate() )
      {
	const Real
	  anExpectedStepInterval( getStepInterval() 
				  * pow( getMaxErrorRatio(),
					 -1.0 / getOrder() ) 
				  * safety );

	  if ( anExpectedStepInterval > getMinInterval() )
	    {
	      // shrink it if the error exceeds 110%
	      setStepInterval( anExpectedStepInterval );
	      
	      //	setStepInterval( getStepInterval() * 0.5 );

	      //	std::cerr << "s " << getCurrentTime() 
	      //		  << ' ' << getStepInterval()
	      //		  << std::endl;
	    }
	  else
	    {
	      setStepInterval( getMinInterval() );

	      // this must return false,
	      // so theOriginalStepInterval does NOT LIMIT the error.
	      calculate();
	      setOriginalStepInterval( getMinInterval() );
	      break;
	    }
      }

    if ( getOriginalStepInterval() < getMinInterval() )
      {
	THROW_EXCEPTION( SimulationError,
			 "The error-limit step interval of Stepper [" + 
			 getID() + "] is too small." );
      }

    const Real anAdaptedStepInterval( getStepInterval() );
    const UnsignedInt aSize( getReadOnlyVariableOffset() );

    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	Real const aTolerance( ( fabs( aVariable->getValue() ) 
				 + theAbsoluteEpsilon ) * theRelativeEpsilon );
	Real const aVelocity( fabs( theVelocityBuffer[ c ] ) );

	if ( aTolerance < aVelocity * getStepInterval() )
	  {
	    setStepInterval( aTolerance / aVelocity );
	  }
      }

    if ( anAdaptedStepInterval > getStepInterval() )
      {
	reset();

	if ( !calculate() )
	  {
	    ; // this should not be called
	  }
      }

    const Real maxError( getMaxErrorRatio() );

    // grow it if error is 50% less than desired
    if ( maxError < 0.5 )
      {
	Real aNewStepInterval( getStepInterval() 
			       * pow(maxError , -1.0 / ( getOrder() + 1 ) )
			       * safety );

	//	Real aNewStepInterval( getStepInterval() * 2.0 );

	//	std::cerr << "g " << getCurrentTime() << ' ' 
	//		  << getStepInterval() << std::endl;

	setNextStepInterval( aNewStepInterval );
      }
    else 
      {
	setNextStepInterval( getStepInterval() );
      }
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

