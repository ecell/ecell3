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
#include "Logger.hpp"

#include "Stepper.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( Stepper, Stepper );

  ////////////////////////// Stepper

  Stepper::Stepper() 
    :
    theReadWriteVariableOffset( 0 ),
    theReadOnlyVariableOffset( 0 ),
    theModel( NULLPTR ),
    theSchedulerIndex( -1 ),
    thePriority( 0 ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theOriginalStepInterval( 0.001 ),
    //    theMinStepInterval( std::numeric_limits<Real>::min() * 1000 ),
    //    theMaxStepInterval( std::numeric_limits<Real>::max() * .1 )
    theMinStepInterval( 1e-50 ),
    theMaxStepInterval( 1e+50 )
  {
    gsl_rng_env_setup();

    theRng = gsl_rng_alloc( gsl_rng_default );
  }

  Stepper::~Stepper()
  {
    gsl_rng_free( theRng );
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

    updateLoggerVector();
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


  void Stepper::updateLoggerVector()
  {
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


    theLoggerVector.clear();

    // Scan all the relevant Entities, and find loggers
    for( EntityVectorConstIterator i( anEntityVector.begin() );
	 i != anEntityVector.end() ; ++i )
      {
	EntityPtr anEntityPtr( *i );

	LoggerVectorCref aLoggerVector( anEntityPtr->getLoggerVector() );

	if( ! aLoggerVector.empty() )
	  {
	    theLoggerVector.insert( theLoggerVector.end(),
				    aLoggerVector.begin(),
				    aLoggerVector.end() );
	  }
      }

    // optimization: sort by memory address.
    std::sort( theLoggerVector.begin(), theLoggerVector.end() );
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


  /*
  void Stepper::registerLoggedPropertySlot( PropertySlotPtr aPropertySlotPtr )
  {
    theLoggedPropertySlotVector.push_back( aPropertySlotPtr );
  }
  */

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
    const Real aCurrentTime( getCurrentTime() );
    FOR_ALL( LoggerVector, theLoggerVector )
      {
	(*i)->log( aCurrentTime );
      }
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
    const UnsignedInt aSize( theVariableVector.size() );
    for( UnsignedInt c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// save original value values
	theValueBuffer[ c ] = aVariable->getValue();

	aVariable->clearVelocity();
      }

  }

  void Stepper::process()
  {
    FOR_ALL( ProcessVector, theProcessVector )
      {
	(*i)->process();
      }
  }


  void Stepper::integrate( const Real aTime )
  {
    //
    // Variable::integrate()
    //
    FOR_ALL( VariableVector, theVariableVector )
      {
	(*i)->integrate( aTime );
      }

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
	aVariable->clearVelocity();
      }
  }


  void Stepper::dispatchInterruptions()
  {
    FOR_ALL( StepperVector, theDependentStepperVector )
      {
	(*i)->interrupt( this );
      }
  }

  void Stepper::interrupt( StepperPtr const )
  {
    ; // do nothing
  }


  SET_METHOD_DEF( String, RngSeed, Stepper )
  {
    UnsignedInt aSeed( 0 );
    
    if( value == "TIME" )
      {
	aSeed = static_cast<UnsignedInt>( time( NULLPTR ) );
      }
    else
      {
	aSeed = stringTo<UnsignedInt>( value );
      }

    gsl_rng_set( getRng(), aSeed );
  }

  GET_METHOD_DEF( String, RngType, Stepper )
  {
    return gsl_rng_name( getRng() );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

