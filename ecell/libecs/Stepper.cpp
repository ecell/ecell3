//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include <functional>
#include <algorithm>
#include <limits>

#include "Util.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "Process.hpp"
#include "Model.hpp"
#include "FullID.hpp"
#include "Logger.hpp"
#include "SystemStepper.hpp"

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
    //    theMinStepInterval( std::numeric_limits<Real>::min() ),
    //    theMaxStepInterval( std::numeric_limits<Real>::max() )
    theMinStepInterval( 0.0 ),
    theMaxStepInterval( std::numeric_limits<Real>::infinity() )
  {
    gsl_rng_env_setup();

    theRng = gsl_rng_alloc( gsl_rng_default );

    setRngSeed( "TIME" );
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
    // Update theVariableVector.  This also calls updateInterpolantVector.
    //
    updateVariableVector();


    //    clearEntityListChanged();
    //      }

    // size of the value buffer == the number of *all* variables.
    // (not just read or write variables)
    theValueBuffer.resize( theVariableVector.size() );

    updateLoggerVector();

    
    //  Don't call
    //    createInterpolants();
    //  here:  only DifferentialSteppers need this.
  }

 
  void Stepper::updateProcessVector()
  {
    // lighter implementation of this method is 
    // to merge this into registerProcess() and removeProcess() and
    // find a position to insert/remove each time.

    // sort by memory address. this is an optimization.
    std::sort( theProcessVector.begin(), theProcessVector.end() );

    // sort by Process priority, conserving the partial order in memory
    std::stable_sort( theProcessVector.begin(), theProcessVector.end(),
		      Process::PriorityCompare() );

    // partition by isContinuous().
    ProcessVectorConstIterator aDiscreteProcessIterator
      ( std::stable_partition( theProcessVector.begin(),
			       theProcessVector.end(),
			       std::mem_fun( &Process::isContinuous ) ) );

    theDiscreteProcessOffset = 
      aDiscreteProcessIterator - theProcessVector.begin();
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

    VariableVectorIterator aReadWriteVariableIterator = 
      theVariableVector.begin() + theReadWriteVariableOffset;
    VariableVectorIterator aReadOnlyVariableIterator = 
      theVariableVector.begin() + theReadOnlyVariableOffset;

    // For each part of the vector, sort by memory address. 
    // This is an optimization.
    std::sort( theVariableVector.begin(),  aReadWriteVariableIterator );
    std::sort( aReadWriteVariableIterator, aReadOnlyVariableIterator );
    std::sort( aReadOnlyVariableIterator,  theVariableVector.end() );
  }


  void Stepper::updateIntegratedVariableVector()
  {
    theIntegratedVariableVector.clear();

    // want copy_if()...
    for( VariableVectorConstIterator i( theVariableVector.begin() );
	 i != theVariableVector.end(); ++i )
      {
	VariablePtr aVariablePtr( *i );

	if( aVariablePtr->isIntegrationNeeded() )
	  {
	    theIntegratedVariableVector.push_back( aVariablePtr );
	  }
      }

    // optimization: sort by memory address.
    std::sort( theIntegratedVariableVector.begin(), 
	       theIntegratedVariableVector.end() );
  }


  void Stepper::createInterpolants()
  {
    // create Interpolants.
    for( VariableVector::size_type c( 0 );  
	 c != theReadOnlyVariableOffset; ++c )
      {
	VariablePtr aVariablePtr( theVariableVector[ c ] );
	aVariablePtr->registerInterpolant( createInterpolant( aVariablePtr ) );
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

  const bool Stepper::isDependentOn( const StepperCptr aStepper )
  {
    // Every Stepper depends on the SystemStepper.
    // FIXME: UGLY -- reimplement SystemStepper in another way
    if( typeid( *aStepper ) == typeid( SystemStepper ) )
      {
	return true;
      }

    VariableVectorCref aTargetVector( aStepper->getVariableVector() );
    
    VariableVectorConstIterator aReadWriteTargetVariableIterator
      ( aTargetVector.begin() + 
	aStepper->getReadWriteVariableOffset() );
    
    VariableVectorConstIterator aReadOnlyTargetVariableIterator
      ( aTargetVector.begin() +
	aStepper->getReadOnlyVariableOffset() );
    
    // For efficiency, binary_search should be done for supposedly longer
    // vector, and linear iteration for supposedly shorter vector.
    //
    
    // if one Variable in this::readlist appears in the target::write list
    for( VariableVector::size_type c( 0 ); 
	 c != theReadOnlyVariableOffset; ++c )
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
	    return true;
	  }
      }
    
    return false;
  }


  GET_METHOD_DEF( Polymorph, SystemList, Stepper )
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


  /*
  GET_METHOD_DEF( Polymorph, DependentStepperList, Stepper )
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
  */

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


  void Stepper::log()
  {
    // update loggers
    FOR_ALL( LoggerVector, theLoggerVector )
      {
	(*i)->log( theCurrentTime );
      }
  }

  GET_METHOD_DEF( Polymorph, WriteVariableList, Stepper )
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableVector.size() );

    for( VariableVector::size_type c( 0 ); 
	 c != theReadOnlyVariableOffset; ++c )
      {
	aVector.push_back( theVariableVector[c]->getFullID().getString() );
      }
    
    return aVector;
  }


  GET_METHOD_DEF( Polymorph, ReadVariableList, Stepper )
  {
    PolymorphVector aVector;
    aVector.reserve( theVariableVector.size() );
    
    for( VariableVector::size_type c( theReadWriteVariableOffset ); 
	 c != theVariableVector.size(); ++c )
      {
	aVector.push_back( theVariableVector[c]->getFullID().getString() );
      }
    
    return aVector;
  }
  
  GET_METHOD_DEF( Polymorph, ProcessList, Stepper )
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
  
  const VariableVector::size_type 
  Stepper::getVariableIndex( VariableCptr const aVariable )
  {
    VariableVectorConstIterator
      anIterator( std::find( theVariableVector.begin(), 
			     theVariableVector.end(), 
			     aVariable ) ); 

    DEBUG_EXCEPTION( anIterator != theVariableVector.end() , NotFound, 
 		     "Variable not found." );

    return anIterator - theVariableVector.begin();
  }


  void Stepper::clearVariables()
  {
    const VariableVector::size_type aSize( theVariableVector.size() );
    for( VariableVector::size_type c( 0 ); c < aSize; ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// save original value values
	theValueBuffer[ c ] = aVariable->getValue();
      }

  }

  void Stepper::initializeProcesses()
  {
    FOR_ALL( ProcessVector, theProcessVector )
      {
	(*i)->initialize();
      }
  }


  void Stepper::fireProcesses()
  {
    FOR_ALL( ProcessVector, theProcessVector )
      {
	(*i)->fire();
      }
  }


  void Stepper::integrate( RealParam aTime )
  {
    //
    // Variable::integrate()
    //
    //    FOR_ALL( VariableVector, theVariableVector )
    FOR_ALL( VariableVector, theIntegratedVariableVector )
      {
	(*i)->integrate( aTime );
      }

    // this must be after Variable::integrate()
    setCurrentTime( aTime );
  }


  void Stepper::reset()
  {
    // restore original values and clear velocity of all the *write* variables.
    for( VariableVector::size_type c( 0 ); 
	 c < getReadOnlyVariableOffset(); ++c )
      {
	VariablePtr const aVariable( theVariableVector[ c ] );

	// restore x (original value) and clear velocity
	aVariable->setValue( theValueBuffer[ c ] );
      }
  }


  void Stepper::interrupt( TimeParam aTime )
  {
    ; // do nothing
  }


  SET_METHOD_DEF( String, RngSeed, Stepper )
  {
    UnsignedInteger aSeed( 0 );
    
    if( value == "TIME" )
      {
	// Using just time() still gives the same seeds to Steppers
	// in multi-stepper model.  Stepper index is added to prevent this.
	aSeed = static_cast<UnsignedInteger>( time( NULLPTR ) 
					      + getSchedulerIndex() );
      }
    else if( value == "DEFAULT" )
      {
	aSeed = gsl_rng_default_seed;
      }
    else
      {
	aSeed = stringCast<UnsignedInteger>( value );
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

