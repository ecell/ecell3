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

#include <algorithm>

#include "FullID.hpp"
#include "Model.hpp"
#include "Logger.hpp"

#include "DiscreteEventStepper.hpp"


namespace libecs
{

  LIBECS_DM_INIT_STATIC( DiscreteEventStepper, Stepper );


  //////////////////// DiscreteEventStepper

  DiscreteEventStepper::DiscreteEventStepper()
    :
    theTimeScale( 0.0 ),
    theTolerance( 0.0 ),
    theLastProcess( NULLPTR )
  {
    ; // do nothing
  }


  GET_METHOD_DEF( String, LastProcessName, DiscreteEventStepper )
  {
    if( theLastProcess != NULLPTR )
      {
	return theLastProcess->getFullID().getString();
      }
    else
      {
	return "";
      }
  }

  void DiscreteEventStepper::initialize()
  {
    Stepper::initialize();

    // dynamic_cast each Process in theProcessVector of this Stepper
    // to DiscreteEventProcess, and store it in theDiscreteEventProcessVector.
    theDiscreteEventProcessVector.clear();
    try
      {
	std::transform( theProcessVector.begin(), theProcessVector.end(),
			std::back_inserter( theDiscreteEventProcessVector ),
			DynamicCaster<DiscreteEventProcessPtr,ProcessPtr>() );
      }
    catch( const libecs::TypeError& )
      {
	THROW_EXCEPTION( InitializationFailed,
			 getClassNameString() + 
			 ": Only DiscreteEventProcesses are allowed to exist "
			 "in this Stepper." );
      }

    if( theDiscreteEventProcessVector.empty() )
      {
	THROW_EXCEPTION( InitializationFailed,
			 getClassNameString() + 
			 ": at least one DiscreteEventProcess "
			 "must be defined in this Stepper." );
      }

    // (1) check Process dependency
    // (2) update step interval of each Process
    // (3) construct the priority queue (scheduler)
    thePriorityQueue.clear();
    const Real aCurrentTime( getCurrentTime() );
    for( DiscreteEventProcessVector::const_iterator 
	   i( theDiscreteEventProcessVector.begin() );
	 i != theDiscreteEventProcessVector.end(); ++i )
      {      
	DiscreteEventProcessPtr anDiscreteEventProcessPtr( *i );
	
	// check Process dependencies
	anDiscreteEventProcessPtr->clearDependentProcessVector();
	// here assume aCoefficient != 0
	for( DiscreteEventProcessVector::const_iterator 
	       j( theDiscreteEventProcessVector.begin() );
	     j != theDiscreteEventProcessVector.end(); ++j )
	  {
	    DiscreteEventProcessPtr const anDiscreteEventProcess2Ptr( *j );
	  
	    if( anDiscreteEventProcessPtr->
		checkProcessDependency( anDiscreteEventProcess2Ptr ) )
	      {
		anDiscreteEventProcessPtr->
		  addDependentProcess( anDiscreteEventProcess2Ptr );
	      }
	  }

	// warning: implementation dependent
	// here we assume size() is the index of the newly pushed element
	const int anIndex( thePriorityQueue.size() );

	anDiscreteEventProcessPtr->setIndex( anIndex );
	anDiscreteEventProcessPtr->updateStepInterval();
	thePriorityQueue.
	  push( StepperEvent( anDiscreteEventProcessPtr->getStepInterval()
			      + aCurrentTime,
			      anDiscreteEventProcessPtr ) );
      }

    // here all the DiscreteEventProcesses are updated, then set new
    // step interval and reschedule this stepper.
    // That means, this Stepper doesn't necessary steps immediately
    // after initialize().
    StepperEventCref aTopEvent( thePriorityQueue.top() );
    const Real aNewTime( aTopEvent.getTime() );

    setStepInterval( aNewTime - aCurrentTime );
    getModel()->reschedule( this );

  }
  

  void DiscreteEventStepper::step()
  {
    StepperEventCref anEvent( thePriorityQueue.top() );

    DiscreteEventProcessPtr const aMuProcess( anEvent.getProcess() );
    aMuProcess->fire();
    theLastProcess = aMuProcess;

    const Real aCurrentTime( getCurrentTime() );

    // Update relevant processes
    DiscreteEventProcessVectorCref 
      theDependentProcessVector( aMuProcess->getDependentProcessVector() );
    for ( DiscreteEventProcessVectorConstIterator 
	    i( theDependentProcessVector.begin() );
	  i != theDependentProcessVector.end(); ++i ) 
      {
	DiscreteEventProcessPtr const anAffectedProcess( *i );
	anAffectedProcess->updateStepInterval();
	const Real aStepInterval( anAffectedProcess->getStepInterval() );
	// aTime is time in the priority queue

	int anIndex( anAffectedProcess->getIndex() );
	thePriorityQueue.
	  changeOneKey( anIndex,
			StepperEvent( aStepInterval + aCurrentTime,
				      anAffectedProcess ) );
      }

    const StepperEvent aTopEvent( thePriorityQueue.top() );
    const Real aNextStepInterval( aTopEvent.getTime() - aCurrentTime );
    DiscreteEventProcessPtr const aNewTopProcess( aTopEvent.getProcess() );

    // Calculate new timescale.
    // To prevent 0.0 * INF -> NaN from happening, simply set zero 
    // if the tolerance is zero.  
    // ( DiscreteEventProcess::getTimeScale() can return INF. )
    theTimeScale = ( theTolerance == 0.0 ) ? 
      0.0 : theTolerance * aNewTopProcess->getTimeScale();

    setStepInterval( aNextStepInterval );
  }


  void DiscreteEventStepper::interrupt( StepperPtr const aCaller )
  {
    // update current time, because the procedure below
    // is effectively a stepping.
    const Real aNewCurrentTime( aCaller->getCurrentTime() );
    setCurrentTime( aNewCurrentTime );

    // update step intervals of all the DiscreteEventProcesses.
    for( DiscreteEventProcessVector::const_iterator 
	   i( theDiscreteEventProcessVector.begin() );
	 i != theDiscreteEventProcessVector.end(); ++i )
      {      
	DiscreteEventProcessPtr const anDiscreteEventProcessPtr( *i );
	
	anDiscreteEventProcessPtr->updateStepInterval();
	const Real 
	  aStepInterval( anDiscreteEventProcessPtr->getStepInterval() );

	thePriorityQueue.
	  changeOneKey( anDiscreteEventProcessPtr->getIndex(),
			StepperEvent( aStepInterval + aNewCurrentTime,
				      anDiscreteEventProcessPtr ) );
      }

    StepperEventCref aTopEvent( thePriorityQueue.top() );
    const Real aNewTime( aTopEvent.getTime() );

    // reschedule this Stepper
    loadStepInterval( aNewTime - getCurrentTime() );

    getModel()->reschedule( this );
  }


  void DiscreteEventStepper::log()
  {
    if( theLoggerVector.empty() )
      {
	return;
      }


    // call Logger::log() of Loggers that are attached to
    // theLastProcess and Variables in its VariableReferenceVector.

    const Real aCurrentTime( getCurrentTime() );

    LoggerVectorCref aProcessLoggerVector( theLastProcess->getLoggerVector() );

    FOR_ALL( LoggerVector, aProcessLoggerVector )
      {
	(*i)->log( aCurrentTime );
      }

    VariableReferenceVectorCref
      aVariableReferenceVector( theLastProcess->getVariableReferenceVector() );

    for( VariableReferenceVectorConstIterator 
	   anIterator( aVariableReferenceVector.begin() );
	 anIterator != aVariableReferenceVector.end(); ++anIterator )
      {
	const VariableCptr aVariablePtr( (*anIterator).getVariable() );
	LoggerVectorCref aLoggerVector( aVariablePtr->getLoggerVector() );

	FOR_ALL( LoggerVector, aLoggerVector )
	  {
	    (*i)->log( aCurrentTime );
	  }
      }

  }



} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

