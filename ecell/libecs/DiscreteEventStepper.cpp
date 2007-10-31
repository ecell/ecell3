//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <algorithm>

#include "FullID.hpp"
#include "Model.hpp"
#include "Logger.hpp"

#include "DiscreteEventStepper.hpp"

#include "Debug.hpp"

namespace libecs
{

  LIBECS_DM_INIT_STATIC( DiscreteEventStepper, Stepper );

  //////////////////// DiscreteEventStepper

  DiscreteEventStepper::DiscreteEventStepper()
    :
    //    theTimeScale( 0.0 ),
    theTolerance( 0.0 ),
    theLastEventID( -1 )
  {
    ; // do nothing
  }

  GET_METHOD_DEF( String, LastProcess, DiscreteEventStepper )
  {
    if( theLastEventID != -1 )
      {
	const ProcessCptr aLastProcess(
	    theScheduler.getEvent( theLastEventID ).getProcess() );
	
	return aLastProcess->getFullID().getString();
      }
    else
      {
	return "";
      }
  }

  void DiscreteEventStepper::initialize()
  {
    Stepper::initialize();

    if ( theProcessVector.empty() )
      {
	THROW_EXCEPTION( InitializationFailed,
			 getClassNameString() + 
			 ": at least one Process "
			 "must be defined in this Stepper." );
      }

    // Is this right?
    const Real aCurrentTime( getCurrentTime() );


    // (1) (re)construct the scheduler's priority queue.

    // can this be done in registerProcess()?

    theScheduler.clear();
    for( ProcessVectorConstIterator i( theProcessVector.begin() );
	 i != theProcessVector.end(); ++i )
      {      
	ProcessPtr aProcessPtr( *i );
	
	// register a Process (as an event generator) to 
	// the priority queue.
	theScheduler.addEvent( ProcessEvent( aProcessPtr->getStepInterval()
					     + aCurrentTime,
					     aProcessPtr ) );
      }

    // (2) (re)construct the event dependency array.

    theScheduler.updateEventDependency();


    // (3) Reschedule this Stepper.

    // Here all the Processes are updated, then set new
    // stepinterval.  This Stepper will be rescheduled
    // by the scheduler with the new stepinterval.
    // That means, this Stepper doesn't necessary step immediately
    // after initialize().
    ProcessEventCref aTopEvent( theScheduler.getTopEvent() );
    const Real aNewTime( aTopEvent.getTime() );

    loadStepInterval( aNewTime - aCurrentTime );
  }




  void DiscreteEventStepper::step()
  {

    theLastEventID = theScheduler.getTopID();

    // assert( getCurrentTime() == theScheduler.getTopEvent().getTime() )


    theScheduler.step();


    // Set new StepInterval.
    ProcessEventCref aNewTopEvent( theScheduler.getTopEvent() );
    const Real aNewStepInterval( aNewTopEvent.getTime() - getCurrentTime() );

    // ProcessPtr const aNewTopProcess( aTopEvent.getProcess() );
    // Calculate new timescale.
    // To prevent 0.0 * INF -> NaN from happening, simply set zero 
    // if the tolerance is zero.  
    // ( DiscreteEventProcess::getTimeScale() can return INF. )

    // temporarily disabled
    //    theTimeScale = ( theTolerance == 0.0 ) ? 
    //      0.0 : theTolerance * aNewTopProcess->getTimeScale();

    // FIXME: should be setStepInterval()
    loadStepInterval( aNewStepInterval );
  }


  void DiscreteEventStepper::interrupt( TimeParam aTime )
  {
    // update current time, because the procedure below
    // is effectively a stepping.
    setCurrentTime( aTime );

    // update step intervals of all the Processes.
    //    for( ProcessVector::const_iterator i( theProcessVector.begin() );
    //	 i != theProcessVector.end(); ++i )
    theScheduler.updateAllEvents( getCurrentTime() );

    ProcessEventCref aTopEvent( theScheduler.getTopEvent() );
    const Real aNewTime( aTopEvent.getTime() );

    loadStepInterval( aNewTime - getCurrentTime() );
  }

  void DiscreteEventStepper::log()
  {
    if( theLoggerVector.empty() )
      {
	return;
      }

    // call Logger::log() of Loggers that are attached to
    // the last fired Process and Variables in its VariableReferenceVector.

    const Real aCurrentTime( getCurrentTime() );

    ProcessEventCref 
      aLastEvent( theScheduler.getEvent( theLastEventID ) );
    const ProcessCptr aLastProcess( aLastEvent.getProcess() );

    LoggerVectorCref aProcessLoggerVector( aLastProcess->getLoggerVector() );

    FOR_ALL( LoggerVector, aProcessLoggerVector )
      {
	(*i)->log( aCurrentTime );
      }

    // Log all relevant processes.
    //
    // this will become unnecessary, in future versions,
    // in which Loggers log only Variables.
    //
    typedef ProcessEventScheduler::EventIDVector EventIDVector;
    const EventIDVector& anEventIDVector(
	theScheduler.getDependencyVector( theLastEventID ) );

    for ( EventIDVector::const_iterator i( anEventIDVector.begin() );
	  i != anEventIDVector.end(); ++i ) 
      {
	ProcessEventCref 
	  aDependentEvent( theScheduler.getEvent( *i ) );
	ProcessPtr const aDependentProcess( aDependentEvent.getProcess() );


	LoggerVectorCref aDependentProcessLoggerVector( aDependentProcess->
							getLoggerVector() );
	for ( LoggerVectorConstIterator 
		j( aDependentProcessLoggerVector.begin() ); 
	      j != aDependentProcessLoggerVector.end(); ++j )
	  {
	    const LoggerPtr aLogger( *j );
	    aLogger->log( aCurrentTime );
	  }
      }


    VariableReferenceVectorCref
      aVariableReferenceVector( aLastProcess->getVariableReferenceVector() );

    for( VariableReferenceVectorConstIterator 
	   j( aVariableReferenceVector.begin() );
	 j != aVariableReferenceVector.end(); ++j )
      {
	const VariableCptr aVariablePtr( (*j).getVariable() );
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

