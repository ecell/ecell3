//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

namespace libecs
{

LIBECS_DM_INIT_STATIC( DiscreteEventStepper, Stepper );

DiscreteEventStepper::DiscreteEventStepper()
    : theTolerance( 0.0 ),
      theLastEventID( -1 )
{
    ; // do nothing
}

GET_METHOD_DEF( String, LastProcess, DiscreteEventStepper )
{
    if( theLastEventID != -1 )
    {
        Process const* const aLastProcess(
                theScheduler.getEvent( theLastEventID ).getProcess() );
        
        return aLastProcess->getFullID().asString();
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
        THROW_EXCEPTION_INSIDE( InitializationFailed,
                                asString() + 
                                ": at least one Process "
                                "must be defined in this Stepper" );
    }

    const Real aCurrentTime( getCurrentTime() );


    // (1) (re)construct the scheduler's priority queue.

    // can this be done in registerProcess()?

    theScheduler.clear();
    for( ProcessVector::const_iterator i( theProcessVector.begin() );
             i != theProcessVector.end(); ++i )
    {            
        Process* const aProcessPtr( *i );
        
        // register a Process (as an event generator) to 
        // the priority queue.
        theScheduler.addEvent( ProcessEvent( aProcessPtr->getStepInterval()
                               + aCurrentTime, aProcessPtr ) );
    }

    // (2) (re)construct the event dependency array.

    theScheduler.updateEventDependency();


    // (3) Reschedule this Stepper.

    // Here all the Processes are updated, then set new
    // stepinterval.    This Stepper will be rescheduled
    // by the scheduler with the new stepinterval.
    // That means, this Stepper doesn't necessary step immediately
    // after initialize().
    ProcessEvent const& aTopEvent( theScheduler.getTopEvent() );
    theNextTime = aTopEvent.getTime();
}




void DiscreteEventStepper::step()
{
    theLastEventID = theScheduler.getTopID();

    theScheduler.step();

    // Set new StepInterval.
    theNextTime = theScheduler.getTopEvent().getTime();
}


void DiscreteEventStepper::interrupt( Time aTime )
{
    // update current time, because the procedure below
    // is effectively a stepping.
    setCurrentTime( aTime );

    // update step intervals of all the Processes.
    theScheduler.updateAllEvents( getCurrentTime() );

    setNextTime( theScheduler.getTopEvent().getTime() );
}

void DiscreteEventStepper::log()
{
    // call Logger::log() of Loggers that are attached to
    // the last fired Process and Variables in its VariableReferenceVector.

    const Real aCurrentTime( getCurrentTime() );

    ProcessEvent const& aLastEvent( theScheduler.getEvent( theLastEventID ) );
    Process const* aLastProcess( aLastEvent.getProcess() );

    FOR_ALL( LoggerBroker::LoggersPerFullID,
             aLastProcess->getLoggers() )
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
        ProcessEvent const& aDependentEvent( theScheduler.getEvent( *i ) );
        Process const* aDependentProcess( aDependentEvent.getProcess() );

        FOR_ALL( LoggerBroker::LoggersPerFullID,
                 aDependentProcess->getLoggers() )
        {
            (*i)->log( aCurrentTime );
        }
    }

    typedef Process::VariableReferenceVector VariableReferenceVector;
    VariableReferenceVector const& aVariableReferenceVector(
            aLastProcess->getVariableReferenceVector() );

    for( VariableReferenceVector::const_iterator
                 j( aVariableReferenceVector.begin() );
         j != aVariableReferenceVector.end(); ++j )
    {
        Variable const* aVariablePtr( (*j).getVariable() );

        FOR_ALL( LoggerBroker::LoggersPerFullID,
                 aVariablePtr->getLoggers() )
        {
            (*i)->log( aCurrentTime );
        }
    }
}

} // namespace libecs
