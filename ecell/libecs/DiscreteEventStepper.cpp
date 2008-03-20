//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
#include "LoggerManager.hpp"
#include "DiscreteEventStepper.hpp"


namespace libecs
{

LIBECS_DM_INIT_STATIC( DiscreteEventStepper, Stepper );

//////////////////// DiscreteEventStepper

DiscreteEventStepper::DiscreteEventStepper()
        :
        //    theTimeScale( 0.0 ),
        tolerance_( 0.0 ),
        lastEventID_( -1 )
{
    ; // do nothing
}

GET_METHOD_DEF( String, LastProcess, DiscreteEventStepper )
{
    if ( lastEventID_ != -1 )
    {
        return scheduler_.getEvent( lastEventID_ ).getProcess()->getFullID().asString();
    }
    else
    {
        return "";
    }
}

void DiscreteEventStepper::initialize()
{
    Stepper::initialize();

    if ( processes_.empty() )
    {
        THROW_EXCEPTION( InitializationFailed,
                         "no process is associated" );
    }

    const Real currentTime( getCurrentTime() );


    // (1) (re)construct the scheduler's priority queue.

    // can this be done in registerProcess()?

    // register a Process (as an event generator) to the priority queue.
    scheduler_.clear();
    for ( ProcessVector::const_iterator i( processes_.begin() );
            i != processes_.end(); ++i )
    {
        scheduler_.addEvent(
            ProcessEvent( (*i)->getStepInterval() + currentTime, *i ) );
    }

    // (2) (re)construct the event dependency array.
    scheduler_.updateEventDependency();


    // (3) Reschedule this Stepper.

    // Here all the Processes are updated, then set new
    // stepinterval.  This Stepper will be rescheduled
    // by the scheduler with the new stepinterval.
    // That means, this Stepper doesn't necessary step immediately
    // after initialize().
    setStepInterval( scheduler_.getTopEvent().getTime() - currentTime );
}

void DiscreteEventStepper::step()
{
    lastEventID_ = scheduler_.getTopID();

    scheduler_.step();

    // Set new StepInterval.
    setStepInterval( scheduler_.getTopEvent().getTime() - getCurrentTime() );
}


void DiscreteEventStepper::interrupt( TimeParam aTime )
{
    // update current time, because the procedure below
    // is effectively a stepping.
    setCurrentTime( aTime );

    // update step intervals of all the Processes.
    scheduler_.updateAllEvents( getCurrentTime() );

    setStepInterval( scheduler_.getTopEvent().getTime() - getCurrentTime() );
}

void DiscreteEventStepper::log()
{
    // call Logger::log() of Loggers that are attached to
    // the last fired Process and Variables in its VariableReferenceVector.
    const Process* lastProcess(
            scheduler_.getEvent( lastEventID_ ).getProcess() );
    loggerManager_->log( currentTime_, lastProcess );

    {
        const Process::VarRefVectorCRange& varRefs(
                lastProcess->getVariableReferences() );

        for ( Process::VarRefVector::const_iterator i( varRefs.begin() );
                i != varRefs.end(); ++i )
        {
            loggerManager_->log( currentTime_, (*i).getVariable() );
        }
    }

    // Log all relevant processes.
    //
    // this will become unnecessary, in future versions,
    // in which Loggers log only Variables.
    //
    typedef ProcessEventScheduler::EventIDVector EventIDVector;
    const EventIDVector& anEventIDVector(
        scheduler_.getDependencyVector( lastEventID_ ) );

    for ( EventIDVector::const_iterator i( anEventIDVector.begin() );
            i != anEventIDVector.end(); ++i )
    {
        const Process* dependentProcess( scheduler_.getEvent( *i ).getProcess() );
        loggerManager_->log( currentTime_, dependentProcess );
        const Process::VarRefVectorCRange& varRefs(
                dependentProcess->getVariableReferences() );

        for ( Process::VarRefVector::const_iterator i( varRefs.begin() );
                i != varRefs.end(); ++i )
        {
            loggerManager_->log( currentTime_, (*i).getVariable() );
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

