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

#include "Stepper.hpp"

#include "Model.hpp"


namespace libecs
{

  Scheduler::Scheduler()
    :
    theCurrentTime( 0.0 )
  {
    ; // do nothing
  }

  Scheduler::~Scheduler()
  {
    ; // do nothing
  }

 
  void Scheduler::registerStepper( StepperPtr aStepper )
  {
    // need check if this is a slave stepper

    Int anIndex( registerEvent( Event( aStepper->getCurrentTime(),
				       aStepper ) ) );

    aStepper->setSchedulerIndex( anIndex );
  }

  void Scheduler::reset()
  {
    //FIXME: slow! :  no theScheduleQueue.clear() ?
    while( ! theScheduleQueue.empty() )
      {
 	theScheduleQueue.pop();
      }

    theCurrentTime = 0.0;
  }


  void Scheduler::step()
  {
    EventCref aTopEvent( theScheduleQueue.top() );
    const StepperPtr aStepperPtr( aTopEvent.getStepper() );
    setCurrentTime( aTopEvent.getTime() );
 
    aStepperPtr->integrate();
    aStepperPtr->setCurrentTime( getCurrentTime() );
    aStepperPtr->step();
    aStepperPtr->dispatchInterruptions();
    aStepperPtr->log();

    const Real aStepInterval( aStepperPtr->getStepInterval() );
    const Real aScheduledTime( getCurrentTime() + aStepInterval );

    // schedule a new event
    theScheduleQueue.changeTopKey( Event( aScheduledTime, aStepperPtr ) );
   }


  void Scheduler::reschedule( StepperPtr const aStepperPtr )
  {
    const Real aScheduledTime( aStepperPtr->getCurrentTime() + 
			       aStepperPtr->getStepInterval() );

    DEBUG_EXCEPTION( aScheduledTime >= getCurrentTime(),
		     UnexpectedError,
		     "Attempt to go past." );

    theScheduleQueue.changeOneKey( aStepperPtr->getSchedulerIndex(),
				   Event( aScheduledTime, aStepperPtr ) );
  }


} // namespace libecs





/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
