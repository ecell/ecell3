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
    Int anIndex( registerEvent( Event( aStepper->getCurrentTime(),
				       aStepper ) ) );

    aStepper->setSchedulerIndex( anIndex );
  }

  void Scheduler::reset()
  {
    theScheduleQueue.clear();
    theCurrentTime = 0.0;
  }


  void Scheduler::step()
  {
    EventCref aTopEvent( theScheduleQueue.top() );
    const Time aCurrentTime( aTopEvent.getTime() );
    const StepperPtr aStepperPtr( aTopEvent.getStepper() );

    setCurrentTime( aCurrentTime );
 
    aStepperPtr->integrate( aCurrentTime );
    aStepperPtr->step();
    aStepperPtr->log();

    aStepperPtr->dispatchInterruptions();

    // Use higher precision for this procedure:
    const Time aStepInterval( aStepperPtr->getStepInterval() );
    const Time aScheduledTime( aCurrentTime + aStepInterval );

    // If the stepinterval is too small to proceed time,
    // throw an exception.   
    // Obviously time needs more precision. Possibly MP or 128-bit float.

    /*
    if( aCurrentTime == aScheduledTime && aStepInterval > 0.0 )
      {
	THROW_EXCEPTION( SimulationError, 
			 "Too small step interval given by Stepper [" +
			 aStepperPtr->getID() + "]." );
      }
    */

    // schedule a new event
    theScheduleQueue.changeTopKey( Event( aScheduledTime, aStepperPtr ) );
   }


  void Scheduler::reschedule( StepperPtr const aStepperPtr )
  {
    // Use higher precision for this addition.
    const Time 
      aScheduledTime( static_cast<Time>( aStepperPtr->getCurrentTime() ) + 
		      static_cast<Time>( aStepperPtr->getStepInterval() ) );

    //    DEBUG_EXCEPTION( aScheduledTime >= getCurrentTime(),
    //		     SimulationError,
    //		     "Attempt to go past." );

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
