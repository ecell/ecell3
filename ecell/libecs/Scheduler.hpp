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

#ifndef __SCHEDULER_HPP
#define __SCHEDULER_HPP

#include "libecs.hpp"
#include "DynamicPriorityQueue.hpp"


namespace libecs
{

  /** @addtogroup model The Model.

      The model.

      @ingroup libecs
      @{ 
   */ 

  /** @file */

  class SchedulerEvent
  {


  public:

    SchedulerEvent( TimeCref aTime, StepperPtr aStepperPtr )
      :
      theTime( aTime ),
      theStepperPtr( aStepperPtr )
    {
      ; // do nothing
    }

    const Time getTime() const
    {
      return theTime;
    }

    const StepperPtr getStepper() const
    {
      return theStepperPtr;
    }

    // this method is basically used in initializing and rescheduling 
    // in the Scheduler to determine if
    // goUp()/goDown (position change) is needed 
    const bool operator< ( SchedulerEventCref rhs ) const
    {
      if( theTime > rhs.theTime )
	{
	  return false;
	}
      else if( theTime < rhs.theTime )
	{
	  return true;
	}
      else // if theTime == rhs.theTime,
	{  // then higher priority comes first 
	  return false;
	  /*
	  if( theStepperPtr->getPriority() > rhs.getStepper()->getPriority() )
	    {
	      return true;
	    }
	  else
	    {
	      return false;
	    }
	  */
	}
    }

    const bool operator!= ( SchedulerEventCref rhs ) const
    {
      if( theTime != rhs.theTime || theStepperPtr != rhs.theStepperPtr )
	{
	  return true;
	}
      else
	{
	  return false;
	}
      // theTime == rhs.theTime
      /*
      else if( theStepperPtr->getPriority() == 
	       rhs.theStepperPtr->getPriority() )
	{
	  return false;
	}
      else
	{
	  return true;
	}
      */
    }


    // dummy, because DynamicPriorityQueue requires this. better without.
    SchedulerEvent()
    {
      ; // do nothing
    }


  private:

    Time       theTime;
    StepperPtr theStepperPtr;

  };


  typedef DynamicPriorityQueue<SchedulerEvent> 
  SchedulerEventDynamicPriorityQueue;

  DECLARE_TYPE( SchedulerEventDynamicPriorityQueue, ScheduleQueue );


  /**
     Simulation scheduler.

     This class works as a event scheduler with a heap-tree based priority
     queue.

     theScheduleQueue is basically a priority queue used for
     scheduling, of which containee is synchronized with the
     StepperMap by resetScheduleQueue() method.

  */

  class Scheduler
  {

  public:

    typedef ScheduleQueue::index_type IndexType;

    Scheduler();
    ~Scheduler();

    /**
       Initialize the whole model.

       This method must be called before running the model, and when
       structure of the model is changed.
    */

    void initialize();

    /**
       Conduct a step of the simulation.

       This method picks a Stepper on the top of theScheduleQueue,
       calls sync(), step(), and push() of the Stepper, and
       reschedules it on the queue.

    */

    void step();

    SchedulerEventCref getNextEvent() const
    {
      return theScheduleQueue.top();
    }


    void reschedule( StepperPtr const aStepper );

    /**
       Returns the current time.

       The current time of this scheduler is scheduled time of the
       SchedulerEvent on the top of the ScheduleQueue.

       @return the current time of this scheduler.
    */

    const Time getCurrentTime() const
    {
      return theCurrentTime;
    }

    void setCurrentTime( TimeCref aTime )
    {
      theCurrentTime = aTime;
    }

    void registerStepper( StepperPtr aStepper );

  protected:

    /**
       This method clears the ScheduleQueue.
    */

    void reset();

    IndexType registerEvent( SchedulerEventCref anEvent )
    {
      theScheduleQueue.push( anEvent );
      return theScheduleQueue.size() - 1;
    }

  private:

    ScheduleQueue       theScheduleQueue;
    Time                theCurrentTime;

  };

  
  /*@}*/

} // namespace libecs




#endif /* __SCHEDULER_HPP */




/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

