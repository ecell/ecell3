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

#ifndef __SCHEDULER_HPP
#define __SCHEDULER_HPP

#include "DynamicPriorityQueue.hpp"

#include "Stepper.hpp"


namespace libecs
{

  /** @addtogroup model The Model.

      The model.

      @ingroup libecs
      @{ 
   */ 

  /** @file */


  class Event
  {


  public:

    Event( RealCref aTime, StepperPtr aStepperPtr )
      :
      theTime( aTime ),
      theStepperPtr( aStepperPtr )
    {
      ; // do nothing
    }

    const Real getTime() const
    {
      return theTime;
    }

    const StepperPtr getStepper() const
    {
      return theStepperPtr;
    }

    const bool operator< ( EventCref rhs ) const
    {
      return theTime < rhs.theTime;
    }

    const bool operator!= ( EventCref rhs ) const
    {
      return theTime != rhs.theTime;
    }


    // dummy, because DynamicPriorityQueue requires this. better without.
    Event()
    {
      ; // do nothing
    }


  private:

    Real       theTime;
    StepperPtr theStepperPtr;

  };


  typedef DynamicPriorityQueue<Event> EventDynamicPriorityQueue;
  DECLARE_TYPE( EventDynamicPriorityQueue, ScheduleQueue );


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


    void reschedule( StepperPtr const aStepper );

    /**
       Returns the current time.

       The current time of this scheduler is scheduled time of the
       Event on the top of the ScheduleQueue.

       @return the current time of this scheduler.
    */

    const Real getCurrentTime() const
    {
      return theCurrentTime;
    }

    void setCurrentTime( RealCref aTime )
    {
      theCurrentTime = aTime;
    }

    void registerStepper( StepperPtr aStepper );

  protected:

    /**
       This method clears the ScheduleQueue.
    */

    void reset();

    IndexType registerEvent( EventCref anEvent )
    {
      theScheduleQueue.push( anEvent );
      return theScheduleQueue.size();
    }

  private:

    ScheduleQueue       theScheduleQueue;
    Real                theCurrentTime;

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

