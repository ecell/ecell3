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

#ifndef __STEPPEREVENT_HPP
#define __STEPPEREVENT_HPP

#include "libecs.hpp"
#include "Stepper.hpp"


namespace libecs
{

  /** @file */

  DECLARE_CLASS( StepperEvent );

  class StepperEvent
    :
    public EventBase
  {


  public:

    StepperEvent( TimeParam aTime, StepperPtr aStepperPtr )
      :
      EventBase( aTime ),
      theStepper( aStepperPtr )
    {
      ; // do nothing
    }


    void fire()
    {
      theStepper->integrate( getTime() );
      theStepper->step();
      theStepper->log();

      reschedule();
    }

    void update( TimeParam aTime )
    {
      theStepper->interrupt( aTime );

      reschedule();
    }

    void reschedule()
    {
      const Time aLocalTime( theStepper->getCurrentTime() );
      const Time aNewStepInterval( theStepper->getStepInterval() );
      setTime( aNewStepInterval + aLocalTime );
    }

    const bool isDependentOn( StepperEventCref anEvent ) const
    {
      return theStepper->isDependentOn( anEvent.getStepper() );
    }


    const StepperPtr getStepper() const
    {
      return theStepper;
    }


    // this method is basically used in initializing and rescheduling 
    // in the Scheduler to determine if
    // goUp()/goDown (position change) is needed 
    const bool operator< ( StepperEventCref rhs ) const
    {
      if( getTime() > rhs.getTime() )
	{
	  return false;
	}
      if( getTime() < rhs.getTime() )
	{
	  return true;
	}
      if( theStepper->getPriority() < rhs.getStepper()->getPriority() )
	{
	  return true;
	}
      return false;
    }

    const bool operator> ( StepperEventCref rhs ) const
    {
      if( getTime() < rhs.getTime() )
	{
	  return false;
	}
      if( getTime() > rhs.getTime() )
	{
	  return true;
	}
      if( theStepper->getPriority() > rhs.getStepper()->getPriority() )
	{
	  return true;
	}
      return false;
    }

    const bool operator<= ( StepperEventCref rhs ) const
    {
      return !( *this > rhs );
    }

    const bool operator>= ( StepperEventCref rhs ) const
    {
      return !( *this < rhs );
    }

    const bool operator!= ( StepperEventCref rhs ) const
    {
      if( getStepper() == rhs.getStepper() &&
	  getTime() == rhs.getTime() )
	{
	  return false;
	}
      else
	{
	  return true;
	}
    }


    // dummy, because DynamicPriorityQueue requires this. better without.
    StepperEvent()
    {
      ; // do nothing
    }


  private:

    StepperPtr theStepper;

  };



} // namespace libecs




#endif /* __STEPPEREVENT_HPP */




/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

