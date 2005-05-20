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

#ifndef __DISCRETEEVENTSTEPPER_HPP
#define __DISCRETEEVENTSTEPPER_HPP

#include "libecs.hpp"
#include "Stepper.hpp"
#include "DynamicPriorityQueue.hpp"
#include "DiscreteEventProcess.hpp"

namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */

  /**

  */

  LIBECS_DM_CLASS( DiscreteEventStepper, Stepper )
  {

  protected:

    DECLARE_CLASS( StepperEvent );
    DECLARE_TYPE( DynamicPriorityQueue<StepperEvent>, PriorityQueue );

    // A pair of (reaction index, time) for inclusion in the priority queue.
    class StepperEvent
    {
    public:

      StepperEvent()
      {
	; // do nothing
      }

      StepperEvent( RealParam aTime, DiscreteEventProcessPtr aProcess )
	:
	theTime( aTime ),
	theProcess( aProcess )
      {
	; // do nothing
      }

      const Real getTime() const
      {
	return theTime;
      }

      DiscreteEventProcessPtr const getProcess() const
      {
	return theProcess;
      }

      const bool operator< ( StepperEventCref rhs ) const
      {
	return theTime < rhs.theTime;
      }

      const bool operator!= ( StepperEventCref rhs ) const
      {
	return theTime != rhs.theTime || 
	  theProcess != rhs.theProcess;
      }

    private:

      Real       theTime;
      DiscreteEventProcessPtr theProcess;

    };

  public:

    LIBECS_DM_OBJECT_ABSTRACT( DiscreteEventStepper )
      {
	INHERIT_PROPERTIES( Stepper );

	PROPERTYSLOT_SET_GET( Real, Tolerance );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, TimeScale );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( String, LastProcessName );
      }

    DiscreteEventStepper();
    virtual ~DiscreteEventStepper() {}

    virtual void initialize();
    virtual void step();
    virtual void interrupt( StepperPtr const aCaller );
    virtual void log();


    SET_METHOD( Real, Tolerance )
      {
	theTolerance = value;
      }
    
    GET_METHOD( Real, Tolerance )
      {
	return theTolerance;
      }
    
    virtual GET_METHOD( Real, TimeScale )
      {
	return theTimeScale;
      }

    GET_METHOD( String, LastProcessName );

    DiscreteEventProcessPtr const getLastProcess() const
      {
	return theLastProcess;
      }

    DiscreteEventProcessVectorCref getDiscreteEventProcessVector() const
      {
	return theDiscreteEventProcessVector;
      }

  protected:

    DiscreteEventProcessVector theDiscreteEventProcessVector;
    PriorityQueue thePriorityQueue;

    Real            theTimeScale;
    Real            theTolerance;

    DiscreteEventProcessPtr    theLastProcess;

    std::vector<DiscreteEventProcessVector> theDependentProcessVector;

  };

} // namespace libecs

#endif /* __STEPPER_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
