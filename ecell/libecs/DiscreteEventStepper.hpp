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

#ifndef __DISCRETEEVENTSTEPPER_HPP
#define __DISCRETEEVENTSTEPPER_HPP

#include "libecs.hpp"
#include "Stepper.hpp"
#include "Process.hpp"

#include "EventScheduler.hpp"
#include "ProcessEvent.hpp"


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

    typedef EventScheduler<ProcessEvent> ProcessEventScheduler;
    typedef ProcessEventScheduler::EventIndex EventIndex;



  public:

    LIBECS_DM_OBJECT( DiscreteEventStepper, Stepper )
      {
	INHERIT_PROPERTIES( Stepper );

	PROPERTYSLOT_SET_GET( Real, Tolerance );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, TimeScale );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( String, LastProcess );
      }

    DiscreteEventStepper();
    virtual ~DiscreteEventStepper() {}

    virtual void initialize();
    virtual void step();
    virtual void interrupt( TimeParam aTime );
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
	//	return theTimeScale;  temporarily disabled
	return 0.0;
      }

    GET_METHOD( String, LastProcess );

    ProcessVectorCref getProcessVector() const
      {
	return theProcessVector;
      }

  protected:

    ProcessEventScheduler  theScheduler;

    // temporarily disabled
    //    Real            theTimeScale;
    Real            theTolerance;

    EventIndex      theLastEventIndex;



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
