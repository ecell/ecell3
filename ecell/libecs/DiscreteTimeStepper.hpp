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

#ifndef __DISCRETETIMESTEPPER_HPP
#define __DISCRETETIMESTEPPER_HPP

#include "libecs.hpp"

#include "Stepper.hpp"



namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */



  /**
     DiscreteTimeStepper has a fixed step interval.
     
     This stepper ignores incoming interruptions, but dispatches 
     interruptions always when it steps.

     Process objects in this Stepper isn't allowed to use 
     Variable::addVelocity() method, but Variable::setValue() method only.

  */

  LIBECS_DM_CLASS( DiscreteTimeStepper, Stepper )
  {

  public:

    LIBECS_DM_OBJECT( DiscreteTimeStepper, Stepper )
      {
	INHERIT_PROPERTIES( Stepper );
      }


    DiscreteTimeStepper();
    virtual ~DiscreteTimeStepper() {}


    virtual void initialize();

    /**
       This method calls fire() method of all Processes.
    */

    virtual void step();

    /**
       Do nothing.   This Stepper ignores interruption.
    */

    virtual void interrupt( TimeParam )
    {
      ; // do nothing -- ignore interruption
    }


    /**
       TimeScale of this Stepper is always zero by default.

       This behavior may be changed in subclasses.
    */

    virtual GET_METHOD( Real, TimeScale )
    {
      return 0.0;
    }

  };



} // namespace libecs

#endif /* __DISCRETETIMESTEPPER_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
