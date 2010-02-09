//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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
// written by Kazunari Kaizu <kaizu@sfc.keio.ac.jp>,
// E-Cell Project.
//

#ifndef __BOOLEANSTEPPER_HPP
#define __BOOLEANSTEPPER_HPP

#include "libecs/libecs.hpp"
#include "libecs/Stepper.hpp"
#include "BooleanProcess.hpp"


USE_LIBECS;

/** @addtogroup stepper
 *@{
 */

/** @file */

/**
   BooleanStepper has a fixed step interval.
     
   This stepper ignores incoming interruptions, but dispatches 
   interruptions always when it steps.

   Process objects in this Stepper isn't allowed to use 
   Variable::addVelocity() method, but Variable::setValue() method only.
*/

DECLARE_CLASS( BooleanProcess );
DECLARE_VECTOR( BooleanProcessPtr, BooleanProcessVector );

LIBECS_DM_CLASS( BooleanStepper, Stepper )
{

 public:

  LIBECS_DM_OBJECT( BooleanStepper, Stepper )
  {
    INHERIT_PROPERTIES( Stepper );
  }

  BooleanStepper( void )
  {
    ; // do nothing
  }

  virtual ~BooleanStepper() {}

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
    ; // ignore interruption
  }

  /**
     TimeScale of this Stepper is always zero by default.
     
     This behavior may be changed in subclasses.
  */

  virtual GET_METHOD( Real, TimeScale )
  {
    return 0.0;
  }

 protected:

  BooleanProcessVector theBooleanProcessVector;

};

#endif /* __BOOLEANSTEPPER_HPP */



/*
  Do not modify
  $Author: moriyoshi $
  $Revision: 3386 $
  $Date: 2009-02-05 15:01:04 +0900 (木, 05  2月 2009) $
  $Locker$
*/
