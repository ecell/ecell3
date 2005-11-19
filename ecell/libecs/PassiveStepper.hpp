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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __PASSIVESTEPPER_HPP
#define __PASSIVESTEPPER_HPP

#include "libecs.hpp"

#include "Stepper.hpp"


namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */


  /**
     PassiveStepper steps only when triggered by incoming interruptions from
     other Steppers.

     Note that this Stepper DOES dispatch interruptions to other Steppers
     when it steps.

     The step interval of this Stepper is usually infinity -- which
     means that this doesn't step spontaneously.  However, when
     interrupted by another Stepper, the step interval will be
     set zero, and this Stepper will step immediately after the
     currently stepping Stepper, at the same time point.

  */

  LIBECS_DM_CLASS( PassiveStepper, Stepper )
  {

  public:

    LIBECS_DM_OBJECT( PassiveStepper, Stepper )
      {
	INHERIT_PROPERTIES( Stepper );
      }


    PassiveStepper();
    ~PassiveStepper() {}
    
    virtual void initialize();

    virtual void step()
    {
      fireProcesses();

      setStepInterval( INF );
    }

    virtual void interrupt( StepperPtr const aCaller )
    {
      setCurrentTime( aCaller->getCurrentTime() );
      setStepInterval( 0.0 );
    }

    virtual SET_METHOD( Real, StepInterval )
    {
      // skip range check
      loadStepInterval( value );
    }


  };


} // namespace libecs

#endif /* __PASSIVESTEPPER_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
