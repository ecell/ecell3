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

#ifndef __ODE45_HPP
#define __ODE45_HPP

#include "libecs/Interpolant.hpp"
#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ODE45Stepper, AdaptiveDifferentialStepper )
{

public:

  LIBECS_DM_OBJECT( ODE45Stepper, Stepper )
    {
      INHERIT_PROPERTIES( AdaptiveDifferentialStepper );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, SpectralRadius );
    }

  ODE45Stepper();
  virtual ~ODE45Stepper();

  virtual void initialize();
  virtual void step();
  virtual bool calculate();

  virtual void interrupt( TimeParam aTime );

  virtual GET_METHOD( Integer, Order ) { return 4; }
  virtual GET_METHOD( Integer, Stage ) { return 5; }

  GET_METHOD( Real, SpectralRadius )
  {
    return theSpectralRadius;
  }

  SET_METHOD( Real, SpectralRadius )
  {
    theSpectralRadius = value;
  }

protected:

  bool isInterrupted;
  Real theSpectralRadius;

  RealMatrix theRungeKuttaBuffer;

  Integer count;

};

#endif /* __ODE45_HPP */
