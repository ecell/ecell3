//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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

#ifndef __FIXEDRUNGEKUTTA4_HPP
#define __FIXEDRUNGEKUTTA4_HPP


// #include <iostream>

#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

// DECLARE_VECTOR( Real, RealVector );

// DECLARE_CLASS( FixedRungeKutta4Stepper );

class FixedRungeKutta4Stepper 
  : 
  public DifferentialStepper
{

  LIBECS_DM_OBJECT( Stepper, FixedRungeKutta4Stepper );

public:

  FixedRungeKutta4Stepper( void );
  
  virtual ~FixedRungeKutta4Stepper( void );

  virtual void step();

protected:

};

#endif /* __FIXEDRUNGEKUTTA4_HPP */
