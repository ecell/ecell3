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

#ifndef __ODE23_HPP
#define __ODE23_HPP


// #include <iostream>

#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ODE23Stepper, AdaptiveDifferentialStepper )
{

  class Interpolant
    :
  public libecs::Interpolant
  {
  public:

    Interpolant( ODE23StepperRef aStepper, VariablePtr const aVariablePtr )
      :
      libecs::Interpolant( aVariablePtr ),
      theStepper( aStepper ),
      theIndex( theStepper.getVariableIndex( aVariablePtr ) )
    {
      ; // do nothing
    }

    virtual const Real getDifference( RealParam aTime, RealParam anInterval )
    {
      const Real aTolerableStepInterval
	( theStepper.getTolerableStepInterval() );

      const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

      const Real theta( FMA( aTimeInterval, 2.0, - anInterval )
			/ aTolerableStepInterval );

      const Real k1( theStepper.getK1()[ theIndex ] );
      const Real k2( theStepper.getK2()[ theIndex ] );

      return ( FMA( theta, k2, k1 ) * anInterval );
    }

  protected:

    ODE23StepperRef theStepper;
    UnsignedInteger theIndex;
  };

public:

  LIBECS_DM_OBJECT( ODE23Stepper, Stepper )
    {
      INHERIT_PROPERTIES( AdaptiveDifferentialStepper );
    }


  ODE23Stepper( void );
  
  virtual ~ODE23Stepper( void );

  virtual void initialize();
  virtual bool calculate();

  virtual GET_METHOD( Integer, Order ) { return 2; }

  virtual InterpolantPtr createInterpolant( VariablePtr aVariable )
  {
    return new 
      ODE23Stepper::Interpolant( *this, aVariable );
  }

  RealVectorCref getK2() const
  {
    return theK2;
  }

  void interIntegrate2();

protected:

  //  RealVector theK1;
  RealVector theK2;
};

#endif /* __ODE23_HPP */
