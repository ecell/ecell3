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

#ifndef __STEPPERS_HPP
#define __STEPPERS_HPP

#include "Stepper.hpp"
#include "VariableProxy.hpp"

namespace libecs
{

  DECLARE_CLASS( FixedEuler1Stepper );

  class FixedEuler1Stepper
    :
    public DifferentialStepper
  {

  public:
    
    FixedEuler1Stepper();
    virtual ~FixedEuler1Stepper() {}

    virtual void step();


    static StepperPtr createInstance() { return new FixedEuler1Stepper; }

    virtual StringLiteral getClassName() const { return "FixedEuler1Stepper"; }

  };


  class FixedRungeKutta4Stepper
    : 
    public DifferentialStepper
  {

  public:

    FixedRungeKutta4Stepper();
    virtual ~FixedRungeKutta4Stepper() {}
    static StepperPtr createInstance() { return new FixedRungeKutta4Stepper; }

    virtual void step();


    virtual StringLiteral getClassName() const { return "FixedRungeKutta4Stepper"; }


  protected:
  };


  class Euler1Stepper
    :
    public DifferentialStepper
  {

  public:

    Euler1Stepper();
    virtual ~Euler1Stepper() {}

    static StepperPtr createInstance() { return new Euler1Stepper; }

    virtual void initialize();
    virtual void step();

    bool calculate();


    virtual StringLiteral getClassName() const { return "Euler1Stepper"; }


  protected:
  };


  class Midpoint2Stepper
    : 
    public DifferentialStepper
  {

  public:

    Midpoint2Stepper();
    virtual ~Midpoint2Stepper() {}

    static StepperPtr createInstance() { return new Midpoint2Stepper; }

    virtual void initialize();
    virtual void step();

    bool calculate();

    virtual StringLiteral getClassName() const { return "Midpoint2Stepper"; }


  protected:

    RealVector theK1;
  };


  class CashKarp4Stepper
    : 
    public DifferentialStepper
  {

  public:

    CashKarp4Stepper();
    virtual ~CashKarp4Stepper() {}

    static StepperPtr createInstance() { return new CashKarp4Stepper; }

    virtual void initialize();
    virtual void step();
 
    bool calculate();

    virtual StringLiteral getClassName() const { return "CashKarp4Stepper"; }


  protected:

    RealVector theK1;
    RealVector theK2;
    RealVector theK3;
    RealVector theK4;
    RealVector theK5;
    RealVector theK6;

    RealVector theErrorEstimate;

  };

  class DormandPrince547MStepper
    : 
    public DifferentialStepper
  {

  public:

    DormandPrince547MStepper();
    virtual ~DormandPrince547MStepper() {}

    static StepperPtr createInstance() { return new DormandPrince547MStepper; }

    virtual void initialize();
    virtual void step();
 
    bool calculate();

    virtual StringLiteral getClassName() const
    { 
      return "DormandPrince547MStepper";
    }


  protected:

    RealVector theK1;
    RealVector theK2;
    RealVector theK3;
    RealVector theK4;
    RealVector theK5;
    RealVector theK6;
    RealVector theK7;

    RealVector theErrorEstimate;

  };

} // namespace libecs



#endif /* __STEPPERS_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
