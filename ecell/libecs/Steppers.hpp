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

    virtual StringLiteral getClassName() const
    {
      return "FixedEuler1Stepper";
    }

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


    virtual StringLiteral getClassName() const
    {
      return "FixedRungeKutta4Stepper";
    }


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


  DECLARE_CLASS( Midpoint2Stepper );

  class Midpoint2Stepper
    : 
    public DifferentialStepper
  {

  public:

    class VariableProxy
      :
      public libecs::VariableProxy
    {
    public:

      VariableProxy( Midpoint2StepperRef aStepper, 
		    VariablePtr const aVariablePtr )
	:
	libecs::VariableProxy( aVariablePtr ),
	theStepper( aStepper ),
	theIndex( theStepper.getVariableProxyIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getVelocity( RealCref aTime )
      {
	const Real theta( ( aTime - theStepper.getCurrentTime() )
			  / theStepper.getStepInterval() );

	const Real k1 = theStepper.getK1()[ theIndex ];
	const Real k2 = theStepper.getVelocityBuffer()[ theIndex ];

	return ( k1 + ( k2 - k1 ) * theta )
	  * ( aTime - theStepper.getCurrentTime() );
      }


    protected:

      Midpoint2StepperRef theStepper;
      UnsignedInt         theIndex;
    };


  public:

    Midpoint2Stepper();
    virtual ~Midpoint2Stepper() {}

    static StepperPtr createInstance() { return new Midpoint2Stepper; }

    virtual void initialize();
    virtual void step();

    bool calculate();

    virtual StringLiteral getClassName() const { return "Midpoint2Stepper"; }

    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    {
      return new VariableProxy( *this, aVariable );
    }

    RealVectorCref getK1() const
    {
      return theK1;
    }


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


  DECLARE_CLASS( DormandPrince547MStepper );

  class DormandPrince547MStepper
    : 
    public DifferentialStepper
  {

    class VariableProxy
      :
      public libecs::VariableProxy
    {
    public:

      VariableProxy( DormandPrince547MStepperRef aStepper, 
		    VariablePtr const aVariablePtr )
	:
	libecs::VariableProxy( aVariablePtr ),
	theStepper( aStepper ),
	theIndex( theStepper.getVariableProxyIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getVelocity( RealCref aTime )
      {
	const Real theta( ( aTime - theStepper.getCurrentTime() )
			  / theStepper.getStepInterval() );

	const Real k1 = theStepper.getK1()[ theIndex ];
	const Real k2 = theStepper.getVelocityBuffer()[ theIndex ];
	const Real k3 = theStepper.getMidVelocityBuffer()[ theIndex ];

	return ( k1 + ( ( -3*k1 - k2 + 8*k3 ) 
			+ ( 2*k1 + 2*k2 - 8*k3 ) * theta ) * theta )
	  * ( aTime - theStepper.getCurrentTime() );
      }


    protected:

      DormandPrince547MStepperRef theStepper;
      UnsignedInt                 theIndex;
    };

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

    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    {
      return new VariableProxy( *this, aVariable );
    }

    RealVectorCref getK1() const
    {
      return theK1;
    }

    RealVectorCref getMidVelocityBuffer() const
    {
      return theMidVelocityBuffer;
    }


  protected:

    RealVector theK1;
    RealVector theK2;
    RealVector theK3;
    RealVector theK4;
    RealVector theK5;
    RealVector theK6;
    RealVector theK7;

    RealVector theMidVelocityBuffer;
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
