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

  DECLARE_CLASS( FixedRungeKutta4Stepper );

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


  DECLARE_CLASS( Fehlberg21Stepper );

  class Fehlberg21Stepper
    :
    public AdaptiveDifferentialStepper
  {

  public:

    Fehlberg21Stepper();
    virtual ~Fehlberg21Stepper() {}

    static StepperPtr createInstance() { return new Fehlberg21Stepper; }

    virtual void initialize();
    virtual bool calculate();

    virtual const Int getOrder() const { return 2; }

    virtual StringLiteral getClassName() const { return "Fehlberg21Stepper"; }


  protected:
  };


  DECLARE_CLASS( Fehlberg23Stepper );

  class Fehlberg23Stepper
    : 
    public AdaptiveDifferentialStepper
  {

  public:

    Fehlberg23Stepper();
    virtual ~Fehlberg23Stepper() {}

    static StepperPtr createInstance() { return new Fehlberg23Stepper; }

    virtual void initialize();
    virtual bool calculate();

    virtual const Int getOrder() const { return 2; }

    virtual StringLiteral getClassName() const { return "Fehlberg23Stepper"; }

    //    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    //    {
    //      return new VariableProxy( *this, aVariable );
    //    }


  protected:

    //    RealVector theK1;
  };


  class CashKarp45Stepper
    : 
    public AdaptiveDifferentialStepper
  {

  public:

    CashKarp45Stepper();
    virtual ~CashKarp45Stepper() {}

    static StepperPtr createInstance() { return new CashKarp45Stepper; }

    virtual void initialize();
    virtual bool calculate();

    virtual const Int getOrder() const { return 4; }

    virtual StringLiteral getClassName() const { return "CashKarp45Stepper"; }


  protected:

    //    RealVector theK1;
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
    public AdaptiveDifferentialStepper
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
	theIndex( theStepper.getVariableIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getDifference( RealCref aTime, RealCref anInterval )
      {
	const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

	const Real theta1( aTimeInterval / theStepper.getStepInterval() );
	const Real theta2( ( aTimeInterval - anInterval )
			   / theStepper.getStepInterval() );

	const Real theta( theta1 + theta2 );

	const Real k1 = theStepper.getK1()[ theIndex ];
	const Real k2 = theStepper.getMidVelocityBuffer()[ theIndex ];
	const Real k3 = theStepper.getVelocityBuffer()[ theIndex ];

	const Real b = (-3) * k1 + 4 * k2 + (-1) * k3;
	const Real c = (+2) * k1 - 4 * k2 + (+2) * k3;

	return ( ( k1 + theta * ( b + c * theta ) - c * theta1 * theta2 )
		 * anInterval );
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
    virtual bool calculate();

    virtual const Int getOrder() const { return 5; }

    virtual StringLiteral getClassName() const
    { 
      return "DormandPrince547MStepper";
    }

    //    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    //    {
    //      return new VariableProxy( *this, aVariable );
    //    }

    RealVectorCref getMidVelocityBuffer() const
    {
      return theMidVelocityBuffer;
    }


  protected:

    //    RealVector theK1;
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
