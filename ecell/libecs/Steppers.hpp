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
    
    LIBECS_DM_OBJECT( Stepper, FixedEuler1Stepper );

    FixedEuler1Stepper();
    virtual ~FixedEuler1Stepper() {}

    virtual void step();


  };

  DECLARE_CLASS( FixedRungeKutta4Stepper );

  class FixedRungeKutta4Stepper
    : 
    public DifferentialStepper
  {

  public:

    LIBECS_DM_OBJECT( Stepper, FixedRungeKutta4Stepper );

    FixedRungeKutta4Stepper();
    virtual ~FixedRungeKutta4Stepper() {}

    virtual void step();

  };


  DECLARE_CLASS( Fehlberg23Stepper );

  class Fehlberg23Stepper
    : 
    public AdaptiveDifferentialStepper
  {

  public:

    LIBECS_DM_OBJECT( Stepper, Fehlberg23Stepper );

    Fehlberg23Stepper();
    virtual ~Fehlberg23Stepper() {}

    virtual void initialize();
    virtual bool calculate();

    virtual const Int getOrder() const { return 2; }

  protected:

    //    RealVector theK1;
  };


  class CashKarp45Stepper
    : 
    public AdaptiveDifferentialStepper
  {

  public:

    LIBECS_DM_OBJECT( Stepper, CashKarp45Stepper );


    CashKarp45Stepper();
    virtual ~CashKarp45Stepper() {}

    virtual void initialize();
    virtual bool calculate();

    virtual const Int getOrder() const { return 4; }

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

  public:

    LIBECS_DM_OBJECT( Stepper, DormandPrince547MStepper );


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

      /**
	 Quartic (4th Order) Hermite interpolation
      */

      const Real interpolate( RealCref anInterval )
      {
	const Real theta( anInterval / theStepper.getOriginalStepInterval() );

	const Real k1 = theStepper.getK1()[ theIndex ];
	const Real k2 = theStepper.getMidVelocityBuffer()[ theIndex ];
	const Real k3 = theStepper.getVelocityBuffer()[ theIndex ];
	const Real k4 = theStepper.getK7()[ theIndex ];

	return anInterval * ( k1 + theta
			      * ( (theta - 0.5) * 2.0 * (k3 - k1)
				  + (theta - 1.0) * 4.0 * (k1 - k2) 
				  + (theta - 0.5) * (theta - 1.0) 
				  * 2.0 * (k4 - k1 - 4*k3 + 4*k2) ) );
      }

      virtual const Real getDifference( RealCref aTime, RealCref anInterval )
      {
	const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

	const Real i1 = interpolate( aTimeInterval );
	const Real i2 = interpolate( aTimeInterval - anInterval );

	return ( i1 - i2 );
      }

    protected:

      DormandPrince547MStepperRef theStepper;
      UnsignedInt                 theIndex;
    };

  public:

    DormandPrince547MStepper();
    virtual ~DormandPrince547MStepper() {}

    virtual void initialize();
    virtual void step();
    virtual bool calculate();

    virtual void interrupt( StepperPtr const aCaller );

    virtual const Int getOrder() const { return 5; }

    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    {
      return new 
	DormandPrince547MStepper::VariableProxy( *this, aVariable );
    }

    RealVectorCref getMidVelocityBuffer() const
    {
      return theMidVelocityBuffer;
    }

    RealVectorCref getK7() const
    {
      return theK7;
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

    bool theInterrupted;
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
