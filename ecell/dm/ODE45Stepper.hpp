//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __DORMANDPRINCE547M_HPP
#define __DORMANDPRINCE547M_HPP


// #include <iostream>

#include "libecs/Interpolant.hpp"
#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ODE45Stepper, AdaptiveDifferentialStepper )
{

  class Interpolant
    :
    public libecs::Interpolant
  {
  public:

    Interpolant( ODE45StepperRef aStepper, 
		   VariablePtr const aVariablePtr )
      :
      libecs::Interpolant( aVariablePtr ),
      theStepper( aStepper ),
      theIndex( theStepper.getVariableIndex( aVariablePtr ) )
    {
      ; // do nothing
    }

    /**
       Quartic (4th Order) Hermite interpolation
    */

    inline static const Real interpolate( const Real k1,
					  const Real k3__k1,
					  const Real k1__k2_2,
					  const Real k1__4k2_4k3__k4,
					  const Real anInterval,
					  const Real aStepIntervalInv )
    {
      const Real theta( anInterval * aStepIntervalInv );

      const Real theta_0_5( theta - 0.5 );

      return anInterval * ( k1 + ( theta + theta ) * 
			    ( theta_0_5 * k3__k1
			      + ( theta - 1.0 ) * 
			      ( k1__k2_2 - theta_0_5 * k1__4k2_4k3__k4 ) ) );
    }
   
    virtual const Real getDifference( RealParam aTime, RealParam anInterval )
    {
      if ( !theStepper.theStateFlag )
      	{
      	  return 0.0;
      	}

      const Real k1( theStepper.getK1()[ theIndex ] );
      const Real k2( theStepper.getMidVelocityBuffer()[ theIndex ] );
      const Real k3( theStepper.getVelocityBuffer()[ theIndex ] );
      const Real k4( theStepper.getK7()[ theIndex ] );

      const Real 
	aStepIntervalInv( 1.0 / theStepper.getTolerableStepInterval() );

      const Real k1__4k2_4k3__k4( ( k1 + ( k3 - k2 ) * 4.0 - k4 ) );
      const Real k1__k2_2( ( k1 - k2 ) * 2.0 );
      const Real k3__k1( k3 - k1 );

      const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

      const Real i1( interpolate( k1, k3__k1, k1__k2_2, k1__4k2_4k3__k4,
				  aTimeInterval, aStepIntervalInv ) );
      const Real i2( interpolate( k1, k3__k1, k1__k2_2, k1__4k2_4k3__k4,
				  ( aTimeInterval - anInterval ), 
				  aStepIntervalInv ) );

      return ( i1 - i2 );
    }

    /**
    const Real interpolate( const Real anInterval )
    {
      const Real theta( anInterval / theStepper.getTolerableStepInterval() );
      
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
   
    virtual const Real getDifference( const Real aTime, const Real anInterval )
    {
      const Real aTimeInterval( aTime - theStepper.getCurrentTime() );
      
      const Real i1 = interpolate( aTimeInterval );
      const Real i2 = interpolate( aTimeInterval - anInterval );
       
      return ( i1 - i2 );
    }
    */

  protected:

    ODE45StepperRef theStepper;
    UnsignedInteger theIndex;
  };

public:

  LIBECS_DM_OBJECT( ODE45Stepper, Stepper )
    {
      INHERIT_PROPERTIES( AdaptiveDifferentialStepper );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Stiffness );
    }

  ODE45Stepper();
  virtual ~ODE45Stepper();

  virtual void initialize();
  virtual void step();
  virtual bool calculate();

  virtual void interrupt( StepperPtr const aCaller );

  virtual GET_METHOD( Integer, Order ) { return 5; }

  virtual InterpolantPtr createInterpolant( VariablePtr aVariable )
  {
    return new 
      ODE45Stepper::Interpolant( *this, aVariable );
  }

  RealVectorCref getMidVelocityBuffer() const
  {
    return theMidVelocityBuffer;
  }

  RealVectorCref getK7() const
  {
    return theK7;
  }

  GET_METHOD( Real, Stiffness )
  {
    return theStiffness;
  }

  SET_METHOD( Real, Stiffness )
  {
    theStiffness = value;
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

  Real       theStiffness;

};

#endif /* __DORMANDPRINCE547M_HPP */
