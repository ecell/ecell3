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

#ifndef __DIFFERENTIALSTEPPER_HPP
#define __DIFFERENTIALSTEPPER_HPP

#include "libecs.hpp"

#include "Stepper.hpp"



namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */

  /**
     DIFFERENTIAL EQUATION SOLVER


  */

  DECLARE_CLASS( DifferentialStepper );

  LIBECS_DM_CLASS( DifferentialStepper, Stepper )
  {

  public:

    LIBECS_DM_OBJECT_ABSTRACT( DifferentialStepper )
      {
	INHERIT_PROPERTIES( Stepper );

	// FIXME: load/save ??
	PROPERTYSLOT( Real, StepInterval,
		      &DifferentialStepper::initializeStepInterval,
		      &DifferentialStepper::getStepInterval );
	
	PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, NextStepInterval );
      }

    class VariableProxy
      :
      public libecs::VariableProxy
    {
    public:
      VariableProxy( DifferentialStepperRef aStepper, 
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
	// First order interpolation.  This should be overridden in
	// higher order DifferentialSteppers.
	return theStepper.getVelocityBuffer()[ theIndex ] * anInterval;
      }
      

    protected:

      DifferentialStepperRef theStepper;
      UnsignedInt            theIndex;

    };


  public:

    DifferentialStepper();
    virtual ~DifferentialStepper() {}

    /**
       Override setStepInterval() for theTolerantStepInterval.

    */

    virtual SET_METHOD( Real, StepInterval )
    {
      theTolerantStepInterval = value;

      Stepper::setStepInterval( value );
    }

    GET_METHOD( Real, TolerantStepInterval )
    {
      return theTolerantStepInterval;
    }

    SET_METHOD( Real, NextStepInterval )
    {
      theNextStepInterval = value;
    }

    GET_METHOD( Real, NextStepInterval )
    {
      return theNextStepInterval;
    }

    void initializeStepInterval( RealCref aStepInterval )
    {
      setStepInterval( aStepInterval );
      setNextStepInterval( aStepInterval );
    }

    virtual void initialize();

    virtual void reset();

    virtual void interrupt( StepperPtr const aCaller );


    RealVectorCref getVelocityBuffer() const
    {
      return theVelocityBuffer;
    }

    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    {
      return new DifferentialStepper::VariableProxy( *this, aVariable );
    }


  protected:

    const bool isExternalErrorTolerable() const;

  protected:

    RealVector theVelocityBuffer;

  private:

    Real theTolerantStepInterval;
    Real theNextStepInterval;

  };


  /**
     ADAPTIVE STEPSIZE DIFFERENTIAL EQUATION SOLVER


  */

  DECLARE_CLASS( AdaptiveDifferentialStepper );

  LIBECS_DM_CLASS( AdaptiveDifferentialStepper, DifferentialStepper )
  {

  public:

    LIBECS_DM_OBJECT_ABSTRACT( AdaptiveDifferentialStepper )
      {
	INHERIT_PROPERTIES( DifferentialStepper );

	PROPERTYSLOT_SET_GET( Real, Tolerance );
	PROPERTYSLOT_SET_GET( Real, AbsoluteToleranceFactor );
	PROPERTYSLOT_SET_GET( Real, StateToleranceFactor );
	PROPERTYSLOT_SET_GET( Real, DerivativeToleranceFactor );
	PROPERTYSLOT_SET_GET( Real, AbsoluteEpsilon );
	PROPERTYSLOT_SET_GET( Real, RelativeEpsilon );


	PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, MaxErrorRatio );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( Int,  Order );
      }


    class VariableProxy
      :
      public libecs::VariableProxy
    {
    public:

      VariableProxy( AdaptiveDifferentialStepperRef aStepper, 
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
	const Real anOriginalStepInterval
	  ( theStepper.getOriginalStepInterval() );

	const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

	const Real theta( FMA( aTimeInterval, 2.0, - anInterval )
			  / anOriginalStepInterval );

	const Real k1( theStepper.getK1()[ theIndex ] );
	const Real k2( theStepper.getVelocityBuffer()[ theIndex ] );

	return ( FMA( theta, ( k2 - k1 ), k1 ) * anInterval );
      }

    protected:

      AdaptiveDifferentialStepperRef theStepper;
      UnsignedInt                    theIndex;
    };

  public:

    AdaptiveDifferentialStepper();
    virtual ~AdaptiveDifferentialStepper() {}

    /**
       Adaptive stepsize control.

       These methods are for handling the standerd error control objects.
    */

    SET_METHOD( Real, Tolerance )
    {
      theTolerance = value;
    }

    GET_METHOD( Real, Tolerance )
    {
      return theTolerance;
    }

    SET_METHOD( Real, AbsoluteToleranceFactor )
    {
      theAbsoluteToleranceFactor = value;
    }

    GET_METHOD( Real, AbsoluteToleranceFactor )
    {
      return theAbsoluteToleranceFactor;
    }

    SET_METHOD( Real, StateToleranceFactor )
    {
      theStateToleranceFactor = value;
    }

    GET_METHOD( Real, StateToleranceFactor )
    {
      return theStateToleranceFactor;
    }


    SET_METHOD( Real, DerivativeToleranceFactor )
    {
      theDerivativeToleranceFactor = value;
    }

    GET_METHOD( Real, DerivativeToleranceFactor )
    {
      return theDerivativeToleranceFactor;
    }

    SET_METHOD( Real, MaxErrorRatio )
    {
      theMaxErrorRatio = value;
    }

    GET_METHOD( Real, MaxErrorRatio )
    {
      return theMaxErrorRatio;
    }

    /**
       check difference in one step
    */

    SET_METHOD( Real, AbsoluteEpsilon )
    {
      theAbsoluteEpsilon = value;
    }

    GET_METHOD( Real, AbsoluteEpsilon )
    {
      return theAbsoluteEpsilon;
    }

    SET_METHOD( Real, RelativeEpsilon )
    {
      theRelativeEpsilon = value;
    }

    GET_METHOD( Real, RelativeEpsilon )
    {
      return theRelativeEpsilon;
    }


    virtual GET_METHOD( Int, Order )
    { 
      return 1; 
    }

    virtual void initialize();
    virtual void step();
    virtual bool calculate() = 0;


    RealVectorCref getK1() const
    {
      return theK1;
    }

    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    {
      return new AdaptiveDifferentialStepper::VariableProxy( *this, aVariable );
    }

  protected:

    Real safety;
    RealVector theK1;

  private:

    Real theTolerance;
    Real theAbsoluteToleranceFactor;
    Real theStateToleranceFactor;
    Real theDerivativeToleranceFactor;

    Real theAbsoluteEpsilon;
    Real theRelativeEpsilon;

    Real theMaxErrorRatio;
  };


} // namespace libecs

#endif /* __DIFFERENTIALSTEPPER_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
