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
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __DIFFERENTIALSTEPPER_HPP
#define __DIFFERENTIALSTEPPER_HPP

#include "libecs.hpp"
#include "Stepper.hpp"

#include "boost/multi_array.hpp"

namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */

  /**
     DIFFERENTIAL EQUATION SOLVER


  */

  //  DECLARE_VECTOR( RealVector, RealMatrix );
  
  typedef boost::multi_array<Real, 2> RealMatrix;
  typedef boost::const_multi_array_ref<Real, 2> RealMatrixCref;


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
	PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real,  TolerableStepInterval );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Stage );
      }

    class Interpolant
      :
      public libecs::Interpolant
    {

    public:

      Interpolant( DifferentialStepperRef aStepper, 
		     VariablePtr const aVariablePtr )
	:
	libecs::Interpolant( aVariablePtr ),
	theStepper( aStepper ),
	theIndex( theStepper.getVariableIndex( aVariablePtr ) )
      {
	; // do nothing
      }
      

      /*
	The getDifference() below is an optimized version of
        the original implementation based on the following two functions.
	(2004/10/19)

      const Real interpolate( const RealMatrixCref aTaylorSeries,
			      const Real anInterval,
			      const Real aStepInterval )
      {
	const Real theta( anInterval / aStepInterval );

	Real aDifference( 0.0 );
	Real aFactorialInv( 1.0 );

	for ( RealMatrix::size_type s( 0 ); s < aTaylorSeries.size(); ++s )
	  {
	    //	    aFactorialInv /= s + 1;
	    aDifference += aTaylorSeries[ s ][ theIndex ] * aFactorialInv;
	    aFactorialInv *= theta;
	  }

	return aDifference * anInterval;
      }

      virtual const Real getDifference( RealParam aTime, 
					RealParam anInterval )
      {

	if ( !theStepper.theStateFlag )
	  {
	    return 0.0;
	  }

	const RealMatrixCref aTaylorSeries( theStepper.getTaylorSeries() );
	const Real aTimeInterval( aTime - theStepper.getCurrentTime() );
	const Real aStepInterval( theStepper.getTolerableStepInterval() );

	const Real i1( interpolate( aTaylorSeries, 
	                            aTimeInterval, 
                                    aStepInterval ) );
	const Real i2( interpolate( aTaylorSeries, 
                                    aTimeInterval - anInterval, 
                                    aStepInterval ) );
	return ( i1 - i2 );

	}
      */

      virtual const Real getDifference( RealParam aTime, 
					RealParam anInterval ) const
      {
        Real aDifference( 0.0 );

        if ( !theStepper.theStateFlag )
          {
            return 0.0;
          }

        const Real aStepIntervalInv( 1.0 / 
				     theStepper.getTolerableStepInterval() );

        const Real aTimeInterval1( aTime - theStepper.getCurrentTime() );
        const Real aTimeInterval2( aTimeInterval1 - anInterval );

        const Real theta1( aTimeInterval1 * aStepIntervalInv );
        const Real theta2( aTimeInterval2 * aStepIntervalInv );

        RealMatrixCref aTaylorSeries( theStepper.getTaylorSeries() );
	const RealMatrix::size_type aTaylorSize( aTaylorSeries.size() );

        Real aFactorialInv1( aTimeInterval1 );
        Real aFactorialInv2( aTimeInterval2 );

	RealCptr aTaylorCoefficientPtr( aTaylorSeries.origin() + theIndex );
	const RealMatrix::size_type aStride( aTaylorSeries.strides()[0] );

	{
	  // aTaylorSeries[ 0 ][ theIndex ]
	  const Real aTaylorCoefficient( *aTaylorCoefficientPtr );
	  aDifference += aTaylorCoefficient * aFactorialInv1;
	  aDifference -= aTaylorCoefficient * aFactorialInv2;
	}

	for ( RealMatrix::size_type s( 1 ); s < aTaylorSize; ++s )
          {
	    // aTaylorSeries[ s ][ theIndex ]
	    aTaylorCoefficientPtr += aStride;
	    const Real aTaylorCoefficient( *aTaylorCoefficientPtr );

            aFactorialInv1 *= theta1;
            aFactorialInv2 *= theta2;

	    aDifference += aTaylorCoefficient * aFactorialInv1;
	    aDifference -= aTaylorCoefficient * aFactorialInv2;

	    // LIBECS_PREFETCH( aTaylorCoefficientPtr + aStride, 0, 1 );
          }

        return aDifference;
      }
      
    protected:

      DifferentialStepperRef    theStepper;
      VariableVector::size_type theIndex;

    };

  public:

    DifferentialStepper();
    virtual ~DifferentialStepper();

    SET_METHOD( Real, NextStepInterval )
    {
      theNextStepInterval = value;
    }

    GET_METHOD( Real, NextStepInterval )
    {
      return theNextStepInterval;
    }

    SET_METHOD( Real, TolerableStepInterval )
    {
      theTolerableStepInterval = value;
    }

    GET_METHOD( Real, TolerableStepInterval )
    {
      return theTolerableStepInterval;
    }

    void initializeStepInterval( RealParam aStepInterval )
    {
      setStepInterval( aStepInterval );
      setTolerableStepInterval( aStepInterval );
      setNextStepInterval( aStepInterval );
    }

    void resetAll();
    void interIntegrate();
 
    virtual void initialize();

    virtual void reset();

    virtual void interrupt( StepperPtr const aCaller );

    virtual InterpolantPtr createInterpolant( VariablePtr aVariable )
    {
      return new DifferentialStepper::Interpolant( *this, aVariable );
    }

    virtual GET_METHOD( Integer, Stage )
    { 
      return 1; 
    }

    RealMatrixCref getTaylorSeries() const
    {
      return theTaylorSeries;
    }

  protected:

    const bool isExternalErrorTolerable() const;

    RealMatrix theTaylorSeries;

  protected:

    bool theStateFlag;

  private:

    Real theNextStepInterval;
    Real theTolerableStepInterval;
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

	PROPERTYSLOT( Integer, IsEpsilonChecked,
		      &AdaptiveDifferentialStepper::setEpsilonChecked,
		      &AdaptiveDifferentialStepper::isEpsilonChecked );
	PROPERTYSLOT_SET_GET( Real, AbsoluteEpsilon );
	PROPERTYSLOT_SET_GET( Real, RelativeEpsilon );

	PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, MaxErrorRatio );
	PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Order );
      }

  public:

    AdaptiveDifferentialStepper();
    virtual ~AdaptiveDifferentialStepper();

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

    SET_METHOD( Integer, EpsilonChecked )
    {
      if ( value > 0 ) {
	theEpsilonChecked = true;
      }
      else {
	theEpsilonChecked = false;
      }
    }

    const Integer isEpsilonChecked() const
    {
      return theEpsilonChecked;
    }

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

    virtual GET_METHOD( Integer, Order )
    { 
      return 1; 
    }

    virtual void initialize();
    virtual void step();
    virtual bool calculate() = 0;

    virtual GET_METHOD( Integer, Stage )
    { 
      return 2;
    }

  private:

    Real safety;
    Real theTolerance;
    Real theAbsoluteToleranceFactor;
    Real theStateToleranceFactor;
    Real theDerivativeToleranceFactor;

    bool theEpsilonChecked;
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
