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

#ifndef __DAE_HPP
#define __DAE_HPP

#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

DECLARE_VECTOR( Integer, IntVector );

LIBECS_DM_CLASS( DAEStepper, DifferentialStepper )
{

 public:

  class Interpolant
    :
    public libecs::Interpolant
  {
  public:

    Interpolant( DAEStepperRef aStepper, 
		   VariablePtr const aVariablePtr )
      :
      libecs::Interpolant( aVariablePtr ),
      theStepper( aStepper ),
      theIndex( theStepper.getVariableIndex( aVariablePtr ) )
    {
      ; // do nothing
    }

    virtual const Real getDifference( RealParam aTime, 
				      RealParam anInterval ) const
    {
     const Real sq6( 2.4494897427831779 );  // sqrt( 6.0 )      
     const Real c1( ( 4.0 - sq6 ) / 10.0 - 1.0 );      
     const Real c2( ( 4.0 + sq6 ) / 10.0 - 1.0 );      

     if ( !theStepper.theStateFlag )      
       { 
         return 0.0;  
       }  

     const Real cont1( theStepper.getTaylorSeries()[ 0 ][ theIndex ] );
     const Real cont2( theStepper.getTaylorSeries()[ 1 ][ theIndex ] );
     const Real cont3( theStepper.getTaylorSeries()[ 2 ][ theIndex ] );

     const Real aStepInterval( theStepper.getStepInterval() );
     const Real aStepIntervalInv( 1.0 / aStepInterval );
     const Real aTimeInterval( aTime - aStepInterval    
			       - theStepper.getCurrentTime() ); 

     const Real s1( aTimeInterval * aStepIntervalInv );
     const Real s2( ( aTimeInterval - anInterval ) * aStepIntervalInv );

     const Real 
       i1( s1 * ( cont1 + ( s1 - c2 ) * ( cont2 + ( s1 - c1 ) * cont3 ) ) );
     const Real 
       i2( s2 * ( cont1 + ( s2 - c2 ) * ( cont2 + ( s2 - c1 ) * cont3 ) ) );

     return ( i1 - i2 );     
     //      return theStepper.getVelocityBuffer()[ theIndex ] * anInterval;
    }

  protected:

    DAEStepperRef         theStepper;
    UnsignedInteger       theIndex;

  };

public:

  LIBECS_DM_OBJECT( DAEStepper, Stepper )
    {
      INHERIT_PROPERTIES( DifferentialStepper );

      PROPERTYSLOT_SET_GET( Integer, MaxIterationNumber );
      PROPERTYSLOT_SET_GET( Real, Uround );

      PROPERTYSLOT_SET_GET( Real, AbsoluteTolerance );
      PROPERTYSLOT_SET_GET( Real, RelativeTolerance );

      PROPERTYSLOT_SET_GET( Real, JacobianRecalculateTheta );
    }

  DAEStepper( void );
  virtual ~DAEStepper( void );

  SET_METHOD( Integer, MaxIterationNumber )
    {
      theMaxIterationNumber = value;
    }

  GET_METHOD( Integer, MaxIterationNumber )
    {
      return theMaxIterationNumber;
    }

  SIMPLE_SET_GET_METHOD( Real, Uround );

  SET_METHOD( Real, AbsoluteTolerance )
    {
      theAbsoluteTolerance = value;

      const Real aRatio( theAbsoluteTolerance / theRelativeTolerance );
      rtoler = 0.1 * pow( theRelativeTolerance, 2.0 / 3.0 );
      atoler = rtoler * aRatio;
    }

  GET_METHOD( Real, AbsoluteTolerance )
    {
      return theAbsoluteTolerance;
    }

  SET_METHOD( Real, RelativeTolerance )
    {
      theRelativeTolerance = value;

      const Real aRatio( theAbsoluteTolerance / theRelativeTolerance );
      rtoler = 0.1 * pow( theRelativeTolerance, 2.0 / 3.0 );
      atoler = rtoler * aRatio;
    }

  GET_METHOD( Real, RelativeTolerance )
    {
      return theRelativeTolerance;
    }

  SET_METHOD( Real, JacobianRecalculateTheta )
    {
      theJacobianRecalculateTheta = value;
    }

  GET_METHOD( Real, JacobianRecalculateTheta )
    {
      return theJacobianRecalculateTheta;
    }

  virtual void initialize();
  bool calculate();
  virtual void step();

  void checkDependency();

  Real estimateLocalError();

  void calculateJacobian();

  void setJacobianMatrix();
  void decompJacobianMatrix();
  void calculateRhs();
  Real solve();

  virtual InterpolantPtr createInterpolant( VariablePtr aVariable )
  {
    return new DAEStepper::Interpolant( *this, aVariable );
  }

  //  virtual GET_METHOD( Integer, Order ) { return 4; }
  virtual GET_METHOD( Integer, Stage ) { return 4; }

protected:

  Real    alpha, beta, gamma;

  VariableVector::size_type     theSystemSize;

  // IntVector as std::vector<VariableVector::size_type>
  IntVector  theContinuousVariableVector;
  RealVector theDiscreteActivityBuffer;

  std::vector<RealVector>    theJacobian;

  gsl_matrix*        theJacobianMatrix1;
  gsl_permutation*   thePermutation1;
  gsl_vector*        theVelocityVector1;
  gsl_vector*        theSolutionVector1;

  gsl_matrix_complex*        theJacobianMatrix2;
  gsl_permutation*           thePermutation2;
  gsl_vector_complex*        theVelocityVector2;
  gsl_vector_complex*        theSolutionVector2;

  RealVector         theW;

  UnsignedInteger     theMaxIterationNumber;
  Real                theStoppingCriterion;
  Real                eta, Uround;

  Real    theAbsoluteTolerance, atoler;
  Real    theRelativeTolerance, rtoler;

  bool    theFirstStepFlag, theRejectedStepFlag;
  Real    theAcceptedError, theAcceptedStepInterval, thePreviousStepInterval;

  bool    theJacobianCalculateFlag;
  Real    theJacobianRecalculateTheta;

};

#endif /* __DAE_HPP */
