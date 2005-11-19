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
// written by Koichi Takahashi <shafi@e-cell.org> at
// E-Cell Project, Lab. for Bioinformatics, Keio University.
//

#ifndef __ODE_HPP
#define __ODE_HPP

#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ODEStepper, AdaptiveDifferentialStepper )
{

public:

  LIBECS_DM_OBJECT( ODEStepper, Stepper )
    {
      INHERIT_PROPERTIES( AdaptiveDifferentialStepper );

      PROPERTYSLOT_SET_GET( Integer, MaxIterationNumber );
      PROPERTYSLOT_SET_GET( Real, Uround );
      
      PROPERTYSLOT( Real, Tolerance,
		    &ODEStepper::initializeTolerance,
		    &AdaptiveDifferentialStepper::getTolerance );

      PROPERTYSLOT( Real, AbsoluteToleranceFactor,
		    &ODEStepper::initializeAbsoluteToleranceFactor,
		    &AdaptiveDifferentialStepper::getAbsoluteToleranceFactor );
      

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Stiffness );
      PROPERTYSLOT_SET_GET( Real, JacobianRecalculateTheta );

      PROPERTYSLOT( Integer, isStiff,
		    &ODEStepper::setIntegrationType,
		    &ODEStepper::getIntegrationType );

      PROPERTYSLOT_SET_GET( Integer, CheckIntervalCount );
      PROPERTYSLOT_SET_GET( Integer, SwitchingCount );
    }

  ODEStepper( void );
  virtual ~ODEStepper( void );

  SET_METHOD( Integer, MaxIterationNumber )
    {
      theMaxIterationNumber = value;
    }

  GET_METHOD( Integer, MaxIterationNumber )
    {
      return theMaxIterationNumber;
    }

  SIMPLE_SET_GET_METHOD( Real, Uround );

  SIMPLE_SET_GET_METHOD( Integer, CheckIntervalCount );
  SIMPLE_SET_GET_METHOD( Integer, SwitchingCount );

  void setIntegrationType( Integer value )
    {
      isStiff = static_cast<bool>( value );
      initializeStepper();
    }

  const Integer getIntegrationType() const { return isStiff; }
  
  SET_METHOD( Real, JacobianRecalculateTheta )
    {
      theJacobianRecalculateTheta = value;
    }

  GET_METHOD( Real, JacobianRecalculateTheta )
    {
      return theJacobianRecalculateTheta;
    }

  GET_METHOD( Real, Stiffness )
  {
    return 3.3 / theSpectralRadius;
  }

  GET_METHOD( Real, SpectralRadius )
  {
    return theSpectralRadius;
  }

  SET_METHOD( Real, SpectralRadius )
  {
    theSpectralRadius = value;
  }

  virtual void initialize();
  virtual void step();
  virtual bool calculate();
  virtual void interrupt( TimeParam aTime );

  void initializeStepper();

  void calculateJacobian();
  Real calculateJacobianNorm();
  void setJacobianMatrix();
  void decompJacobianMatrix();
  void calculateRhs();
  Real solve();
  Real estimateLocalError();

  void initializeRadauIIA();
  bool calculateRadauIIA();
  void stepRadauIIA();

  void initializeTolerance( RealParam value )
  {
    setTolerance( value ); // AdaptiveDifferentialStepper::
    rtoler = 0.1 * pow( getTolerance(), 2.0 / 3.0 );
    atoler = rtoler * getAbsoluteToleranceFactor();
  }

  void initializeAbsoluteToleranceFactor( RealParam value )
  {
    setAbsoluteToleranceFactor( value ); // AdaptiveDifferentialStepper::
    atoler = rtoler * getAbsoluteToleranceFactor();
  }

  virtual GET_METHOD( Integer, Order )
  {
    if ( isStiff ) return 3;
    else return 4;
  }

  virtual GET_METHOD( Integer, Stage )
  {
    return 4;
  }

protected:

  Real    alpha, beta, gamma;

  VariableVector::size_type     theSystemSize;

  RealMatrix    theJacobian, theW;

  gsl_matrix*        theJacobianMatrix1;
  gsl_permutation*   thePermutation1;
  gsl_vector*        theVelocityVector1;
  gsl_vector*        theSolutionVector1;

  gsl_matrix_complex*        theJacobianMatrix2;
  gsl_permutation*           thePermutation2;
  gsl_vector_complex*        theVelocityVector2;
  gsl_vector_complex*        theSolutionVector2;

  UnsignedInteger    theMaxIterationNumber;
  Real               theStoppingCriterion;
  Real               eta, Uround;

  Real               rtoler, atoler;

  Real    theAcceptedError, theAcceptedStepInterval, thePreviousStepInterval;

  Real    theJacobianRecalculateTheta;
  Real    theSpectralRadius;

  UnsignedInteger    theStiffnessCounter;
  Integer    CheckIntervalCount, SwitchingCount;

  bool    theFirstStepFlag, theJacobianCalculateFlag, theRejectedStepFlag;
  bool    isInterrupted, isStiff;
};

#endif /* __ODE_HPP */
