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

#ifndef __DAE_HPP
#define __DAE_HPP

#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( DAEStepper, DifferentialStepper )
{

public:

  LIBECS_DM_OBJECT( DAEStepper, Stepper )
    {
      INHERIT_PROPERTIES( DifferentialStepper );

      PROPERTYSLOT_SET_GET( Int, MaxIterationNumber );
      PROPERTYSLOT_SET_GET( Real, Uround );

      PROPERTYSLOT_SET_GET( Real, AbsoluteTolerance );
      PROPERTYSLOT_SET_GET( Real, RelativeTolerance );

      PROPERTYSLOT_SET_GET( Real, JacobianRecalculateTheta );
    }

  DAEStepper( void );
  virtual ~DAEStepper( void );

  SET_METHOD( Int, MaxIterationNumber )
    {
      theMaxIterationNumber = value;
    }

  GET_METHOD( Int, MaxIterationNumber )
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

  Real estimateLocalError();

  void calculateJacobian();

  void setJacobianMatrix();
  void decompJacobianMatrix();
  void calculateVelocityVector();
  Real solve();

protected:

  Real    alpha, beta, gamma;

  VariableVector::size_type     theSystemSize;

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
