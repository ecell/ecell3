//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2004 Keio University
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
// written by Tomoya Kitayama <tomo@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __FLUXDISTRIBUTIONSTEPPER_HPP
#define __FLUXDISTRIBUTIONSTEPPER_HPP

#define GSL_RANGE_CHECK_OFF

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include <map>

#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/DifferentialStepper.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( FluxDistributionStepper, DifferentialStepper )
{  

 public:
  
  LIBECS_DM_OBJECT( FluxDistributionStepper, DifferentialStepper )
    {
      INHERIT_PROPERTIES( Stepper );

      PROPERTYSLOT_SET_GET( Real, Epsilon );
    }
  
  FluxDistributionStepper();
  ~FluxDistributionStepper();

  SIMPLE_SET_GET_METHOD( Real, Epsilon );
  
  virtual void initialize();

  virtual void interrupt( StepperPtr const aCaller )
    {
      integrate( aCaller->getCurrentTime() );

      VariableVector::size_type aVariableVectorSize( theVariableVector.size() );  
      for( VariableVector::size_type i( 0 ); i < aVariableVectorSize; i++ )
	{      
	  gsl_vector_set( theVariableVelocityVector, i, theVariableVector[i]->getVelocity() );
	}
      
      clearVariables();
	  
      gsl_blas_dgemv( CblasNoTrans, -1.0, theInverseMatrix, 
		      theVariableVelocityVector, 0.0, theFluxVector );
      
      ProcessVector::size_type aProcessVectorSize( theProcessVector.size() );
      for( ProcessVector::size_type i( 0 ); i < aProcessVectorSize; ++i )
	{
	  theProcessVector[i]->setFlux( gsl_vector_get( theFluxVector, i ) );
	}
      
      for( UnsignedInteger c( 0 ); c < getReadOnlyVariableOffset(); ++c )
	{
	  theTaylorSeries[0][c] = theVariableVector[c]->getVelocity();
	}

      step();

      log();
    }
  
  virtual void step()
    {
      // do nothing.
    }
  
 protected:

  gsl_matrix* generateInverse( gsl_matrix *m_unknown, 
			       Integer matrix_size );

  gsl_matrix* theUnknownMatrix;
  gsl_matrix* theInverseMatrix;
  gsl_vector* theVariableVelocityVector;
  gsl_vector* theFluxVector;

  Integer theMatrixSize;
  Real Epsilon;

};

#endif /* __FLUXDISTRIBUTIONSTEPPER_HPP */
