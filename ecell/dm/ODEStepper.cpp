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
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-Cell Project, Lab. for Bioinformatics, Keio University.
//

#include "Variable.hpp"
#include "Process.hpp"

#define GSL_RANGE_CHECK_OFF

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "ODEStepper.hpp"

LIBECS_DM_INIT( ODEStepper, Stepper );

ODEStepper::ODEStepper()
  :
  theSystemSize( 0 ),
  theJacobianMatrix1( NULLPTR ),
  thePermutation1( NULLPTR ),
  theVelocityVector1( NULLPTR ),
  theSolutionVector1( NULLPTR ),
  theJacobianMatrix2( NULLPTR ),
  thePermutation2( NULLPTR ),
  theVelocityVector2( NULLPTR ),
  theSolutionVector2( NULLPTR ),
  theMaxIterationNumber( 7 ),
  eta( 1.0 ),
  Uround( 1e-16 ),
  theStoppingCriterion( 0.0 ),
  theFirstStepFlag( true ),
  theRejectedStepFlag( false ),
  theJacobianCalculateFlag( true ),
  theAcceptedError( 0.0 ),
  theAcceptedStepInterval( 0.0 ),
  thePreviousStepInterval( 0.001 ),
  theJacobianRecalculateTheta( 0.001 ),
  rtoler( 1e-10 ),
  atoler( 1e-10 )
{
  const Real pow913( pow( 9.0, 1.0 / 3.0 ) );
  
  alpha = ( 12.0 - pow913*pow913 + pow913 ) / 60.0;
  beta = ( pow913*pow913 + pow913 ) * sqrt( 3.0 ) / 60.0;
  gamma = ( 6.0 + pow913*pow913 - pow913 ) / 30.0;
  
  const Real aNorm( alpha*alpha + beta*beta );
  const Real aStepInterval( getStepInterval() );
  
  alpha /= aNorm;
  beta /= aNorm;
  gamma = 1.0 / gamma;
  
  rtoler = 0.1 * pow( getTolerance(), 2.0 / 3.0 );
  atoler = rtoler * getAbsoluteToleranceFactor();
}

ODEStepper::~ODEStepper()
{
  gsl_matrix_free( theJacobianMatrix1 );
  gsl_permutation_free( thePermutation1 );
  gsl_vector_free( theVelocityVector1 );
  gsl_vector_free( theSolutionVector1 );
  
  gsl_matrix_complex_free( theJacobianMatrix2 );
  gsl_permutation_free( thePermutation2 );
  gsl_vector_complex_free( theVelocityVector2 );
  gsl_vector_complex_free( theSolutionVector2 );
}

void ODEStepper::initialize()
{
  AdaptiveDifferentialStepper::initialize();
  
  eta = 1.0;
  theStoppingCriterion
    = std::max( 10.0 * Uround / rtoler,
		std::min( 0.03, sqrt( rtoler ) ) );
  
  theFirstStepFlag = true;
  theJacobianCalculateFlag = true;
  
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  if ( theSystemSize != aSize )
    {
      theSystemSize = aSize;
      
      theJacobian.resize( theSystemSize );
      for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
	theJacobian[ c ].resize( theSystemSize );
      
      if ( theJacobianMatrix1 )
	gsl_matrix_free( theJacobianMatrix1 );
      theJacobianMatrix1 = gsl_matrix_calloc( theSystemSize, theSystemSize );
      
      if ( thePermutation1 )
	gsl_permutation_free( thePermutation1 );
      thePermutation1 = gsl_permutation_alloc( theSystemSize );
      
      if ( theVelocityVector1 )
	gsl_vector_free( theVelocityVector1 );
      theVelocityVector1 = gsl_vector_calloc( theSystemSize );
      
      if ( theSolutionVector1 )
	gsl_vector_free( theSolutionVector1 );
      theSolutionVector1 = gsl_vector_calloc( theSystemSize );
      
      theW.resize( theSystemSize * 3 );
      cont.resize( theSystemSize * 3 );
      
      if ( theJacobianMatrix2 )
	gsl_matrix_complex_free( theJacobianMatrix2 );
      theJacobianMatrix2 = gsl_matrix_complex_calloc( theSystemSize, theSystemSize );
      
      if ( thePermutation2 )
	gsl_permutation_free( thePermutation2 );
      thePermutation2 = gsl_permutation_alloc( theSystemSize );
      
      if ( theVelocityVector2 )
	gsl_vector_complex_free( theVelocityVector2 );
      theVelocityVector2 = gsl_vector_complex_calloc( theSystemSize );
      
      if ( theSolutionVector2 )
	gsl_vector_complex_free( theSolutionVector2 );
      theSolutionVector2 = gsl_vector_complex_calloc( theSystemSize );
    }
}

void ODEStepper::calculateJacobian()
{
  Real aPerturbation;
  
  for ( VariableVector::size_type i( 0 ); i < theSystemSize; ++i )
    {
      const VariablePtr aVariable( theVariableVector[ i ] );
      const Real aValue( aVariable->getValue() );
      
      aPerturbation = sqrt( Uround * std::max( 1e-5, fabs( aValue ) ) );
      aVariable->loadValue( theValueBuffer[ i ] + aPerturbation );

      fireProcesses();
      
      for ( VariableVector::size_type j( 0 ); j < theSystemSize; ++j )
	{
	  const VariablePtr aVariable( theVariableVector[ j ] );
	  
	  theJacobian[ j ][ i ]
	    = - ( aVariable->getVelocity() - theVelocityBuffer[ j ] )
	    / aPerturbation;
	  aVariable->clearVelocity();
	}
      
      aVariable->loadValue( aValue );
    }
}

void ODEStepper::setJacobianMatrix()
{
  const Real aStepInterval( getStepInterval() );
  
  const Real alphah( alpha / aStepInterval );
  const Real betah( beta / aStepInterval );
  const Real gammah( gamma / aStepInterval );
  
  gsl_complex comp1, comp2;
  
  for ( RealVector::size_type i( 0 ); i < theSystemSize; i++ )
    for ( RealVector::size_type j( 0 ); j < theSystemSize; j++ )
      {
	const Real aPartialDerivative( theJacobian[ i ][ j ] );
	
	gsl_matrix_set( theJacobianMatrix1, i, j, aPartialDerivative );
	
	GSL_SET_COMPLEX( &comp1, aPartialDerivative, 0 );
	gsl_matrix_complex_set( theJacobianMatrix2, i, j, comp1 );
      }
  
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
    {
      const Real aPartialDerivative
	( gsl_matrix_get( theJacobianMatrix1, c, c ) );
      gsl_matrix_set( theJacobianMatrix1, c, c,
		      gammah + aPartialDerivative );
      
      comp1 = gsl_matrix_complex_get( theJacobianMatrix2, c, c );
      GSL_SET_COMPLEX( &comp2, alphah, betah );
      gsl_matrix_complex_set( theJacobianMatrix2, c, c,
			      gsl_complex_add( comp1, comp2 ) );	
    }
  
  decompJacobianMatrix();
}

void ODEStepper::decompJacobianMatrix()
{
  int aSignNum;
  
  gsl_linalg_LU_decomp( theJacobianMatrix1, thePermutation1, &aSignNum );
  gsl_linalg_complex_LU_decomp( theJacobianMatrix2, thePermutation2, &aSignNum );
}

Real ODEStepper::calculateJacobianNorm()
{
  Real anEuclideanNorm( 0.0 );

  for ( RealVector::size_type i( 0 ); i < theSystemSize; i++ )
    for ( RealVector::size_type j( 0 ); j < theSystemSize; j++ )
      {
	const Real aPartialDerivative( theJacobian[ i ][ j ] );
	anEuclideanNorm += aPartialDerivative * aPartialDerivative;
      }

  return sqrt( anEuclideanNorm );
}

void ODEStepper::calculateRhs()
{
  const Real aCurrentTime( getCurrentTime() );
  const Real aStepInterval( getStepInterval() );
  
  const Real alphah( alpha / aStepInterval );
  const Real betah( beta / aStepInterval );
  const Real gammah( gamma / aStepInterval );
  
  gsl_complex comp;
  
  RealVector tif( theSystemSize * 3 );
  
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const Real z( theW[ c ] * 0.091232394870892942792
		    - theW[ c + theSystemSize ] * 0.14125529502095420843
		    - theW[ c + 2*theSystemSize ] * 0.030029194105147424492 );
      
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + z );
    }
  
  // ========= 1 ===========
  
  setCurrentTime( aCurrentTime
		  + aStepInterval * ( 4.0 - sqrt( 6.0 ) ) / 10.0 );
  fireProcesses();
  
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const VariablePtr aVariable( theVariableVector[ c ] );
      
      tif[ c ] 
	= aVariable->getVelocity() * 4.3255798900631553510;
      tif[ c + theSystemSize ]
	= aVariable->getVelocity() * -4.1787185915519047273;
      tif[ c + theSystemSize*2 ]
	= aVariable->getVelocity() * -0.50287263494578687595;
      
      aVariable->clearVelocity();
      
      const Real z( theW[ c ] * 0.24171793270710701896
		    + theW[ c + theSystemSize ] * 0.20412935229379993199
		    + theW[ c + 2*theSystemSize ] * 0.38294211275726193779 );

      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + z );
    }
  
  // ========= 2 ===========
  
  setCurrentTime( aCurrentTime
		  + aStepInterval * ( 4.0 + sqrt( 6.0 ) ) / 10.0 );
  fireProcesses();
  
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const VariablePtr aVariable( theVariableVector[ c ] );
      
      tif[ c ] 
	+= aVariable->getVelocity() * 0.33919925181580986954;
      tif[ c + theSystemSize ]
	-= aVariable->getVelocity() * 0.32768282076106238708;
      tif[ c + theSystemSize*2 ]
	+= aVariable->getVelocity() * 2.5719269498556054292;
      
      aVariable->clearVelocity();
      
      const Real z( theW[ c ] * 0.96604818261509293619 + theW[ c + theSystemSize ] );

      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + z );
    }
  
  // ========= 3 ===========

  setCurrentTime( aCurrentTime + aStepInterval );
  fireProcesses();

  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const VariablePtr aVariable( theVariableVector[ c ] );
      
      tif[ c ]
	+= aVariable->getVelocity() * 0.54177053993587487119;
      tif[ c + theSystemSize ]
	+= aVariable->getVelocity() * 0.47662355450055045196;
      tif[ c + theSystemSize*2 ]
	-= aVariable->getVelocity() * 0.59603920482822492497;
      
      gsl_vector_set( theVelocityVector1, c,
		      tif[ c ] - theW[ c ] * gammah );

      GSL_SET_COMPLEX( &comp, tif[ c + theSystemSize ] - theW[ c + theSystemSize ] * alphah + theW[ c + theSystemSize*2 ] * betah, tif[ c + theSystemSize*2 ] - theW[ c + theSystemSize ] * betah - theW[ c + theSystemSize*2 ] * alphah );
      gsl_vector_complex_set( theVelocityVector2, c, comp );

      aVariable->clearVelocity();
    }

  setCurrentTime( aCurrentTime );
}

Real ODEStepper::solve()
{
  gsl_linalg_LU_solve( theJacobianMatrix1, thePermutation1,
		       theVelocityVector1, theSolutionVector1 );
  gsl_linalg_complex_LU_solve( theJacobianMatrix2, thePermutation2,
			       theVelocityVector2, theSolutionVector2 );
  
  Real aNorm( 0.0 );
  Real deltaW( 0.0 );
  gsl_complex comp;

  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      Real aTolerance2( rtoler * fabs( theValueBuffer[ c ] ) + atoler );
      aTolerance2 = aTolerance2 * aTolerance2;
      
      deltaW = gsl_vector_get( theSolutionVector1, c );
      theW[ c ] += deltaW;
      aNorm += deltaW * deltaW / aTolerance2;
      
      comp = gsl_vector_complex_get( theSolutionVector2, c );
      
      deltaW = GSL_REAL( comp );
      theW[ c + theSystemSize ] += deltaW;
      aNorm += deltaW * deltaW / aTolerance2;

      deltaW = GSL_IMAG( comp );
      theW[ c + theSystemSize*2 ] += deltaW;
      aNorm += deltaW * deltaW / aTolerance2;
    }

  return sqrt( aNorm / ( 3 * theSystemSize ) );
}

bool ODEStepper::calculate()
{
  const Real aStepInterval( getStepInterval() );
  Real aNewStepInterval;
  Real aNorm;
  Real theta( fabs( theJacobianRecalculateTheta ) );

  UnsignedInteger anIterator( 0 );
  
  const Real c1( ( 4.0 - sqrt( 6.0 ) ) / 10.0 );
  const Real c2( ( 4.0 + sqrt( 6.0 ) ) / 10.0 );
  const Real c3q( getStepInterval() / thePreviousStepInterval );
  const Real c1q( c3q * c1 );
  const Real c2q( c3q * c2 );
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const Real z1( c1q * ( cont[ c ] + ( c1q - c2 ) * ( cont[ c + theSystemSize ] + ( c1q - c1 ) * cont[ c + theSystemSize*2 ] ) ) );
      const Real z2( c2q * ( cont[ c ] + ( c2q - c2 ) * ( cont[ c + theSystemSize ] + ( c2q - c1 ) * cont[ c + theSystemSize*2 ] ) ) );
      const Real z3( c3q * ( cont[ c ] + ( c3q - c2 ) * ( cont[ c + theSystemSize ] + ( c3q - c1 ) * cont[ c + theSystemSize*2 ] ) ) );
      
      theW[ c ] = 4.3255798900631553510 * z1
	+ 0.33919925181580986954 * z2 + 0.54177053993587487119 * z3;
      theW[ c+theSystemSize ] = -4.1787185915519047273 * z1
	- 0.32768282076106238708 * z2 + 0.47662355450055045196 * z3;
      theW[ c+theSystemSize*2 ] =  -0.50287263494578687595 * z1
	+ 2.5719269498556054292 * z2 - 0.59603920482822492497 * z3;
    }

  eta = pow( std::max( eta, Uround ), 0.8 );
  
  while ( anIterator < getMaxIterationNumber() )
    {
      calculateRhs();
      
      const Real previousNorm( std::max( aNorm, Uround ) );
      aNorm = solve();
      
      if ( anIterator > 0 && ( anIterator != getMaxIterationNumber()-1 ) )
	{
	  const Real aThetaQ = aNorm / previousNorm;
	  if ( anIterator > 1 )
	    theta = sqrt( aThetaQ * theta );
	  else
	    theta = aThetaQ;
	  
	  if ( theta < 0.99 )
	    {
	      eta = theta / ( 1.0 - theta );
	      const Real anIterationError( eta * aNorm * pow( theta, getMaxIterationNumber() - 2 - anIterator ) / theStoppingCriterion );
	      
	      if ( anIterationError >= 1.0 )
		{
		  aNewStepInterval = aStepInterval * 0.8 * pow( std::max( 1e-4, std::min( 20.0, anIterationError ) ) , -1.0 / ( 4 + getMaxIterationNumber() - 2 - anIterator ) );
		  setStepInterval( aNewStepInterval );

		  return false;
		}
	    }
	  else
	    {
	      setStepInterval( aStepInterval * 0.5 );
	      return false;
	    }
	}
      
      if ( eta * aNorm <= theStoppingCriterion )
	{
	  break;
	}
      
      anIterator++;
    }
   
  // theW is transformed to Z-form
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const Real w1( theW[ c ] );
      const Real w2( theW[ c + theSystemSize ] );
      const Real w3( theW[ c + theSystemSize*2 ] );
      
      theW[ c ] = w1 * 0.091232394870892942792
	- w2 * 0.14125529502095420843
	- w3 * 0.030029194105147424492;
      theW[ c + theSystemSize ] = w1 * 0.24171793270710701896
	+ w2 * 0.20412935229379993199
	+ w3 * 0.38294211275726193779;
      theW[ c + theSystemSize*2 ] = w1 * 0.96604818261509293619 + w2;
    }

  const Real anError( estimateLocalError() );

  Real aSafetyFactor( std::min( 0.9, 0.9 * ( 1 + 2*getMaxIterationNumber() ) / ( anIterator + 1 + 2*getMaxIterationNumber() ) ) );
  aSafetyFactor = std::max( 0.125, std::min( 5.0, pow( anError, 0.25 ) / aSafetyFactor ) );
  
  aNewStepInterval = aStepInterval / aSafetyFactor;
  
  if ( anError < 1.0 )
    {
      // step is accepted
      
      if ( !theFirstStepFlag )
	{
	  Real aGustafssonFactor( theAcceptedStepInterval / aStepInterval * pow( anError * anError / theAcceptedError, 0.25 ) / 0.9 );
	  aGustafssonFactor = std::max( 0.125, std::min( 5.0, aGustafssonFactor ) );
	  
	  if ( aSafetyFactor < aGustafssonFactor )
	    {
	      aSafetyFactor = aGustafssonFactor;
	      aNewStepInterval = aStepInterval / aGustafssonFactor;
	    }
	}
      
      theAcceptedStepInterval = aStepInterval;
      theAcceptedError = std::max( 1.0e-2, anError );
      
      if ( theRejectedStepFlag )
	aNewStepInterval = std::min( aNewStepInterval, aStepInterval );
      
      theFirstStepFlag = false;
      
      const Real aStepIntervalRate( aNewStepInterval / aStepInterval );

      if ( theta <= theJacobianRecalculateTheta )
	theJacobianCalculateFlag = false;
      else
	theJacobianCalculateFlag = true;

      if ( aStepIntervalRate >= 1.0 && aStepIntervalRate <= 1.2 )
	{
	  setNextStepInterval( aStepInterval );
	  return true;
	}
      
      setNextStepInterval( aNewStepInterval );
      return true;
    }
  else
    {
      // step is rejected

      if ( theFirstStepFlag )
	{
	  setStepInterval( 0.1 * aStepInterval );
	}
      else
	{
	  setStepInterval( aNewStepInterval );
	}
      
      return false;
    }
}

Real ODEStepper::estimateLocalError()
{
  const Real aStepInterval( getStepInterval() );
  
  Real anError;
  
  const Real hee1( ( -13.0 - 7.0 * sqrt( 6.0 ) ) / ( 3.0 * aStepInterval ) );
  const Real hee2( ( -13.0 + 7.0 * sqrt( 6.0 ) ) / ( 3.0 * aStepInterval ) );
  const Real hee3( -1.0 / ( 3.0 * aStepInterval ) );
  
  // theW will already be transformed to Z-form
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
    {
      gsl_vector_set( theVelocityVector1, c,
		      theVelocityBuffer[ c ]
		      + theW[ c ] * hee1
		      + theW[ c + theSystemSize ] * hee2
		      + theW[ c + 2*theSystemSize ] * hee3 );
    }

  gsl_linalg_LU_solve( theJacobianMatrix1, thePermutation1,
		       theVelocityVector1, theSolutionVector1 );
  
  anError = 0.0;
  Real aDifference;
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      const Real aTolerance( rtoler * fabs( theValueBuffer[ c ] ) + atoler );
      aDifference = gsl_vector_get( theSolutionVector1, c );

      // for the case ( anError >= 1.0 )
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + aDifference );
      
      aDifference /= aTolerance;
      anError += aDifference * aDifference;
    }
  
  anError = std::max( sqrt( anError / theSystemSize ), 1e-10 );

  if ( anError < 1.0 ) return anError;
  
  if ( theFirstStepFlag or theRejectedStepFlag )
    {
      fireProcesses();
      
      for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
	{
	  gsl_vector_set( theVelocityVector1, c,
			  theVariableVector[ c ]->getVelocity()
			  + theW[ c ] * hee1
			  + theW[ c + theSystemSize ] * hee2
			  + theW[ c + 2*theSystemSize ] * hee3 );
	  
	  theVariableVector[ c ]->clearVelocity();
	}
      
      gsl_linalg_LU_solve( theJacobianMatrix1, thePermutation1,
			   theVelocityVector1, theSolutionVector1 );
      
      anError = 0.0;
      for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
	{
	  const Real aTolerance( rtoler * fabs( theValueBuffer[ c ] )
				 + atoler );
	  
	  Real aDifference( gsl_vector_get( theSolutionVector1, c ) );
	  aDifference /= aTolerance;
	  
	  anError += aDifference * aDifference;
	}
      
      anError = std::max( sqrt( anError / theSystemSize ), 1e-10 );
    }

  return anError;
}

void ODEStepper::step()
{
  theStateFlag = false;
  
  thePreviousStepInterval = getStepInterval();
  setStepInterval( getNextStepInterval() );
  clearVariables();
  //    interIntegrate();
  
  theRejectedStepFlag = false;

  fireProcesses();  
  for ( VariableVector::size_type i( 0 ); i < theSystemSize; ++i )
    {
      theVelocityBuffer[ i ] = theVariableVector[ i ]->getVelocity();
      theVariableVector[ i ]->clearVelocity();
    }
  
  if ( theJacobianCalculateFlag )
    {
      calculateJacobian();
      setJacobianMatrix();
    }
  else
    {
      if ( thePreviousStepInterval != getStepInterval() )
	setJacobianMatrix();
    }

  while ( !calculate() )
    {
      theRejectedStepFlag = true;

      if ( !theJacobianCalculateFlag )
	{
	  calculateJacobian();
	  theJacobianCalculateFlag = true;
	}

      setJacobianMatrix();
    }
  
  const Real aStepInterval( getStepInterval() );
  
  // theW will already be transformed to Z-form

  for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
      theVelocityBuffer[ c ] = theW[ c + theSystemSize*2 ];
      theVelocityBuffer[ c ] /= aStepInterval;
      
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] );
      theVariableVector[ c ]->setVelocity( theVelocityBuffer[ c ] );
    }
  
  const Real c1( ( 4.0 - sqrt( 6.0 ) ) / 10.0 );
  const Real c2( ( 4.0 + sqrt( 6.0 ) ) / 10.0 );
  
  for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
    {
      const Real z1( theW[ c ] );
      const Real z2( theW[ c + theSystemSize ] );
      const Real z3( theW[ c + theSystemSize*2 ] );
      
      cont[ c ] = ( z2 - z3 ) / ( c2 - 1.0 );
      
      const Real ak( ( z1 - z2 ) / ( c1 - c2 ) );
      Real acont3 = z1 / c1;
      acont3 = ( ak - acont3 ) / c2;
      
      cont[ c+theSystemSize ] = ( ak - cont[ c ] ) / ( c1 - 1.0 );
      cont[ c+theSystemSize*2 ] = cont[ c+theSystemSize ] - acont3;
    }

  theStateFlag = true;
}
