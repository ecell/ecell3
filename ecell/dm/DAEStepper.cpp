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

#include "Variable.hpp"
#include "Process.hpp"

#define GSL_RANGE_CHECK_OFF

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "DAEStepper.hpp"

#define SQRT6 2.4494897427831779

LIBECS_DM_INIT( DAEStepper, Stepper );

DAEStepper::DAEStepper()
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
  Uround( 1e-10 ),
  theAbsoluteTolerance( 1e-6 ),
  theRelativeTolerance( 1e-6 ),
  theStoppingCriterion( 0.0 ),
  theFirstStepFlag( true ),
  theRejectedStepFlag( false ),
  theJacobianCalculateFlag( true ),
  theAcceptedError( 0.0 ),
  theAcceptedStepInterval( 0.0 ),
  thePreviousStepInterval( 0.001 ),
  theJacobianRecalculateTheta( 0.001 ),
  theContinuousVariableVector( NULLPTR ),
  isInterrupted( true )
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
  
  const Real aRatio( theAbsoluteTolerance / theRelativeTolerance );
  rtoler = 0.1 * pow( theRelativeTolerance, 2.0 / 3.0 );
  atoler = rtoler * aRatio;
}
	    
DAEStepper::~DAEStepper()
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

void DAEStepper::initialize()
{
  DifferentialStepper::initialize();
  
  eta = 1.0;
  theStoppingCriterion
    = std::max( 10.0 * Uround / rtoler,
		std::min( 0.03, sqrt( rtoler ) ) );

  theFirstStepFlag = true;
  theJacobianCalculateFlag = true;
  isInterrupted = true;

  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  if ( theSystemSize != aSize )
    {
      checkDependency();

      theSystemSize = theContinuousVariableVector.size()
	+ theProcessVector.size() - getDiscreteProcessOffset();

      if ( aSize != theSystemSize )
	THROW_EXCEPTION( InitializationFailed,
			 "definitions are required, are given." );

      theJacobian.resize( aSize );
      for ( VariableVector::size_type c( 0 ); c < aSize; c++ )
	theJacobian[ c ].resize( aSize );

      if ( theJacobianMatrix1 )
	gsl_matrix_free( theJacobianMatrix1 );
      theJacobianMatrix1 = gsl_matrix_calloc( aSize, aSize );

      if ( thePermutation1 )
	gsl_permutation_free( thePermutation1 );
      thePermutation1 = gsl_permutation_alloc( aSize );

      if ( theVelocityVector1 )
	gsl_vector_free( theVelocityVector1 );
      theVelocityVector1 = gsl_vector_calloc( aSize );

      if ( theSolutionVector1 )
	gsl_vector_free( theSolutionVector1 );
      theSolutionVector1 = gsl_vector_calloc( aSize );

      theW.resize( aSize * 3 );

      if ( theJacobianMatrix2 )
	gsl_matrix_complex_free( theJacobianMatrix2 );
      theJacobianMatrix2 = gsl_matrix_complex_calloc( aSize, aSize );

      if ( thePermutation2 )
	gsl_permutation_free( thePermutation2 );
      thePermutation2 = gsl_permutation_alloc( aSize );

      if ( theVelocityVector2 )
	gsl_vector_complex_free( theVelocityVector2 );
      theVelocityVector2 = gsl_vector_complex_calloc( aSize );

      if ( theSolutionVector2 )
	gsl_vector_complex_free( theSolutionVector2 );
      theSolutionVector2 = gsl_vector_complex_calloc( aSize );
    }
}

void DAEStepper::checkDependency()
{
  theContinuousVariableVector.clear();

  // FOR_ALL( ProcessVector, theProcessVector )
  const ProcessVector::size_type
    aDiscreteProcessOffset( getDiscreteProcessOffset() );
  for ( ProcessVector::size_type c( 0 ); c < aDiscreteProcessOffset; c++ )
    {
      ProcessPtr aProcess( theProcessVector[ c ] );
      
      VariableReferenceVectorCref
	aVariableReferenceVector( aProcess->getVariableReferenceVector() );

      const VariableReferenceVector::size_type
	aPositiveVariableReferenceOffset
	( aProcess->getPositiveVariableReferenceOffset() );
      const VariableReferenceVector::size_type
	aZeroVariableReferenceOffset
	( aProcess->getZeroVariableReferenceOffset() );

      const bool aContinuity( aProcess->isContinuous() );
      if ( aContinuity ) {
	for ( VariableReferenceVector::size_type i( 0 );
	      i < aVariableReferenceVector.size(); i++)
	  {
	    if ( i == aZeroVariableReferenceOffset )
	      {
		if ( aPositiveVariableReferenceOffset
		     == aVariableReferenceVector.size() )
		  break;
		else
		  i = aPositiveVariableReferenceOffset;
	      }
	    
	    VariablePtr const 
	      aVariable( aVariableReferenceVector[ i ].getVariable() );
	    const VariableVector::size_type
	      anIndex( getVariableIndex( aVariable ) );
	    
	    if ( std::find( theContinuousVariableVector.begin(),
			    theContinuousVariableVector.end(), anIndex )
		 == theContinuousVariableVector.end() )
	      theContinuousVariableVector.push_back( anIndex );
	  }
      }
    }

  std::sort( theContinuousVariableVector.begin(),
	     theContinuousVariableVector.end() );
  
  theDiscreteActivityBuffer.clear();
  theDiscreteActivityBuffer.resize( theProcessVector.size() 
				    - aDiscreteProcessOffset );
}

void DAEStepper::calculateJacobian()
{
  UnsignedInteger aSize( getReadOnlyVariableOffset() );
  Real aPerturbation;

  for ( VariableVector::size_type i( 0 ); i < aSize; ++i )
    {
      const VariablePtr aVariable( theVariableVector[ i ] );
      const Real aValue( aVariable->getValue() );
      
      aPerturbation = sqrt( Uround * std::max( 1e-5, fabs( aValue ) ) );
      aVariable->loadValue( theValueBuffer[ i ] + aPerturbation );

      fireProcesses();
      setVariableVelocity( theTaylorSeries[ 4 ] );

      const ProcessVector::size_type
	aDiscreteProcessOffset( getDiscreteProcessOffset() );

      for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	    c < theProcessVector.size(); c++ )
	{
	  const ProcessVector::size_type
	    anIndex( c - aDiscreteProcessOffset );

	  theJacobian[ theContinuousVariableVector.size() + anIndex ][ i ]
	    = - ( theProcessVector[ c ]->getActivity()
		  - theDiscreteActivityBuffer[ anIndex ] ) / aPerturbation;
	}

      for ( IntVector::size_type j( 0 );
	    j < theContinuousVariableVector.size(); ++j )
	{
	  // int as VariableVector::size_type
	  const int anIndex( theContinuousVariableVector[ j ] );
	  theJacobian[ j ][ i ] = - ( theTaylorSeries[ 4 ][ anIndex ] - theTaylorSeries[ 3 ][ anIndex ] ) / aPerturbation;
	}
      
      aVariable->loadValue( aValue );
    }
}

void DAEStepper::setJacobianMatrix()
{
  VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  const Real aStepInterval( getStepInterval() );
  
  const Real alphah( alpha / aStepInterval );
  const Real betah( beta / aStepInterval );
  const Real gammah( gamma / aStepInterval );

  /**
    gsl_complex comp;
     
    for ( VariableVector::size_type i( 0 ); i < aSize; ++i )
      for ( VariableVector::size_type j( 0 ); j < aSize; ++j )
	{
	  const Real aPartialDerivative( theJacobian[ i ][ j ] );

	  if ( i == j ) {
	    gsl_matrix_set( theJacobianMatrix1, i, j,
			    gammah - aPartialDerivative );

	    GSL_SET_COMPLEX( &comp, alphah - aPartialDerivative, betah );
	    gsl_matrix_complex_set( theJacobianMatrix2, i, j, comp );
	  }
	  else {
	    gsl_matrix_set( theJacobianMatrix1, i, j,
			    -1.0 * aPartialDerivative );

	    GSL_SET_COMPLEX( &comp, -1.0 * aPartialDerivative, 0.0 );
	    gsl_matrix_complex_set( theJacobianMatrix2, i, j, comp );
	  }
	}
    */

  gsl_complex comp1, comp2;
  
  for ( RealVector::size_type i( 0 ); i < theSystemSize; i++ )
    for ( RealVector::size_type j( 0 ); j < theSystemSize; j++ )
      {
	const Real aPartialDerivative( theJacobian[ i ][ j ] );
	
	gsl_matrix_set( theJacobianMatrix1, i, j, aPartialDerivative );
	
	GSL_SET_COMPLEX( &comp1, aPartialDerivative, 0 );
	gsl_matrix_complex_set( theJacobianMatrix2, i, j, comp1 );
      }

  for ( IntVector::size_type c( 0 );
	c < theContinuousVariableVector.size(); c++ )
    {
      const int anIndex( theContinuousVariableVector[ c ] );
	
      const Real aPartialDerivative
	( gsl_matrix_get( theJacobianMatrix1, c, anIndex ) );
      gsl_matrix_set( theJacobianMatrix1, c, anIndex,
		      gammah + aPartialDerivative );

      comp1 = gsl_matrix_complex_get( theJacobianMatrix2, c, anIndex );
      GSL_SET_COMPLEX( &comp2, alphah, betah );
      gsl_matrix_complex_set( theJacobianMatrix2, c, anIndex,
			      gsl_complex_add( comp1, comp2 ) );	
    }

  decompJacobianMatrix();
}

void DAEStepper::decompJacobianMatrix()
{
  int aSignNum;

  gsl_linalg_LU_decomp( theJacobianMatrix1, thePermutation1, &aSignNum );
  gsl_linalg_complex_LU_decomp( theJacobianMatrix2, thePermutation2, &aSignNum );
}

void DAEStepper::calculateRhs()
{
  const Real aCurrentTime( getCurrentTime() );
  const Real aStepInterval( getStepInterval() );
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  const ProcessVector::size_type
    aDiscreteProcessOffset( getDiscreteProcessOffset() );

  const Real alphah( alpha / aStepInterval );
  const Real betah( beta / aStepInterval );
  const Real gammah( gamma / aStepInterval );

  gsl_complex comp;

  RealVector aTIF;
  aTIF.resize( aSize * 3 );

  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      const Real z( theW[ c ] * 0.091232394870892942792
		    - theW[ c + aSize ] * 0.14125529502095420843
		    - theW[ c + 2*aSize ] * 0.030029194105147424492 );
      
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + z );
    }

  // ========= 1 ===========
  
  setCurrentTime( aCurrentTime
		  + aStepInterval * ( 4.0 - SQRT6 ) / 10.0 );
  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 4 ] );

  for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	c < theProcessVector.size(); c++ )
    {
      const ProcessVector::size_type anIndex
	( theContinuousVariableVector.size() + c - aDiscreteProcessOffset );
      const ProcessPtr aProcess( theProcessVector[ c ] );
      
      aTIF[ anIndex ] 
	= aProcess->getActivity() * 4.3255798900631553510;
      aTIF[ anIndex + aSize ]
	= aProcess->getActivity() * -4.1787185915519047273;
      aTIF[ anIndex + aSize*2 ]
	= aProcess->getActivity() * -0.50287263494578687595;
    }

  for ( IntVector::size_type c( 0 );
	c < theContinuousVariableVector.size(); c++ )
    {
      const int anIndex( theContinuousVariableVector[ c ] );
      
      aTIF[ c ] 
	= theTaylorSeries[ 4 ][ anIndex ] * 4.3255798900631553510;
      aTIF[ c + aSize ]
	= theTaylorSeries[ 4 ][ anIndex ] * -4.1787185915519047273;
      aTIF[ c + aSize*2 ]
	= theTaylorSeries[ 4 ][ anIndex ] * -0.50287263494578687595;
    }

  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      const Real z( theW[ c ] * 0.24171793270710701896
		    + theW[ c + aSize ] * 0.20412935229379993199
		    + theW[ c + 2*aSize ] * 0.38294211275726193779 );
      
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + z );
    }
  
  // ========= 2 ===========
  
  setCurrentTime( aCurrentTime
		  + aStepInterval * ( 4.0 + SQRT6 ) / 10.0 );
  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 4 ] );

  for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	c < theProcessVector.size(); c++ )
    {
      const ProcessVector::size_type anIndex
	( theContinuousVariableVector.size() + c - aDiscreteProcessOffset );
      const ProcessPtr aProcess( theProcessVector[ c ] );
      
      aTIF[ anIndex ] 
	+= aProcess->getActivity() * 0.33919925181580986954;
      aTIF[ anIndex + aSize ]
	-= aProcess->getActivity() * 0.32768282076106238708;
      aTIF[ anIndex + aSize*2 ]
	+= aProcess->getActivity() * 2.5719269498556054292;
    }

  for ( IntVector::size_type c( 0 );
	c < theContinuousVariableVector.size(); c++ )
    {
      const int anIndex( theContinuousVariableVector[ c ] );

      aTIF[ c ] 
	+= theTaylorSeries[ 4 ][ anIndex ] * 0.33919925181580986954;
      aTIF[ c + aSize ]
	-= theTaylorSeries[ 4 ][ anIndex ] * 0.32768282076106238708;
      aTIF[ c + aSize*2 ]
	+= theTaylorSeries[ 4 ][ anIndex ] * 2.5719269498556054292;
    }

  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      const Real z( theW[ c ] * 0.96604818261509293619 + theW[ c + aSize ] );
	
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + z );
    }

  // ========= 3 ===========

  setCurrentTime( aCurrentTime + aStepInterval );
  fireProcesses();
  setVariableVelocity( theTaylorSeries[ 4 ] );

  for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	c < theProcessVector.size(); c++ )
    {
      const ProcessVector::size_type anIndex
	( theContinuousVariableVector.size() + c - aDiscreteProcessOffset );
      const ProcessPtr aProcess( theProcessVector[ c ] );

      aTIF[ anIndex ]
	+= aProcess->getActivity() * 0.54177053993587487119;
      aTIF[ anIndex + aSize ]
	+= aProcess->getActivity() * 0.47662355450055045196;
      aTIF[ anIndex + aSize*2 ]
	-= aProcess->getActivity() * 0.59603920482822492497;

      gsl_vector_set( theVelocityVector1, anIndex, aTIF[ anIndex ] );

      GSL_SET_COMPLEX( &comp, aTIF[ anIndex + aSize ], 
		       aTIF[ anIndex + aSize*2 ] );
      gsl_vector_complex_set( theVelocityVector2, anIndex, comp );
    }

  for ( IntVector::size_type c( 0 );
	c < theContinuousVariableVector.size(); c++ )
    {
      const int anIndex( theContinuousVariableVector[ c ] );

      aTIF[ c ]
	+= theTaylorSeries[ 4 ][ anIndex ] * 0.54177053993587487119;
      aTIF[ c + aSize ]
	+= theTaylorSeries[ 4 ][ anIndex ] * 0.47662355450055045196;
      aTIF[ c + aSize*2 ]
	-= theTaylorSeries[ 4 ][ anIndex ] * 0.59603920482822492497;

      gsl_vector_set( theVelocityVector1, c,
		      aTIF[ c ] - theW[ anIndex ] * gammah );

      GSL_SET_COMPLEX( &comp, aTIF[ c + aSize ] - theW[ anIndex + aSize ] * alphah + theW[ anIndex + aSize*2 ] * betah, aTIF[ c + aSize*2 ] - theW[ anIndex + aSize ] * betah - theW[ anIndex + aSize*2 ] * alphah );
      gsl_vector_complex_set( theVelocityVector2, c, comp );
    }

  setCurrentTime( aCurrentTime );
}

Real DAEStepper::solve()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

  gsl_linalg_LU_solve( theJacobianMatrix1, thePermutation1,
		       theVelocityVector1, theSolutionVector1 );
  gsl_linalg_complex_LU_solve( theJacobianMatrix2, thePermutation2,
			       theVelocityVector2, theSolutionVector2 );

  Real aNorm( 0.0 );
  Real deltaW( 0.0 );
  gsl_complex comp;

  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      Real aTolerance2( rtoler * fabs( theValueBuffer[ c ] ) + atoler );

      aTolerance2 = aTolerance2 * aTolerance2;
      
      deltaW = gsl_vector_get( theSolutionVector1, c );
      theW[ c ] += deltaW;
      aNorm += deltaW * deltaW / aTolerance2;

      comp = gsl_vector_complex_get( theSolutionVector2, c );

      deltaW = GSL_REAL( comp );
      theW[ c + aSize ] += deltaW;
      aNorm += deltaW * deltaW / aTolerance2;

      deltaW = GSL_IMAG( comp );
      theW[ c + aSize*2 ] += deltaW;
      aNorm += deltaW * deltaW / aTolerance2;
    }

  return sqrt( aNorm / ( 3 * aSize ) );
}

bool DAEStepper::calculate()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  const Real aStepInterval( getStepInterval() );
  Real aNewStepInterval;
  Real aNorm;
  Real theta( fabs( theJacobianRecalculateTheta ) );

  UnsignedInteger anIterator( 0 );

  if ( !isInterrupted )
    {
      const Real c1( ( 4.0 - SQRT6 ) / 10.0 );
      const Real c2( ( 4.0 + SQRT6 ) / 10.0 );
      const Real c3q( aStepInterval / thePreviousStepInterval );
      const Real c1q( c3q * c1 );
      const Real c2q( c3q * c2 );

      for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  const Real cont3( theTaylorSeries[ 2 ][ c ] );
	  const Real cont2( theTaylorSeries[ 1 ][ c ] + 3.0 * cont3 );
	  const Real cont1( theTaylorSeries[ 0 ][ c ] + 2.0 * cont2 - 3.0 * cont3 );

	  const Real z1( thePreviousStepInterval * c1q * ( cont1 + c1q * ( cont2 + c1q * cont3 ) ) );
	  const Real z2( thePreviousStepInterval * c2q * ( cont1 + c2q * ( cont2 + c2q * cont3 ) ) );
	  const Real z3( thePreviousStepInterval * c3q * ( cont1 + c3q * ( cont2 + c3q * cont3 ) ) );

	  theW[ c ] = 4.3255798900631553510 * z1
	    + 0.33919925181580986954 * z2 + 0.54177053993587487119 * z3;
	  theW[ c+aSize ] = -4.1787185915519047273 * z1
	    - 0.32768282076106238708 * z2 + 0.47662355450055045196 * z3;
	  theW[ c+aSize*2 ] =  -0.50287263494578687595 * z1
	    + 2.5719269498556054292 * z2 - 0.59603920482822492497 * z3;
	}
    }
  else
    {
      for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  theW[ c ] = 0.0;
	  theW[ c+aSize ] = 0.0;
	  theW[ c+aSize*2 ] = 0.0;
	}
    }

  eta = pow( std::max( eta, Uround ), 0.8 );

  while ( anIterator < getMaxIterationNumber() )
    {
      calculateRhs();
      
      const Real aPreviousNorm( std::max( aNorm, Uround ) );
      aNorm = solve();

      if ( anIterator > 0 && ( anIterator != getMaxIterationNumber()-1 ) )
	{
	  const Real aThetaQ = aNorm / aPreviousNorm;
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
  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      const Real w1( theW[ c ] );
      const Real w2( theW[ c + aSize ] );
      const Real w3( theW[ c + aSize*2 ] );
	    
      theW[ c ] = w1 * 0.091232394870892942792
	- w2 * 0.14125529502095420843
	- w3 * 0.030029194105147424492;
      theW[ c + aSize ] = w1 * 0.24171793270710701896
	+ w2 * 0.20412935229379993199
	+ w3 * 0.38294211275726193779;
      theW[ c + aSize*2 ] = w1 * 0.96604818261509293619 + w2;
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

Real DAEStepper::estimateLocalError()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  const ProcessVector::size_type 
    aDiscreteProcessOffset( getDiscreteProcessOffset() );
  const Real aStepInterval( getStepInterval() );
  
  Real anError;
  
  const Real hee1( ( -13.0 - 7.0 * SQRT6 ) / ( 3.0 * aStepInterval ) );
  const Real hee2( ( -13.0 + 7.0 * SQRT6 ) / ( 3.0 * aStepInterval ) );
  const Real hee3( -1.0 / ( 3.0 * aStepInterval ) );
  
  // theW will already be transformed to Z-form
  for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	c < theProcessVector.size(); c++ )
    {
      const ProcessVector::size_type anIndex
	( theContinuousVariableVector.size() + c - aDiscreteProcessOffset );
      
      gsl_vector_set( theVelocityVector1, anIndex,
		      theDiscreteActivityBuffer[ c - aDiscreteProcessOffset ] );	
    }
  
  for ( IntVector::size_type c( 0 );
	c < theContinuousVariableVector.size(); c++ )
    {
      const int anIndex( theContinuousVariableVector[ c ] );
      
      gsl_vector_set( theVelocityVector1, c,
		      theTaylorSeries[ 3 ][ anIndex ]
		      + theW[ anIndex ] * hee1
		      + theW[ anIndex + aSize ] * hee2
		      + theW[ anIndex + 2*aSize ] * hee3 );
    }

  gsl_linalg_LU_solve( theJacobianMatrix1, thePermutation1,
		       theVelocityVector1, theSolutionVector1 );

  anError = 0.0;
  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      const Real aTolerance( rtoler * fabs( theValueBuffer[ c ] ) + atoler );
      Real aDifference( gsl_vector_get( theSolutionVector1, c ) );

      // for the case ( anError >= 1.0 )
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] + aDifference );

      aDifference /= aTolerance;
      anError += aDifference * aDifference;
    }
  
  anError = std::max( sqrt( anError / aSize ), 1e-10 );

  if ( anError < 1.0 ) return anError;

  if ( theFirstStepFlag or theRejectedStepFlag )
    {
      fireProcesses();
      setVariableVelocity( theTaylorSeries[ 4 ] );

      for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	    c < theProcessVector.size(); c++ )
	{
	  const ProcessVector::size_type anIndex
	    ( theContinuousVariableVector.size() + c
		- aDiscreteProcessOffset );
	  
	  gsl_vector_set( theVelocityVector1, anIndex,
			  theProcessVector[ c ]->getActivity() );	
	}
      
      for ( IntVector::size_type c( 0 );
	    c < theContinuousVariableVector.size(); c++ )
	{
	  const int anIndex( theContinuousVariableVector[ c ] );
	  
	  gsl_vector_set( theVelocityVector1, c,
			  theTaylorSeries[ 4 ][ anIndex ]
			  + theW[ anIndex ] * hee1
			  + theW[ anIndex + aSize ] * hee2
			  + theW[ anIndex + 2*aSize ] * hee3 );
	}

      gsl_linalg_LU_solve( theJacobianMatrix1, thePermutation1,
			   theVelocityVector1, theSolutionVector1 );

      anError = 0.0;
      for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  const Real aTolerance( rtoler * fabs( theValueBuffer[ c ] )
				 + atoler );
	  
	  Real aDifference( gsl_vector_get( theSolutionVector1, c ) );
	  aDifference /= aTolerance;
	  
	  anError += aDifference * aDifference;
	}

      anError = std::max( sqrt( anError / aSize ), 1e-10 );
    }

  return anError;
}

void DAEStepper::step()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  
  theStateFlag = false;
  
  thePreviousStepInterval = getStepInterval();
  setStepInterval( getNextStepInterval() );
  clearVariables();
  //    interIntegrate();

  theRejectedStepFlag = false;
  
  fireProcesses();

  const ProcessVector::size_type
    aDiscreteProcessOffset( getDiscreteProcessOffset() );
  for ( ProcessVector::size_type c( aDiscreteProcessOffset );
	c < theProcessVector.size(); c++ )
    {
      theDiscreteActivityBuffer[ c - aDiscreteProcessOffset ] 
	= theProcessVector[ c ]->getActivity();
    }
  
  setVariableVelocity( theTaylorSeries[ 3 ] );
    
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

  UnsignedInteger count( 0 );
  while ( !calculate() )
    {
      if ( count++ > 3 ) break;

      theRejectedStepFlag = true;
      
      if ( !theJacobianCalculateFlag )
	{
	  calculateJacobian();
	  theJacobianCalculateFlag = true;
	}

      setJacobianMatrix();
    }

  const Real aStepInterval( getStepInterval() );
  setTolerableStepInterval( aStepInterval );

  // theW will already be transformed to Z-form

  for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
      theTaylorSeries[ 3 ][ c ] = theW[ c + aSize*2 ];
      theTaylorSeries[ 3 ][ c ] /= aStepInterval;
      
      theVariableVector[ c ]->loadValue( theValueBuffer[ c ] );
    }

  for ( VariableVector::size_type c( 0 ); c < aSize; c++ )
    {
      const Real z1( theW[ c ] );
      const Real z2( theW[ c + aSize ] );
      const Real z3( theW[ c + aSize*2 ] );

      theTaylorSeries[ 0 ][ c ] = ( 13.0 + 7.0 * SQRT6 ) / 3.0 * z1
	+ ( 13.0 - 7.0 * SQRT6 ) / 3.0 * z2
	+ 1.0 / 3.0 * z3;
      theTaylorSeries[ 1 ][ c ] = - ( 23.0 + 22.0 * SQRT6 ) / 3.0 * z1
	+ ( -23.0 + 22.0 * SQRT6 ) / 3.0 * z2
	- 8.0 / 3.0 * z3;
      theTaylorSeries[ 2 ][ c ] = ( 10.0 + 15.0 * SQRT6 ) / 3.0 * z1 
	+ ( 10.0 - 15.0 * SQRT6 ) / 3.0 * z2
	+ 10.0 / 3.0 * z3;
      
      theTaylorSeries[ 0 ][ c ] /= aStepInterval;
      theTaylorSeries[ 1 ][ c ] /= aStepInterval;
      theTaylorSeries[ 2 ][ c ] /= aStepInterval;
    }

  theStateFlag = true;

  if ( fabs( getTolerableStepInterval() - getStepInterval() )
       > std::numeric_limits<Real>::epsilon() )
    isInterrupted = true;
  else
    isInterrupted = false;
}

void DAEStepper::interrupt( StepperPtr aCaller )
{
  isInterrupted = true;
  DifferentialStepper::interrupt( aCaller );
}
