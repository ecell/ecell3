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

#include "FixedDAE1Stepper.hpp"

LIBECS_DM_INIT( FixedDAE1Stepper, Stepper );

FixedDAE1Stepper::FixedDAE1Stepper()
  :
  theJacobianMatrix( NULLPTR ),
  theVelocityVector( NULLPTR ),
  theSolutionVector( NULLPTR ),
  thePermutation( NULLPTR ),
  theSystemSize( 0 ),
  theTolerance( 1e-10 ),
  thePerturbationRate( 1e-9 ),
  theEachVelocityBuffer( NULLPTR ),
  theDependentProcessVector( NULLPTR ),
  theContinuousVariableVector( NULLPTR )
{
  ; // do nothing
}
	    
FixedDAE1Stepper::~FixedDAE1Stepper()
{
  // free an allocated matrix
  gsl_matrix_free( theJacobianMatrix );
  gsl_vector_free( theVelocityVector );
  gsl_vector_free( theSolutionVector );
  gsl_permutation_free( thePermutation );
}

void FixedDAE1Stepper::initialize()
{
  DifferentialStepper::initialize();
    
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
  if ( theSystemSize != aSize )
    {
      checkDependency();

      theSystemSize = theContinuousVariableVector.size()
	+ theProcessVector.size() - getDiscreteProcessOffset();
      if ( aSize != theSystemSize )
	{
	  THROW_EXCEPTION( InitializationFailed,
			   "definitions are requred, are given." );
	}

      // allocate a matrix and set all elements to zero
      if ( theJacobianMatrix )
	gsl_matrix_free( theJacobianMatrix );
      theJacobianMatrix = gsl_matrix_calloc( theSystemSize, theSystemSize );

      if ( theVelocityVector )
	gsl_vector_free( theVelocityVector );
      theVelocityVector = gsl_vector_calloc( theSystemSize );

      if ( theSolutionVector )
	gsl_vector_free( theSolutionVector );
      theSolutionVector = gsl_vector_alloc( theSystemSize );

      if ( thePermutation )
	gsl_permutation_free( thePermutation );
      thePermutation = gsl_permutation_alloc( theSystemSize );
    }

  //    printJacobianMatrix();
}

void FixedDAE1Stepper::checkDependency()
{
  const VariableVector::size_type 
    aReadOnlyVariableOffset( getReadOnlyVariableOffset() );
  const ProcessVector::size_type aProcessNum( theProcessVector.size() );

  theDependentProcessVector.clear();
  theDependentProcessVector.resize( aReadOnlyVariableOffset );
  theDependentVariableVector.clear();
  theDependentVariableVector.resize( aReadOnlyVariableOffset );

  theContinuousVariableVector.clear();

  IntVector anIndexVector;
  IntVectorConstIterator aWriteVariableIterator;

  ProcessVectorConstIterator anIterator( theProcessVector.begin() );
  for( ProcessVector::size_type c( 0 ); c < aProcessNum; c++ )
    {
      VariableReferenceVectorCref aVariableReferenceVector
	( (*anIterator)->getVariableReferenceVector() );

      const VariableReferenceVector::size_type aZeroVariableReferenceOffset
	( (*anIterator)->getZeroVariableReferenceOffset() );
      const VariableReferenceVector::size_type
	aPositiveVariableReferenceOffset
	( (*anIterator)->getPositiveVariableReferenceOffset() );

      anIndexVector.clear();
      for ( VariableReferenceVector::size_type 
	      i( aZeroVariableReferenceOffset );
	    i != aPositiveVariableReferenceOffset; i++ )
	{
	  VariablePtr const aVariable
	    ( aVariableReferenceVector[ i ].getVariable() );

	  const VariableVector::size_type 
	    anIndex( getVariableIndex( aVariable ) );

	  // std::binary_search?
	  if ( std::find( theDependentProcessVector[ anIndex ].begin(),
			  theDependentProcessVector[ anIndex ].end(), c )
	       == theDependentProcessVector[ anIndex ].end() )
	    {
	      theDependentProcessVector[ anIndex ].push_back( c );
	      anIndexVector.push_back( anIndex );
	    }
	}

      const IntVector::size_type aNonZeroOffset( anIndexVector.size() );
      const bool aContinuity( (*anIterator)->isContinuous() );

      for( VariableReferenceVector::size_type i( 0 ); 
	   i < aVariableReferenceVector.size(); i++ )
	{
	  if ( i == aZeroVariableReferenceOffset )
	    {
	      if ( aPositiveVariableReferenceOffset
		   == aVariableReferenceVector.size() )
		break;
	      else
		i = aPositiveVariableReferenceOffset;
	    }

	  VariablePtr const aVariable
	    ( aVariableReferenceVector[ i ].getVariable() );

	  const VariableVector::size_type 
	    anIndex( getVariableIndex( aVariable ) );

	  if ( aContinuity )
	    {
	      if ( std::find( theContinuousVariableVector.begin(),
			      theContinuousVariableVector.end(), anIndex )
		   == theContinuousVariableVector.end() )
		{
		  theContinuousVariableVector.push_back( anIndex );
		}
	    }

	  // std::binary_search?
	  if ( std::find( theDependentProcessVector[ anIndex ].begin(),
			  theDependentProcessVector[ anIndex ].end(), c )
	       == theDependentProcessVector[ anIndex ].end() )
	    {
	      theDependentProcessVector[ anIndex ].push_back( c );
	      anIndexVector.push_back( anIndex );
	    }
	}

      FOR_ALL( IntVector, anIndexVector )
	{
	  for( IntVector::size_type j( aNonZeroOffset );
	       j != anIndexVector.size(); ++j )
	    {
	      const int anIndex( anIndexVector[ j ] );
	      if ( std::find( theDependentVariableVector[ (*i) ].begin(),
			      theDependentVariableVector[ (*i) ].end(),
			      anIndex )
		   == theDependentVariableVector[ (*i) ].end() )
		{
		  theDependentVariableVector[ (*i) ].push_back( anIndex );
		}
	    }
	    
	  // stable_sort?
	  std::sort( theDependentVariableVector[ (*i) ].begin(),
		     theDependentVariableVector[ (*i) ].end() );
	}
	
      anIterator++;
    }

  std::sort( theContinuousVariableVector.begin(),
	     theContinuousVariableVector.end() );

  theEachVelocityBuffer.clear();
  theEachVelocityBuffer.resize( theContinuousVariableVector.size()
				* getDiscreteProcessOffset() );

  theDiscreteActivityBuffer.clear();
  theDiscreteActivityBuffer.resize( aProcessNum
				    - getDiscreteProcessOffset() );
}

void FixedDAE1Stepper::resetVelocity()
{
  for( RealVector::size_type c( 0 ); c < theTaylorSeries[ 0 ].size(); c++ )
    {
      theTaylorSeries[ 0 ][ c ] = 0.0;
      theVariableVector[ c ]->clearVelocity();
    }
}
  
void FixedDAE1Stepper::calculateVelocityVector()
{
  const Real aCurrentTime( getCurrentTime() );
  const Real aStepInterval( getStepInterval() );

  const ProcessVector::size_type 
    aDiscreteProcessOffset( getDiscreteProcessOffset() );

  gsl_vector_set_zero( theVelocityVector );

  setCurrentTime( aCurrentTime + aStepInterval );

  // almost equal to call fire()
  for( ProcessVector::size_type c( 0 ); c < aDiscreteProcessOffset; c++ )
    {
      theProcessVector[ c ]->fire();

      for( IntVector::size_type i( 0 ); 
	   i < theContinuousVariableVector.size(); i++ )
	{
	  const int anIndex( theContinuousVariableVector[ i ] );
	  VariablePtr const aVariable( theVariableVector[ anIndex ] );

	  theEachVelocityBuffer[ i * aDiscreteProcessOffset + c ]
	    = aVariable->getVelocity();
	    
	  /**
	     std::cout << theProcessVector[ c ]->getID() 
	     << ":" << aVariable->getID()
	     << ":" << aVariable->getVelocity() << std::endl;
	  */

	  theTaylorSeries[ 0 ][ anIndex ] += aVariable->getVelocity();
	    
	  aVariable->clearVelocity();
	}
    }

  for( IntVector::size_type i( 0 ); 
       i < theContinuousVariableVector.size(); ++i )
    {
      const int anIndex( theContinuousVariableVector[ i ] );
      const Real aVelocity( theTaylorSeries[ 0 ][ anIndex ] * aStepInterval
			    + theValueBuffer[ anIndex ]
			    - theVariableVector[ anIndex ]->getValue() );

      gsl_vector_set( theVelocityVector, i, aVelocity );

      /**
	 std::cout << i << ":" << theVariableVector[ anIndex ]->getID()
	 << ":" << aVelocity << std::endl;
      */

      theTaylorSeries[ 0 ][ anIndex ] = 0.0;
      theVariableVector[ anIndex ]->clearVelocity();
    }

  for( ProcessVector::size_type c( aDiscreteProcessOffset );
       c < theProcessVector.size(); c++ )
    {
      theProcessVector[ c ]->fire();

      const Real anActivity( theProcessVector[ c ]->getActivity() );

      /**
	 std::cout << theContinuousVariableVector.size() + c
	 - aDiscreteProcessOffset
	 << ":" << theProcessVector[ c ]->getID()
	 << ":" << -1.0 * anActivity << std::endl;
      */

      theDiscreteActivityBuffer[ c - aDiscreteProcessOffset ] = anActivity;
      gsl_vector_set( theVelocityVector,
		      theContinuousVariableVector.size() + c
		      - aDiscreteProcessOffset,
		      -1.0 * anActivity );
    }

  setCurrentTime( aCurrentTime );
}
  
void FixedDAE1Stepper::calculateJacobian()
{
  const Real aCurrentTime( getCurrentTime() );
  const Real aStepInterval( getStepInterval() );
  const Real aPerturbation( thePerturbationRate * aStepInterval );
  const VariableVector::size_type 
    aReadOnlyVariableOffset( getReadOnlyVariableOffset() );
  const ProcessVector::size_type
    aDiscreteProcessOffset( getDiscreteProcessOffset() );

  Real aJacobian( 0.0 );

  gsl_matrix_set_zero( theJacobianMatrix );

  setCurrentTime( aCurrentTime + aStepInterval );

  for( VariableVector::size_type c( 0 ); c < aReadOnlyVariableOffset; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
      const Real aValue( aVariable->getValue() );

      aVariable->loadValue( aValue + aPerturbation );

      FOR_ALL( IntVector, theDependentProcessVector[ c ] )
	{
	  theProcessVector[ (*i) ]->fire();

	  if ( aDiscreteProcessOffset > (*i) )
	    {
	      for( IntVector::size_type j( 0 );
		   j < theContinuousVariableVector.size(); ++j )
		{
		  theTaylorSeries[ 0 ][ theContinuousVariableVector[ j ] ]
		    += theEachVelocityBuffer[ j * aDiscreteProcessOffset
					      + (*i) ];
		}
	    }
	  else
	    {
	      aJacobian = theProcessVector[ (*i) ]->getActivity()
		- theDiscreteActivityBuffer[ (*i) - aDiscreteProcessOffset ];
	      aJacobian /= aPerturbation;

	      gsl_matrix_set( theJacobianMatrix,
			      theContinuousVariableVector.size() + (*i)
			      - aDiscreteProcessOffset, c, aJacobian );
	    }
	}

      for( IntVector::size_type i( 0 ); 
	   i < theContinuousVariableVector.size(); ++i )
	{
	  // this calculation already includes negative factor
	  const int anIndex( theContinuousVariableVector[ i ] );

	  aJacobian = theTaylorSeries[ 0 ][ anIndex ] 
	    - theVariableVector[ anIndex ]->getVelocity();
	  aJacobian /= aPerturbation;

	  gsl_matrix_set( theJacobianMatrix, i, c,
			  aJacobian * getStepInterval() );

	  theTaylorSeries[ 0 ][ anIndex ] = 0.0;
	  theVariableVector[ anIndex ]->clearVelocity();
	}

      aVariable->loadValue( aValue );
    }

  for ( IntVector::size_type c( 0 ); 
	c < theContinuousVariableVector.size(); c++ )
    {
      const int anIndex( theContinuousVariableVector[ c ] );
      aJacobian = gsl_matrix_get( theJacobianMatrix, c, anIndex );
      gsl_matrix_set( theJacobianMatrix, c, anIndex, 1.0 + aJacobian );
    }

  setCurrentTime( aCurrentTime );
}

const Real FixedDAE1Stepper::solve()
{
  const VariableVector::size_type 
    aReadOnlyVariableOffset( getReadOnlyVariableOffset() );

  int aSignNum;
  gsl_linalg_LU_decomp( theJacobianMatrix, thePermutation, &aSignNum );
  gsl_linalg_LU_solve( theJacobianMatrix, thePermutation,
		       theVelocityVector, theSolutionVector );

  Real anError( 0.0 );
  Real aTotalVelocity( 0.0 );
  for( VariableVector::size_type c( 0 ); c < aReadOnlyVariableOffset; ++c )
    {
      VariablePtr const aVariable( theVariableVector[ c ] );
      const Real aDifference( gsl_vector_get( theSolutionVector, c ) );

      aVariable->addValue( aDifference );
      anError += aDifference;

      const Real aVelocity( aVariable->getValue() - theValueBuffer[ c ] );
      aTotalVelocity += aVelocity;

      theTaylorSeries[ 0 ][ c ] = aVelocity / getStepInterval();
      aVariable->setVelocity( theTaylorSeries[ 0 ][ c ] );
    }

  return fabs( anError / aTotalVelocity );
}

void FixedDAE1Stepper::step()
{
  theStateFlag = false;

  clearVariables();

  // Newton iteration
  int anIterator( 0 );
  while ( anIterator < 5 )
    {
      resetVelocity();
      calculateVelocityVector();
      calculateJacobian();
      //	printJacobianMatrix();

      const Real anError( solve() );

      if ( anError < getTolerance() )
	{
	  break;
	}

      anIterator++;
    }

  resetAll();

  //    interIntegrate();
  theStateFlag = true;
}

