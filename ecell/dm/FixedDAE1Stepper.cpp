//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#include <libecs/Variable.hpp>
#include <libecs/Process.hpp>
#include <libecs/DifferentialStepper.hpp>

#include <gsl/gsl_linalg.h>

USE_LIBECS;

DECLARE_VECTOR( int, IntVector );

LIBECS_DM_CLASS( FixedDAE1Stepper, DifferentialStepper )
{

public:

    LIBECS_DM_OBJECT( FixedDAE1Stepper, Stepper )
    {
        INHERIT_PROPERTIES( DifferentialStepper );

        PROPERTYSLOT_SET_GET( Real, PerturbationRate );
        PROPERTYSLOT_SET_GET( Real, Tolerance );
    }

    FixedDAE1Stepper()
        : theJacobianMatrix( NULLPTR ),
          theVelocityVector( NULLPTR ),
          theSolutionVector( NULLPTR ),
          thePermutation( NULLPTR ),
          theSystemSize( 0 ),
          theTolerance( 1e-10 ),
          thePerturbationRate( 1e-9 ),
          theDependentProcessVector( NULLPTR ),
          theContinuousVariableVector( NULLPTR ),
          theActivityBuffer( NULLPTR )
    {
        ; // do nothing
    }
                                                
    virtual ~FixedDAE1Stepper()
    {
        // free an allocated matrix
        if ( theJacobianMatrix )
        {
            gsl_matrix_free( theJacobianMatrix );
        }
        if ( theVelocityVector )
        {
            gsl_vector_free( theVelocityVector );
        }
        if ( theSolutionVector )
        {
            gsl_vector_free( theSolutionVector );
        }
        if ( thePermutation )
        {
            gsl_permutation_free( thePermutation );
        }
    }

    SET_METHOD( Real, PerturbationRate )
    {
        thePerturbationRate = value;
    }

    GET_METHOD( Real, PerturbationRate )
    {
        return thePerturbationRate;
    }

    SET_METHOD( Real, Tolerance )
    {
        theTolerance = value;
    }

    GET_METHOD( Real, Tolerance )
    {
        return theTolerance;
    }

    virtual void initialize()
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
                THROW_EXCEPTION_INSIDE( InitializationFailed,
                                 asString()
                                 + ": the number of algebraic variables "
                                 + "must be the same as the equations" );
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
    }

    virtual void step()
    {
        theStateFlag = false;

        clearVariables();

        // Newton iteration
        int anIterator( 0 );
        while ( anIterator < 5 )
        {
            calculateVelocityVector();
            calculateJacobian();

            const Real anError( solve() );

            if ( anError < getTolerance() )
            {
                break;
            }

            anIterator++;
        }

        resetAll();

        theStateFlag = true;
    }

    void calculateVelocityVector()
    {
        const Real aCurrentTime( getCurrentTime() );
        const Real aStepInterval( getStepInterval() );

        const ProcessVector::size_type aDiscreteProcessOffset(
                getDiscreteProcessOffset() );

        gsl_vector_set_zero( theVelocityVector );

        setCurrentTime( aCurrentTime + aStepInterval );

        // almost equal to call fire()

        fireProcesses();
        setVariableVelocity( theTaylorSeries[ 0 ] );

        for( ProcessVector::size_type c( 0 ); c < theProcessVector.size(); ++c )
        {
            theActivityBuffer[ c ] = theProcessVector[ c ]->getActivity();
        }

        for( IntVector::size_type i( 0 ); 
                 i < theContinuousVariableVector.size(); ++i )
        {
            const int anIndex( theContinuousVariableVector[ i ] );
            const Real aVelocity( theTaylorSeries[ 0 ][ anIndex ] *
                                  aStepInterval + theValueBuffer[ anIndex ]
                                  - theVariableVector[ anIndex ]->getValue() );

            gsl_vector_set( theVelocityVector, i, aVelocity );

            theTaylorSeries[ 0 ][ anIndex ] = 0.0;
        }

        for( ProcessVector::size_type c( aDiscreteProcessOffset );
                 c < theProcessVector.size(); ++c )
        {
            const Real anActivity( theProcessVector[ c ]->getActivity() );

            gsl_vector_set( theVelocityVector,
                            theContinuousVariableVector.size() + c
                                - aDiscreteProcessOffset,
                                -theActivityBuffer[ c ] );
        }

        setCurrentTime( aCurrentTime );
    }

    void calculateJacobian()
    {
        const Real aCurrentTime( getCurrentTime() );
        const Real aStepInterval( getStepInterval() );
        const Real aPerturbation( thePerturbationRate * aStepInterval );
        const VariableVector::size_type aReadOnlyVariableOffset(
                getReadOnlyVariableOffset() );
        const ProcessVector::size_type aDiscreteProcessOffset(
                getDiscreteProcessOffset() );

        gsl_matrix_set_zero( theJacobianMatrix );

        setCurrentTime( aCurrentTime + aStepInterval );

        for( VariableVector::size_type c( 0 ); c < aReadOnlyVariableOffset;
             ++c )
        {
            VariablePtr const aVariable( theVariableVector[ c ] );
            const Real aValue( aVariable->getValue() );

            aVariable->loadValue( aValue + aPerturbation );

            FOR_ALL( IntVector, theDependentProcessVector[ c ] )
            {
                const Integer anIndex( *i );

                theProcessVector[ anIndex ]->fire();
                const Real aDifference( theProcessVector[ anIndex ]->getActivity() - theActivityBuffer[ anIndex ] );

                if ( aDiscreteProcessOffset > anIndex )
                {
                    VariableReferenceListVector::value_type const& varRef(
                        theVariableReferenceListVector[ anIndex ] ); 
                    for ( VariableReferenceList::const_iterator anIterator(
                            varRef.begin() ), end( varRef.end() );
                          anIterator < end; ++anIterator )
                    {
                        ExprComponent const& aComponent( *anIterator );
                        theTaylorSeries[ 0 ][ static_cast< RealMatrix::index >(
                                              aComponent.first ) ] +=
                                aComponent.second * aDifference;
                    }
                }
                else
                {
                    gsl_matrix_set( theJacobianMatrix,
                                    theContinuousVariableVector.size()
                                    + anIndex - aDiscreteProcessOffset, c,
                                    aDifference / aPerturbation );
                }
            }

            for ( IntVector::size_type i( 0 ); 
                  i < theContinuousVariableVector.size(); ++i )
            {
                // this calculation already includes negative factor
                const int anIndex( theContinuousVariableVector[ i ] );
                const Real aDerivative(
                        theTaylorSeries[ 0 ][ anIndex ] / aPerturbation );

                gsl_matrix_set( theJacobianMatrix, i, c,
                                -1.0 * aDerivative * getStepInterval() );

                theTaylorSeries[ 0 ][ anIndex ] = 0.0;
            }

            aVariable->loadValue( aValue );
        }

        for ( IntVector::size_type c( 0 ); 
              c < theContinuousVariableVector.size(); c++ )
        {
            const int anIndex( theContinuousVariableVector[ c ] );
            const Real aDerivative( gsl_matrix_get( theJacobianMatrix, c,
                                                    anIndex ) );
            gsl_matrix_set( theJacobianMatrix, c, anIndex, 1.0 + aDerivative );
        }

        setCurrentTime( aCurrentTime );
    }

    void checkDependency()
    {
        const VariableVector::size_type aReadOnlyVariableOffset(
                getReadOnlyVariableOffset() );

        theDependentProcessVector.clear();
        theDependentProcessVector.resize( aReadOnlyVariableOffset );
        theDependentVariableVector.clear();
        theDependentVariableVector.resize( aReadOnlyVariableOffset );

        theContinuousVariableVector.clear();

        IntVector anIndexVector;
        IntVectorConstIterator aWriteVariableIterator;

        ProcessVectorConstIterator anIterator( theProcessVector.begin() );
        for( ProcessVector::size_type c( 0 ); c < theProcessVector.size(); ++c )
        {
            VariableReferenceVectorCref aVariableReferenceVector(
                    (*anIterator)->getVariableReferenceVector() );

            const VariableReferenceVector::size_type aZeroVariableReferenceOffset(
                    (*anIterator)->getZeroVariableReferenceOffset() );
            const VariableReferenceVector::size_type
                aPositiveVariableReferenceOffset(
                    (*anIterator)->getPositiveVariableReferenceOffset() );

            anIndexVector.clear();
            for ( VariableReferenceVector::size_type i(
                    aZeroVariableReferenceOffset );
                  i != aPositiveVariableReferenceOffset; ++i )
            {
                VariablePtr const aVariable(
                    aVariableReferenceVector[ i ].getVariable() );

                const VariableVector::size_type anIndex(
                    getVariableIndex( aVariable ) );

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
                     i < aVariableReferenceVector.size(); ++i )
            {
                if ( i == aZeroVariableReferenceOffset )
                {
                    if ( aPositiveVariableReferenceOffset ==
                            aVariableReferenceVector.size() )
                        break;

                    i = aPositiveVariableReferenceOffset;
                }

                VariablePtr const aVariable(
                        aVariableReferenceVector[ i ].getVariable() );

                const VariableVector::size_type anIndex(
                        getVariableIndex( aVariable ) );

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
                                    anIndex ) == 
                         theDependentVariableVector[ (*i) ].end() )
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

        theActivityBuffer.clear();
        theActivityBuffer.resize( theProcessVector.size() );
    }

    const Real solve()
    {
        const VariableVector::size_type aReadOnlyVariableOffset(
                getReadOnlyVariableOffset() );

        int aSignNum;
        gsl_linalg_LU_decomp( theJacobianMatrix, thePermutation, &aSignNum );
        gsl_linalg_LU_solve( theJacobianMatrix, thePermutation,
                             theVelocityVector, theSolutionVector );

        Real anError( 0.0 );
        Real aTotalVelocity( 0.0 );
        for( VariableVector::size_type c( 0 ); c < aReadOnlyVariableOffset;
             ++c )
        {
            VariablePtr const aVariable( theVariableVector[ c ] );

            const Real aDifference( gsl_vector_get( theSolutionVector, c ) );
            aVariable->addValue( aDifference );
            anError += aDifference;
            
            const Real aVelocity( aVariable->getValue() - theValueBuffer[ c ] );
            aTotalVelocity += aVelocity;
            
            theTaylorSeries[ 0 ][ c ] = aVelocity / getStepInterval();
        }

        return fabs( anError / aTotalVelocity );
    }

protected:

    UnsignedInteger         theSystemSize;
    Real                    thePerturbationRate;
    Real                    theTolerance;

    std::vector<IntVector>  theDependentProcessVector;
    std::vector<IntVector>  theDependentVariableVector;

    gsl_matrix*             theJacobianMatrix;
    gsl_vector*             theVelocityVector;
    gsl_vector*             theSolutionVector;
    gsl_permutation*        thePermutation;

    IntVector             theContinuousVariableVector;
    RealVector            theActivityBuffer;
};

LIBECS_DM_INIT( FixedDAE1Stepper, Stepper );
