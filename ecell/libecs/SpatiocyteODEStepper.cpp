//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2006-2009 Keio University
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
// written by Satya Arjunan <satya.arjunan@gmail.com>
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//


#ifndef __SpatiocyteODEStepper_hpp
#define __SpatiocyteODEStepper_hpp

#define GSL_RANGE_CHECK_OFF

#include <libecs/Stepper.hpp>
#include <SpatiocyteProcessInterface.hpp>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <cmath>
#include <libecs/Variable.hpp>
#include <libecs/Process.hpp>
#include <libecs/AdaptiveDifferentialStepper.hpp>

USE_LIBECS;

LIBECS_DM_CLASS( SpatiocyteODEStepper, AdaptiveDifferentialStepper )
{
public:
    LIBECS_DM_OBJECT( SpatiocyteODEStepper, Stepper )
    {
        INHERIT_PROPERTIES( AdaptiveDifferentialStepper );

        PROPERTYSLOT_SET_GET( Integer, MaxIterationNumber );
        PROPERTYSLOT_SET_GET( Real, Uround );
        
        PROPERTYSLOT( Real, Tolerance,
                      &SpatiocyteODEStepper::initializeTolerance,
                      &AdaptiveDifferentialStepper::getTolerance );

        PROPERTYSLOT( Real, AbsoluteToleranceFactor,
                      &SpatiocyteODEStepper::initializeAbsoluteToleranceFactor,
                      &AdaptiveDifferentialStepper::getAbsoluteToleranceFactor );
        

        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Stiffness );
        PROPERTYSLOT_SET_GET( Real, JacobianRecalculateTheta );

        PROPERTYSLOT( Integer, isStiff,
                      &SpatiocyteODEStepper::setIntegrationType,
                      &SpatiocyteODEStepper::getIntegrationType );

        PROPERTYSLOT_SET_GET( Integer, CheckIntervalCount );
        PROPERTYSLOT_SET_GET( Integer, SwitchingCount );
    }

    SpatiocyteODEStepper( void );
    virtual ~SpatiocyteODEStepper( void );
    virtual void step()
      {
        AdaptiveDifferentialStepper::step();
        for(std::vector<SpatiocyteProcessInterface*>::const_iterator
            i(theProcesses.begin()); i != theProcesses.end(); ++i)
          {
            (*i)->finalizeFire();
          }
      }
    /*
    bool isDependentOn( Stepper const* aStepper )
      {
        return false;
      }
    virtual void interrupt(Time aTime)
      {
        AdaptiveDifferentialStepper::interrupt(aTime);
      }
      */
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

    Integer getIntegrationType() const { return isStiff; }
    
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
    virtual void updateInternalState( Real aStepInterval );
    virtual bool calculate( Real aStepInterval );
    void initializeStepper();

    void calculateJacobian();
    Real calculateJacobianNorm();
    void setJacobianMatrix( Real const aStepInterval );
    void decompJacobianMatrix();
    void calculateRhs( Real const aStepInterval );
    Real solve();
    Real estimateLocalError( Real const aStepInterval );

    void initializeRadauIIA( VariableVector::size_type );
    std::pair< bool, Real > calculateRadauIIA( Real aStepInterval,
                                               Real aPreviousStepInterval );
    void updateInternalStateRadauIIA( Real aStepInterval );

    void initializeTolerance( libecs::Param<Real>::type value )
    {
        setTolerance( value ); // AdaptiveDifferentialStepper::
        rtoler = 0.1 * pow( getTolerance(), 2.0 / 3.0 );
        atoler = rtoler * getAbsoluteToleranceFactor();
    }

    void initializeAbsoluteToleranceFactor( libecs::Param<Real>::type value )
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

    Real alpha, beta, gamma;

    VariableVector::size_type theSystemSize;

    RealMatrix           theJacobian, theW;
    RealVector           theEigenVector, theTempVector;

    gsl_matrix*          theJacobianMatrix1;
    gsl_permutation*     thePermutation1;
    gsl_vector*          theVelocityVector1;
    gsl_vector*          theSolutionVector1;

    gsl_matrix_complex* theJacobianMatrix2;
    gsl_permutation*    thePermutation2;
    gsl_vector_complex* theVelocityVector2;
    gsl_vector_complex* theSolutionVector2;

    UnsignedInteger theMaxIterationNumber;
    Real theStoppingCriterion;
    Real eta, Uround;

    Real rtoler, atoler;

    Real theAcceptedError, theAcceptedStepInterval;

    Real theJacobianRecalculateTheta;
    Real theSpectralRadius;

    Integer theStiffnessCounter, theRejectedStepCounter;
    Integer CheckIntervalCount, SwitchingCount;

    bool theFirstStepFlag, theJacobianCalculateFlag;
    bool isStiff;
    std::vector<SpatiocyteProcessInterface*> theProcesses;
};


LIBECS_DM_INIT( SpatiocyteODEStepper, Stepper );

#define SQRT6 2.4494897427831779

SpatiocyteODEStepper::SpatiocyteODEStepper()
    : theSystemSize( -1 ),
      theJacobianMatrix1( 0 ),
      thePermutation1( 0 ),
      theVelocityVector1( 0 ),
      theSolutionVector1( 0 ),
      theJacobianMatrix2( 0 ),
      thePermutation2( 0 ),
      theVelocityVector2( 0 ),
      theSolutionVector2( 0 ),
      theMaxIterationNumber( 7 ),
      theStoppingCriterion( 0.0 ),
      eta( 1.0 ),
      Uround( 1e-10 ),
      rtoler( 1e-6 ),
      atoler( 1e-6 ),
      theAcceptedError( 0.0 ),
      theAcceptedStepInterval( 0.0 ),
      theJacobianRecalculateTheta( 0.001 ),
      theSpectralRadius( 0.0 ),
      theStiffnessCounter( 0 ),
      theRejectedStepCounter( 0 ),
      CheckIntervalCount( 100 ),
      SwitchingCount( 20 ),
      theFirstStepFlag( true ),
      theJacobianCalculateFlag( true ),
      isStiff( true )
{
    const Real pow913( pow( 9.0, 1.0 / 3.0 ) );

    alpha = ( 12.0 - pow913 * pow913 + pow913 ) / 60.0;
    beta = ( pow913 * pow913 + pow913 ) * sqrt( 3.0 ) / 60.0;
    gamma = ( 6.0 + pow913 * pow913 - pow913 ) / 30.0;

    const Real aNorm( alpha * alpha + beta * beta );

    alpha /= aNorm;
    beta /= aNorm;
    gamma = 1.0 / gamma;

    rtoler = 0.1 * pow( getTolerance(), 2.0 / 3.0 );
    atoler = rtoler * getAbsoluteToleranceFactor();
}

SpatiocyteODEStepper::~SpatiocyteODEStepper()
{
    if ( theJacobianMatrix1 )
    {
        gsl_matrix_free( theJacobianMatrix1 );
    }
    if ( thePermutation1 )
    {
        gsl_permutation_free( thePermutation1 );
    }
    if ( theVelocityVector1 )
    {
        gsl_vector_free( theVelocityVector1 );
    }
    if ( theSolutionVector1 )
    {
        gsl_vector_free( theSolutionVector1 );
    }
    if ( theJacobianMatrix2 )
    {
        gsl_matrix_complex_free( theJacobianMatrix2 );
    }
    if ( thePermutation2 )
    {
        gsl_permutation_free( thePermutation2 );
    }
    if ( theVelocityVector2 ) 
    {
        gsl_vector_complex_free( theVelocityVector2 );
    }
    if ( theSolutionVector2 )
    {
        gsl_vector_complex_free( theSolutionVector2 );
    }
}

void SpatiocyteODEStepper::initialize()
{
    AdaptiveDifferentialStepper::initialize();
    initializeStepper();
    for(std::vector<Process*>::const_iterator i(theProcessVector.begin());
        i != theProcessVector.end(); ++i)
    {
      SpatiocyteProcessInterface* 
        aProcess(dynamic_cast<SpatiocyteProcessInterface*>(*i));
      if(aProcess)
        {
          theProcesses.push_back(aProcess);
        }
    }
}

void SpatiocyteODEStepper::initializeStepper()
{
    isStiff = true;
    theStiffnessCounter = 0;

    const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

    if ( isStiff )
        initializeRadauIIA( aSize );

    if ( aSize != theSystemSize )
        theW.resize( boost::extents[ 6 ][ aSize ] );

    theSystemSize = aSize;
}

void SpatiocyteODEStepper::initializeRadauIIA(
                                  VariableVector::size_type aNewSystemSize )
{
    eta = 1.0;
    theStoppingCriterion =
          std::max( 10.0 * Uround / rtoler, std::min( 0.03, sqrt( rtoler ) ) );

    theFirstStepFlag = true;
    theJacobianCalculateFlag = true;

    if ( aNewSystemSize != theSystemSize )
    {
        theJacobian.resize(boost::extents[ aNewSystemSize ][ aNewSystemSize ]);
        theEigenVector.resize( aNewSystemSize );
        theTempVector.resize( aNewSystemSize );

        if ( theJacobianMatrix1 )
        {
            gsl_matrix_free( theJacobianMatrix1 );
            theJacobianMatrix1 = 0;
        }

        if ( aNewSystemSize > 0 )
            theJacobianMatrix1 = gsl_matrix_calloc( aNewSystemSize,
                                                    aNewSystemSize );

        if ( thePermutation1 )
        {
            gsl_permutation_free( thePermutation1 );
            thePermutation1 = 0;
        }

        if ( aNewSystemSize > 0 )
            thePermutation1 = gsl_permutation_alloc( aNewSystemSize );

        if ( theVelocityVector1 )
        {
            gsl_vector_free( theVelocityVector1 );
            theVelocityVector1 = 0;
        }

        if ( aNewSystemSize > 0 )
            theVelocityVector1 = gsl_vector_calloc( aNewSystemSize );

        if ( theSolutionVector1 )
        {
            gsl_vector_free( theSolutionVector1 );
            theSolutionVector1 = 0;
        }

        if ( aNewSystemSize > 0 )
            theSolutionVector1 = gsl_vector_calloc( aNewSystemSize );

        if ( theJacobianMatrix2 )
        {
            gsl_matrix_complex_free( theJacobianMatrix2 );
            theJacobianMatrix2 = 0;
        }

        if ( aNewSystemSize > 0 )
            theJacobianMatrix2 = gsl_matrix_complex_calloc( aNewSystemSize,
                                                            aNewSystemSize );

        if ( thePermutation2 )
        {
            gsl_permutation_free( thePermutation2 );
            thePermutation2 = 0;
        }

        if ( aNewSystemSize > 0 )
            thePermutation2 = gsl_permutation_alloc( aNewSystemSize );

        if ( theVelocityVector2 )
        {
            gsl_vector_complex_free( theVelocityVector2 );
            theVelocityVector2 = 0;
        }

        if ( aNewSystemSize > 0 )
            theVelocityVector2 = gsl_vector_complex_calloc( aNewSystemSize );

        if ( theSolutionVector2 )
        {
            gsl_vector_complex_free( theSolutionVector2 );
            theSolutionVector2 = 0;
        }

        if ( aNewSystemSize > 0 )
            theSolutionVector2 = gsl_vector_complex_calloc( aNewSystemSize );
    }    
}

void SpatiocyteODEStepper::calculateJacobian()
{
    Real aPerturbation;

    for ( VariableVector::size_type i( 0 ); i < theSystemSize; ++i )
    {
        Variable* const aVariable1( theVariableVector[ i ] );
        const Real aValue( aVariable1->getValue() );

        aPerturbation = sqrt( Uround * std::max( 1e-5, fabs( aValue ) ) );
        aVariable1->setValue( theValueBuffer[ i ] + aPerturbation );

        fireProcesses();
        setVariableVelocity( theW[ 4 ] );

        for ( VariableVector::size_type j( 0 ); j < theSystemSize; ++j )
        {
            theJacobian[ j ][ i ] =
                    - ( theW[ 4 ][ j ] - theW[ 3 ][ j ] ) / aPerturbation;
        }

        aVariable1->setValue( aValue );
    }
}

void SpatiocyteODEStepper::setJacobianMatrix( Real const aStepInterval )
{
    const Real alphah( alpha / aStepInterval );
    const Real betah( beta / aStepInterval );
    const Real gammah( gamma / aStepInterval );

    gsl_complex comp1, comp2;

    for ( RealVector::size_type i( 0 ); i < theSystemSize; i++ )
    {
        for ( RealVector::size_type j( 0 ); j < theSystemSize; j++ )
        {
            const Real aPartialDerivative( theJacobian[ i ][ j ] );

            gsl_matrix_set( theJacobianMatrix1, i, j, aPartialDerivative );

            GSL_SET_COMPLEX( &comp1, aPartialDerivative, 0 );
            gsl_matrix_complex_set( theJacobianMatrix2, i, j, comp1 );
        }
    }

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
    {
        const Real aPartialDerivative(
                gsl_matrix_get( theJacobianMatrix1, c, c ) );
        gsl_matrix_set( theJacobianMatrix1, c, c,
                        gammah + aPartialDerivative );

        comp1 = gsl_matrix_complex_get( theJacobianMatrix2, c, c );
        GSL_SET_COMPLEX( &comp2, alphah, betah );
        gsl_matrix_complex_set( theJacobianMatrix2, c, c,
                                gsl_complex_add( comp1, comp2 ) );
    }

    decompJacobianMatrix();
}

void SpatiocyteODEStepper::decompJacobianMatrix()
{
    int aSignNum;

    gsl_linalg_LU_decomp( theJacobianMatrix1, thePermutation1, &aSignNum );
    gsl_linalg_complex_LU_decomp( theJacobianMatrix2, thePermutation2,
                                  &aSignNum );
}

Real SpatiocyteODEStepper::calculateJacobianNorm()
{
    std::fill( theEigenVector.begin(), theEigenVector.end(),
               sqrt( 1.0 / theSystemSize ) );

    Real sum, norm;

    for ( Integer count( 0 ); count < 3; count++ )
    {
        norm = 0.0;
        for ( RealVector::size_type i( 0 ); i < theSystemSize; i++ )
        {
            sum = 0.0;
            for ( RealVector::size_type j( 0 ); j < theSystemSize; j++ )
            {
                const Real aPartialDerivative( theJacobian[ i ][ j ] );
                sum += aPartialDerivative * theEigenVector[ j ];
            }
            theTempVector[ i ] = sum;

            norm += theTempVector[ i ] * theTempVector[ i ];
        }

        norm = sqrt( norm );

        for ( RealVector::size_type i( 0 ); i < theSystemSize; i++ )
        {
            theEigenVector[ i ] = theTempVector[ i ] / norm;
        }
    }

    return norm;
}

void SpatiocyteODEStepper::calculateRhs( Real const aStepInterval )
{
    const Real aCurrentTime( getCurrentTime() );

    const Real alphah( alpha / aStepInterval );
    const Real betah( beta / aStepInterval );
    const Real gammah( gamma / aStepInterval );

    gsl_complex comp;

    RealVector tif( theSystemSize * 3 );

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        const Real z( theW[ 0 ][ c ] * 0.091232394870892942792
                      - theW[ 1 ][ c ] * 0.14125529502095420843
                      - theW[ 2 ][ c ] * 0.030029194105147424492 );

        theVariableVector[ c ]->setValue( theValueBuffer[ c ] + z );
    }

    // ========= 1 ===========

    setCurrentTime( aCurrentTime + aStepInterval * ( 4.0 - SQRT6 ) / 10.0 );
    fireProcesses();
    setVariableVelocity( theW[ 4 ] );

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        tif[ c ] = theW[ 4 ][ c ] * 4.3255798900631553510;
        tif[ c + theSystemSize ] = theW[ 4 ][ c ] * -4.1787185915519047273;
        tif[ c + theSystemSize*2 ] = theW[ 4 ][ c ] * -0.50287263494578687595;

        const Real z( theW[ 0 ][ c ] * 0.24171793270710701896
                      + theW[ 1 ][ c ] * 0.20412935229379993199
                      + theW[ 2 ][ c ] * 0.38294211275726193779 );

        theVariableVector[ c ]->setValue( theValueBuffer[ c ] + z );
    }

    // ========= 2 ===========

    setCurrentTime( aCurrentTime + aStepInterval * ( 4.0 + SQRT6 ) / 10.0 );
    fireProcesses();
    setVariableVelocity( theW[ 4 ] );

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        tif[ c ] += theW[ 4 ][ c ] * 0.33919925181580986954;
        tif[ c + theSystemSize ] -= theW[ 4 ][ c ] * 0.32768282076106238708;
        tif[ c + theSystemSize*2 ] += theW[ 4 ][ c ] * 2.5719269498556054292;

        const Real z( theW[ 0 ][ c ] * 0.96604818261509293619 + theW[ 1 ][ c ] );

        theVariableVector[ c ]->setValue( theValueBuffer[ c ] + z );
    }

    // ========= 3 ===========

    setCurrentTime( aCurrentTime + aStepInterval );
    fireProcesses();
    setVariableVelocity( theW[ 4 ] );

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        tif[ c ] += theW[ 4 ][ c ] * 0.54177053993587487119;
        tif[ c + theSystemSize ] += theW[ 4 ][ c ] * 0.47662355450055045196;
        tif[ c + theSystemSize*2 ] -= theW[ 4 ][ c ] * 0.59603920482822492497;

        const Real w1( theW[ 0 ][ c ] );
        const Real w2( theW[ 1 ][ c ] );
        const Real w3( theW[ 2 ][ c ] );

        gsl_vector_set( theVelocityVector1, c, tif[ c ] - w1 * gammah );

        GSL_SET_COMPLEX( &comp, tif[ c + theSystemSize ] - w2 * alphah + w3 * betah, tif[ c + theSystemSize*2 ] - w2 * betah - w3 * alphah );
        gsl_vector_complex_set( theVelocityVector2, c, comp );
    }

    setCurrentTime( aCurrentTime );
}

Real SpatiocyteODEStepper::solve()
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
        theW[ 0 ][ c ] += deltaW;
        aNorm += deltaW * deltaW / aTolerance2;

        comp = gsl_vector_complex_get( theSolutionVector2, c );

        deltaW = GSL_REAL( comp );
        theW[ 1 ][ c ] += deltaW;
        aNorm += deltaW * deltaW / aTolerance2;

        deltaW = GSL_IMAG( comp );
        theW[ 2 ][ c ] += deltaW;
        aNorm += deltaW * deltaW / aTolerance2;
    }

    return sqrt( aNorm / ( 3 * theSystemSize ) );
}

std::pair< bool, Real > SpatiocyteODEStepper::calculateRadauIIA( Real aStepInterval, Real aPreviousStepInterval )
{
    Real aNewStepInterval;
    Real aNorm( 0. );
    Real theta( fabs( theJacobianRecalculateTheta ) );

    Integer anIterator( 0 );

    if ( !isInterrupted )
    {
        const Real c1( ( 4.0 - SQRT6 ) / 10.0 );
        const Real c2( ( 4.0 + SQRT6 ) / 10.0 );
        const Real c3q( aStepInterval / aPreviousStepInterval );
        const Real c1q( c3q * c1 );
        const Real c2q( c3q * c2 );
        for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
        {
            const Real cont3( theTaylorSeries[ 2 ][ c ] );
            const Real cont2( theTaylorSeries[ 1 ][ c ] + 3.0 * cont3 );
            const Real cont1( theTaylorSeries[ 0 ][ c ] + 2.0 * cont2 - 3.0 * cont3 );

            const Real z1( aPreviousStepInterval * c1q * ( cont1 + c1q * ( cont2 + c1q * cont3 ) ) );
            const Real z2( aPreviousStepInterval * c2q * ( cont1 + c2q * ( cont2 + c2q * cont3 ) ) );
            const Real z3( aPreviousStepInterval * c3q * ( cont1 + c3q * ( cont2 + c3q * cont3 ) ) );

            theW[ 0 ][ c ] = 4.3255798900631553510 * z1
                + 0.33919925181580986954 * z2 + 0.54177053993587487119 * z3;
            theW[ 1 ][ c ] = -4.1787185915519047273 * z1
                - 0.32768282076106238708 * z2 + 0.47662355450055045196 * z3;
            theW[ 2 ][ c ] =    -0.50287263494578687595 * z1
                + 2.5719269498556054292 * z2 - 0.59603920482822492497 * z3;
        }
    }
    else
    {
        for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
        {
            theW[ 0 ][ c ] = 0.0;
            theW[ 1 ][ c ] = 0.0;
            theW[ 2 ][ c ] = 0.0;
        }
    }

    eta = pow( std::max( eta, Uround ), 0.8 );

    for (;;)
    {
        if ( anIterator == getMaxIterationNumber() )
        {
            // XXX: this will be addressed somehow.
            // std::cerr << "matrix is repeatedly singular" << std::endl;
            break;
        }

        calculateRhs( aStepInterval );

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
                const Real anIterationError( eta * aNorm * pow( theta, static_cast<int>( getMaxIterationNumber() - 2 - anIterator) ) / theStoppingCriterion );
                if ( anIterationError >= 1.0 )
                {
                    return std::make_pair( false, aStepInterval * 0.8 * pow( std::max( 1e-4, std::min( 20.0, anIterationError ) ) , -1.0 / ( 4 + getMaxIterationNumber() - 2 - anIterator ) ) );
                }
            }
            else
            {
                return std::make_pair( false, aStepInterval * 0.5 );
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
        const Real w1( theW[ 0 ][ c ] );
        const Real w2( theW[ 1 ][ c ] );
        const Real w3( theW[ 2 ][ c ] );

        theW[ 0 ][ c ] = w1 * 0.091232394870892942792
            - w2 * 0.14125529502095420843
            - w3 * 0.030029194105147424492;
        theW[ 1 ][ c ] = w1 * 0.24171793270710701896
            + w2 * 0.20412935229379993199
            + w3 * 0.38294211275726193779;
        theW[ 2 ][ c ] = w1 * 0.96604818261509293619 + w2;
    }

    const Real anError( estimateLocalError( aStepInterval ) );

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

        if ( theRejectedStepCounter != 0 )
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
        }
        else
        {
            setNextStepInterval( aNewStepInterval );
        }
        return std::make_pair( true, aStepInterval );
    }
    else
    {
        // step is rejected

        if ( theFirstStepFlag )
        {
            aNewStepInterval = 0.1 * aStepInterval;
        }

        return std::make_pair( false, aNewStepInterval );
    }
}

Real SpatiocyteODEStepper::estimateLocalError( Real const aStepInterval )
{
    Real anError;

    const Real hee1( ( -13.0 - 7.0 * SQRT6 ) / ( 3.0 * aStepInterval ) );
    const Real hee2( ( -13.0 + 7.0 * SQRT6 ) / ( 3.0 * aStepInterval ) );
    const Real hee3( -1.0 / ( 3.0 * aStepInterval ) );

    // theW will already be transformed to Z-form
    for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
    {
        gsl_vector_set( theVelocityVector1, c,
                        theW[ 3 ][ c ]
                        + theW[ 0 ][ c ] * hee1
                        + theW[ 1 ][ c ] * hee2
                        + theW[ 2 ][ c ] * hee3 );
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
        theVariableVector[ c ]->setValue( theValueBuffer[ c ] + aDifference );

        aDifference /= aTolerance;
        anError += aDifference * aDifference;
    }

    anError = std::max( sqrt( anError / theSystemSize ), 1e-10 );

    if ( anError < 1.0 )
        return anError;

    if ( theFirstStepFlag || theRejectedStepCounter != 0 )
    {
        fireProcesses();
        setVariableVelocity( theW[ 4 ] );

        for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
        {
            gsl_vector_set( theVelocityVector1, c,
                            theW[ 4 ][ c ]
                            + theW[ 0 ][ c ] * hee1
                            + theW[ 1 ][ c ] * hee2
                            + theW[ 2 ][ c ] * hee3 );
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

void SpatiocyteODEStepper::updateInternalStateRadauIIA( Real aStepInterval )
{
    if ( !theJacobianMatrix1 )
    {
        DifferentialStepper::updateInternalState( aStepInterval );
        return;
    }

    theStateFlag = false;

    Real const aPreviousStepInterval( getStepInterval() );
    clearVariables();

    theRejectedStepCounter = 0;

    fireProcesses();
    setVariableVelocity( theW[ 3 ] );

    if ( theJacobianCalculateFlag || isInterrupted )
    {
        calculateJacobian();
        setJacobianMatrix( aStepInterval );
    }
    else
    {
        if ( aPreviousStepInterval != aStepInterval )
            setJacobianMatrix( aStepInterval );
    }

    for ( ;; )
    {
        std::pair< bool, Real > const aResult( calculateRadauIIA( aStepInterval, aPreviousStepInterval ) );
        if ( aResult.first )
        {
            break;
        }

        aStepInterval = aResult.second;

        if ( ++theRejectedStepCounter >= getTolerableRejectedStepCount() )
        {
            THROW_EXCEPTION_INSIDE( SimulationError,
                String( "The times of rejections of step calculation "
                    "exceeded a maximum tolerable count (" )
                + stringCast( getTolerableRejectedStepCount() ) + ")." );
        }

        if ( !theJacobianCalculateFlag )
        {
            calculateJacobian();
            theJacobianCalculateFlag = true;
        }

        setJacobianMatrix( aStepInterval );
    }

    setTolerableStepInterval( aStepInterval );

    setSpectralRadius( calculateJacobianNorm() );

    // theW will already be transformed to Z-form

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        theW[ 3 ][ c ] = theW[ 2 ][ c ];
        theW[ 3 ][ c ] /= aStepInterval;

        theVariableVector[ c ]->setValue( theValueBuffer[ c ] );
    }

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; c++ )
    {
        const Real z1( theW[ 0 ][ c ] );
        const Real z2( theW[ 1 ][ c ] );
        const Real z3( theW[ 2 ][ c ] );

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

    // an extra calculation for resetting the activities of processes
    fireProcesses();

    DifferentialStepper::updateInternalState( aStepInterval );
}

bool SpatiocyteODEStepper::calculate( Real aStepInterval )
{
    const Real eps_rel( getTolerance() );
    const Real eps_abs( getTolerance() * getAbsoluteToleranceFactor() );
    const Real a_y( getStateToleranceFactor() );
    const Real a_dydt( getDerivativeToleranceFactor() );

    const Real aCurrentTime( getCurrentTime() );

    // ========= 1 ===========

    if ( isInterrupted )
    {
        interIntegrate();
        fireProcesses();
        setVariableVelocity( theTaylorSeries[ 0 ] );

        for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
        {
            Variable* const aVariable( theVariableVector[ c ] );

            aVariable->setValue( theTaylorSeries[ 0 ][ c ] * ( 1.0 / 5.0 )
                                  * aStepInterval
                                  + theValueBuffer[ c ] );
        }
    }
    else
    {
        for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
        {
            Variable* const aVariable( theVariableVector[ c ] );

            // get k1
            theTaylorSeries[ 0 ][ c ] = theW[ 5 ][ c ];

            aVariable->setValue( theTaylorSeries[ 0 ][ c ] * ( 1.0 / 5.0 )
                                  * aStepInterval + theValueBuffer[ c ] );
        }
    }

    // ========= 2 ===========
    setCurrentTime( aCurrentTime + aStepInterval * 0.2 );
    interIntegrate();
    fireProcesses();
    setVariableVelocity( theW[ 0 ] );

    for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        aVariable->setValue( ( theTaylorSeries[ 0 ][ c ] * ( 3.0 / 40.0 )
                                         + theW[ 0 ][ c ] * ( 9.0 / 40.0 ) )
                              * aStepInterval + theValueBuffer[ c ] );
    }

    // ========= 3 ===========
    setCurrentTime( aCurrentTime + aStepInterval * 0.3 );
    interIntegrate();
    fireProcesses();
    setVariableVelocity( theW[ 1 ] );

    for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        aVariable->setValue( ( theTaylorSeries[ 0 ][ c ] * ( 44.0 / 45.0 )
                                 - theW[ 0 ][ c ] * ( 56.0 / 15.0 )
                                 + theW[ 1 ][ c ] * ( 32.0 / 9.0 ) )
                              * aStepInterval + theValueBuffer[ c ] );
    }

    // ========= 4 ===========
    setCurrentTime( aCurrentTime + aStepInterval * 0.8 );
    interIntegrate();
    fireProcesses();
    setVariableVelocity( theW[ 2 ] );

    for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        aVariable->setValue( ( theTaylorSeries[ 0 ][ c ] * ( 19372.0 / 6561.0 )
                                - theW[ 0 ][ c ] * ( 25360.0 / 2187.0 )
                                + theW[ 1 ][ c ] * ( 64448.0 / 6561.0 )
                                - theW[ 2 ][ c ] * ( 212.0 / 729.0 ) )
                              * aStepInterval + theValueBuffer[ c ] );
    }

    // ========= 5 ===========
    setCurrentTime( aCurrentTime + aStepInterval * ( 8.0 / 9.0 ) );
    interIntegrate();
    fireProcesses();
    setVariableVelocity( theW[ 3 ] );

    for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        // temporarily set Y^6
        theTaylorSeries[ 1 ][ c ] = theTaylorSeries[ 0 ][ c ] * ( 9017.0 / 3168.0 )
                - theW[ 0 ][ c ] * ( 355.0 / 33.0 )
                + theW[ 1 ][ c ] * ( 46732.0 / 5247.0 )
                + theW[ 2 ][ c ] * ( 49.0 / 176.0 )
                - theW[ 3 ][ c ] * ( 5103.0 / 18656.0 );

        aVariable->setValue( theTaylorSeries[ 1 ][ c ] * aStepInterval
                              + theValueBuffer[ c ] );
    }

    // ========= 6 ===========

    // estimate stiffness
    Real aDenominator( 0.0 );
    Real aSpectralRadius( 0.0 );

    setCurrentTime( aCurrentTime + aStepInterval );
    interIntegrate();
    fireProcesses();
    setVariableVelocity( theW[ 4 ] );

    for ( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        theTaylorSeries[ 2 ][ c ] =
                theTaylorSeries[ 0 ][ c ] * ( 35.0 / 384.0 )
                // + theW[ 0 ][ c ] * 0.0
                + theW[ 1 ][ c ] * ( 500.0 / 1113.0 )
                + theW[ 2 ][ c ] * ( 125.0 / 192.0 )
                + theW[ 3 ][ c ] * ( -2187.0 / 6784.0 )
                + theW[ 4 ][ c ] * ( 11.0 / 84.0 );

        aDenominator +=
                ( theTaylorSeries[ 2 ][ c ] - theTaylorSeries[ 1 ][ c ] )
                * ( theTaylorSeries[ 2 ][ c ] - theTaylorSeries[ 1 ][ c ] );

        aVariable->setValue( theTaylorSeries[ 2 ][ c ] * aStepInterval
                              + theValueBuffer[ c ] );
    }

    // ========= 7 ===========
    setCurrentTime( aCurrentTime + aStepInterval );
    interIntegrate();
    fireProcesses();
    setVariableVelocity( theW[ 5 ] );

    // evaluate error
    Real maxError( 0.0 );

    for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        // calculate error
        const Real anEstimatedError(
                ( theTaylorSeries[ 0 ][ c ] * ( 71.0 / 57600.0 )
                + theW[ 1 ][ c ] * ( -71.0 / 16695.0 )
                + theW[ 2 ][ c ] * ( 71.0 / 1920.0 )
                + theW[ 3 ][ c ] * ( -17253.0 / 339200.0 )
                + theW[ 4 ][ c ] * ( 22.0 / 525.0 )
                + theW[ 5 ][ c ] * ( -1.0 / 40.0 ) ) * aStepInterval );

        aSpectralRadius +=
                ( theW[ 5 ][ c ] - theW[ 4 ][ c ] ) *
                    ( theW[ 5 ][ c ] - theW[ 4 ][ c ] );

        // calculate velocity for Xn+.5
        theTaylorSeries[ 1 ][ c ] =
            theTaylorSeries[ 0 ][ c ] * ( 6025192743.0 / 30085553152.0 )
            + theW[ 1 ][ c ] * ( 51252292925.0 / 65400821598.0 )
            + theW[ 2 ][ c ] * ( -2691868925.0 / 45128329728.0 )
            + theW[ 3 ][ c ] * ( 187940372067.0 / 1594534317056.0 )
            + theW[ 4 ][ c ] * ( -1776094331.0 / 19743644256.0 )
            + theW[ 5 ][ c ] * ( 11237099.0 / 235043384.0 );

        const Real aTolerance( eps_rel *
                               ( a_y * fabs( theValueBuffer[ c ] )
                                 + a_dydt * fabs( theTaylorSeries[ 2 ][ c ] ) *
                                   aStepInterval )
                               + eps_abs );

        const Real anError( fabs( anEstimatedError / aTolerance ) );

        if ( anError > maxError )
        {
            maxError = anError;
        }
    }

    aSpectralRadius /= aDenominator;
    aSpectralRadius    = sqrt( aSpectralRadius );

    resetAll(); // reset all value

    setMaxErrorRatio( maxError );
    setCurrentTime( aCurrentTime );

    if( maxError > 1.1 )
    {
        // reset the stepper current time
        reset();
        isInterrupted = true;
        return false;
    }

    for( VariableVector::size_type c( 0 ); c < theSystemSize; ++c )
    {
        const Real k1( theTaylorSeries[ 0 ][ c ] );
        const Real v_2( theTaylorSeries[ 1 ][ c ] );
        const Real v1( theTaylorSeries[ 2 ][ c ] );
        const Real k7( theW[ 5 ][ c ] );

        theTaylorSeries[ 1 ][ c ] = -4.0 * k1 + 8.0 * v_2 - 5.0 * v1 + k7;
        theTaylorSeries[ 2 ][ c ] = 5.0 * k1 - 16.0 * v_2 + 14.0 * v1 - 3.0 * k7;
        theTaylorSeries[ 3 ][ c ] = -2.0 * k1 + 8.0 * v_2 - 8.0 * v1 + 2.0 * k7;
    }

    // set the error limit interval
    isInterrupted = false;

    setSpectralRadius( aSpectralRadius / aStepInterval );

    return true;
}

void SpatiocyteODEStepper::updateInternalState( Real aStepInterval )
{
    // check the stiffness of this system by previous results
    if ( getCheckIntervalCount() > 0 )
    {
        if ( theStiffnessCounter % getCheckIntervalCount() == 1 )
        {
            if ( isStiff )
                setSpectralRadius( calculateJacobianNorm() );

            const Real lambdah( getSpectralRadius() * aStepInterval );

            if ( isStiff == ( lambdah < 3.3 * 0.8 ) )
            {
                if ( theStiffnessCounter
                         > getCheckIntervalCount() * getSwitchingCount() )
                {
                    setIntegrationType( !isStiff );
                }
            }
            else
            {
                theStiffnessCounter = 1;
            }
        }

        ++theStiffnessCounter;
    }

    if ( isStiff )
        updateInternalStateRadauIIA( aStepInterval );
    else
        AdaptiveDifferentialStepper::updateInternalState( aStepInterval );
}

#endif /* __SpatiocyteODEStepper_hpp */
