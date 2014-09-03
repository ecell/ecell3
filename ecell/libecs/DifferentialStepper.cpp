//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "libecs.hpp"

#include <limits>

#include "Util.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "Process.hpp"
#include "Model.hpp"

#include "DifferentialStepper.hpp"

#include <boost/array.hpp>


namespace libecs
{
LIBECS_DM_INIT_STATIC( DifferentialStepper, Stepper );

DifferentialStepper::DifferentialStepper()
    : theNextStepInterval( 0.001 ),
      theTolerableStepInterval( 0.001 )
{
    ; // do nothing
}


DifferentialStepper::~DifferentialStepper()
{
    ; // do nothing
}


void DifferentialStepper::initialize()
{
    Stepper::initialize();

    createInterpolants();

    theTaylorSeries.resize( boost::extents[ getStage() ][
                                static_cast< RealMatrix::index >(
                                    getReadOnlyVariableOffset() ) ] );

    // XXX: should go into registerProcess?
    /*
    if ( getDiscreteProcessOffset() < theProcessVector.size() )
    {
        for ( ProcessVectorConstIterator i(
                theProcessVector.begin() + getDiscreteProcessOffset() );
                i < theProcessVector.end(); ++i )
        {
            std::cerr << "WARNING: Process [" << (*i)->getID() << "] is not continuous." << std::endl;
        }
    }
    */

    initializeVariableReferenceList();
}


void DifferentialStepper::initializeVariableReferenceList()
{
    typedef Process::VariableReferenceVector VariableReferenceVector;
    const ProcessVector::size_type aDiscreteProcessOffset(
            getDiscreteProcessOffset() );

    theVariableReferenceListVector.clear();
    theVariableReferenceListVector.resize( aDiscreteProcessOffset );
    
    for ( ProcessVector::size_type i( 0 ); i < aDiscreteProcessOffset; ++i )
    {
        Process* const aProcess( theProcessVector[ i ] );

        VariableReferenceVector const& aVariableReferenceVector(
                aProcess->getVariableReferenceVector() );

        VariableReferenceVector::size_type const aZeroVariableReferenceOffset(
                aProcess->getZeroVariableReferenceOffset() );
        VariableReferenceVector::size_type const aPositiveVariableReferenceOffset(
                aProcess->getPositiveVariableReferenceOffset() );

        theVariableReferenceListVector[ i ].reserve(
                ( aVariableReferenceVector.size() - 
                        aPositiveVariableReferenceOffset + 
                        aZeroVariableReferenceOffset ) );

        for ( VariableReferenceVector::const_iterator
                anIterator( aVariableReferenceVector.begin() ),
                anEnd ( aVariableReferenceVector.begin()
                        + aZeroVariableReferenceOffset );
              anIterator < anEnd; ++anIterator )
        {
            VariableReference const& aVariableReference( *anIterator );

            theVariableReferenceListVector[ i ].push_back(
                ExprComponent( getVariableIndex(
                         aVariableReference.getVariable() ),
                         aVariableReference.getCoefficient() ) );
        }

        for ( VariableReferenceVector::const_iterator
                anIterator( aVariableReferenceVector.begin()
                            + aPositiveVariableReferenceOffset ); 
              anIterator < aVariableReferenceVector.end(); ++anIterator )
        {
            VariableReference const& aVariableReference( *anIterator );

            theVariableReferenceListVector[ i ].push_back(
                    ExprComponent( getVariableIndex(
                            aVariableReference.getVariable() ),
                            aVariableReference.getCoefficient() ) );
        }
    }
}


void DifferentialStepper::setVariableVelocity(
        boost::detail::multi_array::sub_array<Real, 1> aVelocityBuffer )
{
    const ProcessVector::size_type 
        aDiscreteProcessOffset( getDiscreteProcessOffset() );

    for ( RealMatrix::index i( 0 );
          i < static_cast< RealMatrix::index >( aVelocityBuffer.size() );
          ++i )
    {
        aVelocityBuffer[ i ] = 0.0;
    }

    for ( ProcessVector::size_type i( 0 ); i < aDiscreteProcessOffset; ++i )
    {
        const Real anActivity( theProcessVector[ i ]->getActivity() );

        for ( VariableReferenceList::const_iterator anIterator(
                theVariableReferenceListVector[ i ].begin() );
              anIterator < theVariableReferenceListVector[ i ].end();
              anIterator++ )
        {
            ExprComponent const& aComponent = *anIterator;
            const RealMatrix::index anIndex(
                    static_cast< RealMatrix::index >( aComponent.first ) );
            aVelocityBuffer[ anIndex ] += aComponent.second * anActivity;
        }
    }
}


void DifferentialStepper::reset()
{
    // is this needed?
    for ( RealMatrix::index i( 0 ); i != getStage(); ++i )
    {
        for ( RealMatrix::index j( 0 );
              j != getReadOnlyVariableOffset(); ++j )
        {
            theTaylorSeries[ i ][ j ] = 0.0;
        }
    }

    Stepper::reset();
}


void DifferentialStepper::resetAll()
{
    const VariableVector::size_type aSize( theVariableVector.size() );
    for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );
        aVariable->setValue( theValueBuffer[ c ] );
    }
}


void DifferentialStepper::interIntegrate()
{
    Real const aCurrentTime( getCurrentTime() );

    VariableVector::size_type c( theReadWriteVariableOffset );
    for( ; c != theReadOnlyVariableOffset; ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        aVariable->interIntegrate( aCurrentTime );
    }

    // RealOnly Variables must be reset by the values in theValueBuffer
    // before interIntegrate().
    for( ; c != theVariableVector.size(); ++c )
    {
        Variable* const aVariable( theVariableVector[ c ] );

        aVariable->setValue( theValueBuffer[ c ] );
        aVariable->interIntegrate( aCurrentTime );
    }
}


void DifferentialStepper::interrupt( Time aTime )
{
    const Real aCallerCurrentTime( aTime );

    const Real aCallerTimeScale( getModel()->getLastStepper()->getTimeScale() );
    const Real aStepInterval( getStepInterval() );

    // If the step size of this is less than caller's timescale,
    // ignore this interruption.
    if( aCallerTimeScale >= aStepInterval )
    {
        return;
    }

    const Real aCurrentTime( getCurrentTime() );

    // aCallerTimeScale == 0 implies need for immediate reset
    if( aCallerTimeScale != 0.0 )
    {
        // Shrink the next step size to that of caller's
        setNextStepInterval( aCallerTimeScale );

        const Real aNextStep( aCurrentTime + aStepInterval );
        const Real aCallerNextStep( aCallerCurrentTime + aCallerTimeScale );

        // If the next step of this occurs *before* the next step 
        // of the caller, just shrink step size of this Stepper.
        if( aNextStep <= aCallerNextStep )
        {
            return;
        }
    }

    const Real aNewStepInterval( aCallerCurrentTime - aCurrentTime );

    loadStepInterval( aNewStepInterval );
}

const Real DifferentialStepper::Interpolant::getDifference(
        Real aTime, Real anInterval ) const
{
    DifferentialStepper const* const theStepper( reinterpret_cast< DifferentialStepper const    * >( this->theStepper ) );

    const Real aTimeInterval1( aTime - theStepper->getCurrentTime() );
    const Real aTimeInterval2( aTimeInterval1 - anInterval );

    RealMatrix const& aTaylorSeries( theStepper->getTaylorSeries() );
    Real const* aTaylorCoefficientPtr( aTaylorSeries.origin() + theIndex );

    // calculate first order.
    // here it assumes that always aTaylorSeries.size() >= 1

    // *aTaylorCoefficientPtr := aTaylorSeries[ 0 ][ theIndex ]
    Real aValue1( *aTaylorCoefficientPtr * aTimeInterval1 );
    Real aValue2( *aTaylorCoefficientPtr * aTimeInterval2 );


    // check if second and higher order calculations are necessary.
    const RealMatrix::size_type aTaylorSize( theStepper->getOrder() );
    if( aTaylorSize >= 2)
    {
        const Real aStepIntervalInv(
            1.0 / theStepper->getTolerableStepInterval() );
        
        const RealMatrix::size_type aStride( aTaylorSeries.strides()[0] );

        Real aFactorialInv1( aTimeInterval1 );
        Real aFactorialInv2( aTimeInterval2 );

        RealMatrix::size_type s( aTaylorSize - 1 );

        const Real theta1( aTimeInterval1 * aStepIntervalInv );
        const Real theta2( aTimeInterval2 * aStepIntervalInv );

        do 
        {
            // main calculation for the 2+ order
            
            // aTaylorSeries[ s ][ theIndex ]
            aTaylorCoefficientPtr += aStride;
            const Real aTaylorCoefficient( *aTaylorCoefficientPtr );
            
            aFactorialInv1 *= theta1;
            aFactorialInv2 *= theta2;
            
            aValue1 += aTaylorCoefficient * aFactorialInv1;
            aValue2 += aTaylorCoefficient * aFactorialInv2;
            
            --s;
        } while( s != 0 );
    }

    return aValue1 - aValue2;
}


const Real DifferentialStepper::Interpolant::getVelocity( Real aTime ) const
{
    DifferentialStepper const* const theStepper( reinterpret_cast< DifferentialStepper const* >( this->theStepper ) );

    const Real aTimeInterval( aTime - theStepper->getCurrentTime() );

    RealMatrix const& aTaylorSeries( theStepper->getTaylorSeries() );
    Real const* aTaylorCoefficientPtr( aTaylorSeries.origin() + theIndex );

    // calculate first order.
    // here it assumes that always aTaylorSeries.size() >= 1

    // *aTaylorCoefficientPtr := aTaylorSeries[ 0 ][ theIndex ]
    Real aValue( *aTaylorCoefficientPtr );

    // check if second and higher order calculations are necessary.
    const RealMatrix::size_type aTaylorSize( theStepper->getStage() );
    if( aTaylorSize >= 2 && aTimeInterval != 0.0 )
    {
        const RealMatrix::size_type aStride( aTaylorSeries.strides()[0] );

        Real aFactorialInv( 1.0 );

        RealMatrix::size_type s( 1 );

        const Real theta( aTimeInterval 
                          / theStepper->getTolerableStepInterval() );

        do 
        {
            // main calculation for the 2+ order
            ++s;
            
            aTaylorCoefficientPtr += aStride;
            const Real aTaylorCoefficient( *aTaylorCoefficientPtr );
            
            aFactorialInv *= theta * s;
            
            aValue += aTaylorCoefficient * aFactorialInv;
            
            // LIBECS_PREFETCH( aTaylorCoefficientPtr + aStride, 0, 1 );
        } while( s != aTaylorSize );
    }

    return aValue;
}


Interpolant* DifferentialStepper::createInterpolant( Variable const* aVariable ) const
{
    return new DifferentialStepper::Interpolant( aVariable, this );
}

SET_METHOD_DEF( Real, StepInterval, DifferentialStepper )
{
    Stepper::setStepInterval( value );
    setTolerableStepInterval( value );
    setNextStepInterval( value );
}

} // namespace libecs
