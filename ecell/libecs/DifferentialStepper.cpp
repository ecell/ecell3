//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "libecs.hpp"

#include <limits>

#include "Util.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "VariableValueIntegrator.hpp"
#include "Process.hpp"
#include "Model.hpp"

#include "DifferentialStepper.hpp"

#include <boost/array.hpp>


namespace libecs
{

LIBECS_DM_INIT_STATIC( DifferentialStepper, Stepper );
        
void DifferentialStepper::Interpolant::setVariable( Variable* var )
{
    VariableVectorCRange vars( theStepper.getInvolvedVariables() );
    VariableVectorCRange::iterator pos(
            std::find( vars.begin(), vars.end(), var ) );
    BOOST_ASSERT( pos != vars.end() );
    theIndex = pos - vars.begin();
    libecs::Interpolant::setVariable( var );
}

const Real
DifferentialStepper::Interpolant::getDifference(
        RealParam aTime, RealParam anInterval ) const
{
    if ( !theStepper.theStateFlag || anInterval == 0.0 )
    {
        return 0.0;
    }

    const Real aTimeInterval1( aTime - theStepper.getCurrentTime() );
    const Real aTimeInterval2( aTimeInterval1 - anInterval );

    const RealMatrix& aTaylorSeries( theStepper.getTaylorSeries() );
    RealCptr aTaylorCoefficientPtr( aTaylorSeries.origin() + theIndex );

    // calculate first order.
    // here it assumes that always aTaylorSeries.size() >= 1

    // *aTaylorCoefficientPtr := aTaylorSeries[ 0 ][ theIndex ]
    Real aValue1( *aTaylorCoefficientPtr * aTimeInterval1 );
    Real aValue2( *aTaylorCoefficientPtr * aTimeInterval2 );


    // check if second and higher order calculations are necessary.
    // const RealMatrix::size_type aTaylorSize( aTaylorSeries.size() );
    const RealMatrix::size_type aTaylorSize( theStepper.getOrder() );
    if ( aTaylorSize >= 2 )
    {
        const Real
        aStepIntervalInv( 1.0 / theStepper.getTolerableStepInterval() );

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

            // LIBECS_PREFETCH( aTaylorCoefficientPtr + aStride, 0, 1 );
            --s;
        } while ( s != 0 );
    }

    return aValue1 - aValue2;
}

const Real
DifferentialStepper::Interpolant::getVelocity( TimeParam aTime ) const
{
    if ( !theStepper.theStateFlag )
    {
        return 0.0;
    }

    const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

    const RealMatrix& aTaylorSeries( theStepper.getTaylorSeries() );
    RealCptr aTaylorCoefficientPtr( aTaylorSeries.origin() + theIndex );

    // calculate first order.
    // here it assumes that always aTaylorSeries.size() >= 1

    // *aTaylorCoefficientPtr := aTaylorSeries[ 0 ][ theIndex ]
    Real aValue( *aTaylorCoefficientPtr );

    // check if second and higher order calculations are necessary.
    // const RealMatrix::size_type aTaylorSize( aTaylorSeries.size() );

    const RealMatrix::size_type aTaylorSize( theStepper.getStage() );
    if ( aTaylorSize >= 2 && aTimeInterval != 0.0 )
    {
        const RealMatrix::size_type aStride( aTaylorSeries.strides()[0] );

        Real aFactorialInv( 1.0 );

        RealMatrix::size_type s( 1 );

        const Real theta( aTimeInterval
                          / theStepper.getTolerableStepInterval() );

        do
        {
            // main calculation for the 2+ order
            ++s;

            aTaylorCoefficientPtr += aStride;
            const Real aTaylorCoefficient( *aTaylorCoefficientPtr );

            aFactorialInv *= theta * s;

            aValue += aTaylorCoefficient * aFactorialInv;

            // LIBECS_PREFETCH( aTaylorCoefficientPtr + aStride, 0, 1 );
        } while ( s != aTaylorSize );
    }

    return aValue;
}


DifferentialStepper::DifferentialStepper()
        :
        theNextStepInterval( 0.001 ),
        theTolerableStepInterval( 0.001 ),
        theStateFlag( true )
{
    ; // do nothing
}

DifferentialStepper::~DifferentialStepper()
{
    ; // do nothing
}

libecs::Interpolant* DifferentialStepper::createInterpolant()
{
    return new DifferentialStepper::Interpolant( *this );
}

void DifferentialStepper::initialize()
{
    Stepper::initialize();

    createInterpolants();

    theTaylorSeries.resize(
        boost::extents[ getStage() ][ getAffectedVariables().size() ]
    );

    initializeVariableReferenceList();
}

void DifferentialStepper::registerProcess( Process* proc )
{
    if ( !proc->isContinuous() )
    {
        THROW_EXCEPTION( ValueError, proc->getClassName()
                + " is not a continuous process" );
    }

    Stepper::registerProcess( proc );
}

DifferentialStepper::ExprComponent
DifferentialStepper::toExprComponent( const VariableReference varRef ) const
{
    return ExprComponent( getVariableIndex( varRef.getVariable() ),
            varRef.getCoefficient() );
}

void DifferentialStepper::initializeVariableReferenceList()
{
    const ProcessVectorRange continuousProcesses( getContinuousProcesses() );

    varRefsOfProcesses_.resize( continuousProcesses.size() );

    for ( ProcessVectorRange::size_type i( 0 );
                i < continuousProcesses.size(); ++i )
    {
        Process* const aProcess( continuousProcesses[ i ] );
        const Process::VarRefVectorCRange negVarRefs(
                aProcess->getNegativeVariableReferences() );
        const Process::VarRefVectorCRange posiVarRefs(
                aProcess->getPositiveVariableReferences() );
        VarRefs& varRefs( varRefsOfProcesses_[ i ] );

        varRefs.reserve( negVarRefs.size() + posiVarRefs.size() );

        std::transform( negVarRefs.begin(), negVarRefs.end(),
                std::back_inserter( varRefs ),
                std::bind1st( boost::mem_fun(
                    &DifferentialStepper::toExprComponent ), this ) );
        std::transform( posiVarRefs.begin(), posiVarRefs.end(),
                std::back_inserter( varRefs ),
                std::bind1st( boost::mem_fun(
                    &DifferentialStepper::toExprComponent ), this ) );
    }
}

void DifferentialStepper::setVariableVelocity(
        boost::detail::multi_array::sub_array<Real, 1> aVelocityBuffer )
{
    const ProcessVectorRange continuousProcesses( getContinuousProcesses() );

    for ( RealMatrix::index i( 0 );
            i < static_cast< RealMatrix::index >( aVelocityBuffer.size() );
            ++i )
    {
        aVelocityBuffer[ i ] = 0.0;
    }

    for ( ProcessVectorRange::size_type i( 0 );
                i < continuousProcesses.size(); ++i )
    {
        const Real activity( continuousProcesses[ i ]->getActivity() );
        const VarRefs& varRefs( varRefsOfProcesses_[ i ] );

        for ( VarRefs::const_iterator i( varRefs.begin() );
                i < varRefs.end(); i++ )
        {
            ExprComponent const& aComponent = *i;
            const RealMatrix::index anIndex(
                static_cast< RealMatrix::index >(
                    aComponent.first ) );
            aVelocityBuffer[ anIndex ] += aComponent.second * activity;
        }
    }
}

void DifferentialStepper::reset()
{
    // XXX: is this needed?
    for ( RealMatrix::index i( 0 ); i != getStage(); ++i )
    {
        for ( RealMatrix::index j( 0 ); j < getAffectedVariables().size(); ++j )
        {
            theTaylorSeries[ i ][ j ] = 0.0;
        }
    }

    Stepper::reset();
}


void DifferentialStepper::interIntegrate()
{
    Time const currentTime( getCurrentTime() );

    VariableVectorRange const readOnlyVariables( getReadOnlyVariables() );

    // save buffer only to read-only variables.
    for ( VariableVector::iterator i( readOnlyVariables.begin() );
            i < readOnlyVariables.end(); ++i )
    {
        (*i)->setValue( valueBuffer_[ i - variables_.begin() ] );
    }

    VariableVectorRange const readVariables( getReadVariables() );
    for ( VariableVector::iterator i( readVariables.begin() );
            i < readVariables.end(); ++i )
    {
        BOOST_ASSERT( (*i)->getVariableValueIntegrator() );
        (*i)->getVariableValueIntegrator()->integrate( currentTime );
    }
}

void DifferentialStepper::interrupt( TimeParam callerCurrentTime )
{
    const TimeDifference callerTimeScale(
            getModel()->getLastStepper()->getTimeScale() );
    const TimeDifference stepInterval( getStepInterval() );

    // If the step size of this is less than caller's timescale,
    // ignore this interruption.
    if ( callerTimeScale >= stepInterval )
    {
        return;
    }

    const Real aCurrentTime( getCurrentTime() );

    // aCallerTimeScale == 0 implies need for immediate reset
    if ( callerTimeScale != 0.0 )
    {
        // Shrink the next step size to that of caller's
        setNextStepInterval( callerTimeScale );

        const Real aNextStep( aCurrentTime + stepInterval );
        const Real aCallerNextStep( callerCurrentTime + callerTimeScale );

        // If the next step of this occurs *before* the next step
        // of the caller, just shrink step size of this Stepper.
        if ( aNextStep <= aCallerNextStep )
        {
            return;
        }


        // If the next step of this will occur *after* the caller,
        // reschedule this Stepper, as well as shrinking the next step size.
        //    setStepInterval( callerCurrentTime + ( aCallerTimeScale * 0.5 )
        //       - aCurrentTime );
    }
    else
    {
        // reset step interval to the default
        setNextStepInterval( 0.001 );
    }

    const Real aNewStepInterval( callerCurrentTime - aCurrentTime );

    setStepInterval( aNewStepInterval );
}


} // namespace libecs
/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
