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
#include "Process.hpp"
#include "Model.hpp"

#include "DifferentialStepper.hpp"

#include <boost/array.hpp>


namespace libecs
{

LIBECS_DM_INIT_STATIC( DifferentialStepper, Stepper );
LIBECS_DM_INIT_STATIC( AdaptiveDifferentialStepper, Stepper );

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

void DifferentialStepper::initialize()
{
    Stepper::initialize();

    createInterpolants();

    theTaylorSeries.resize( boost::extents[ getStage() ][
                                static_cast< RealMatrix::index >(
                                    getReadOnlyVariableOffset() ) ] );

    // should registerProcess be overrided?
    if ( getDiscreteProcessOffset() < theProcessVector.size() )
    {
        for ( ProcessVectorConstIterator
                i( theProcessVector.begin() + getDiscreteProcessOffset() );
                i < theProcessVector.end(); ++i )
        {
            // XXX: To be addressed later.
            // std::cerr << "WARNING: Process [" << (*i)->getID() << "] is not continuous." << std::endl;
        }
    }

    initializeVariableReferenceList();

    // should create another method for property slot ?
    //    setNextStepInterval( getStepInterval() );

    //    theStateFlag = false;
}

void DifferentialStepper::initializeVariableReferenceList()
{
    const ProcessVector::size_type
    aDiscreteProcessOffset( getDiscreteProcessOffset() );

    theVariableReferenceListVector.clear();
    theVariableReferenceListVector.resize( aDiscreteProcessOffset );

    for ( ProcessVector::size_type i( 0 ); i < aDiscreteProcessOffset; ++i )
    {
        ProcessPtr const aProcess( theProcessVector[ i ] );

        const VariableReferenceVector& aVariableReferenceVector(
            aProcess->getVariableReferenceVector() );

        VariableReferenceVector::size_type const
        aZeroVariableReferenceOffset(
            aProcess->getZeroVariableReferenceOffset() );
        VariableReferenceVector::size_type const
        aPositiveVariableReferenceOffset(
            aProcess->getPositiveVariableReferenceOffset() );

        theVariableReferenceListVector[ i ].reserve(
            ( aVariableReferenceVector.size() -
              aPositiveVariableReferenceOffset +
              aZeroVariableReferenceOffset ) );

        for ( VariableReferenceVectorConstIterator
                anIterator( aVariableReferenceVector.begin() ),
                anEnd ( aVariableReferenceVector.begin() +
                        aZeroVariableReferenceOffset );
                anIterator < anEnd; ++anIterator )
        {
            VariableReference const& aVariableReference( *anIterator );

            theVariableReferenceListVector[ i ].push_back(
                ExprComponent( getVariableIndex(
                                   aVariableReference.getVariable() ),
                               aVariableReference.getCoefficient() ) );
        }

        for ( VariableReferenceVectorConstIterator anIterator(
                    aVariableReferenceVector.begin() +
                    aPositiveVariableReferenceOffset );
                anIterator < aVariableReferenceVector.end();
                ++anIterator )
        {
            VariableReference const& aVariableReference( *anIterator );

            theVariableReferenceListVector[ i ].push_back(
                ExprComponent( getVariableIndex(
                                   aVariableReference.getVariable() ),
                               aVariableReference.getCoefficient() ) );
        }
    }
}

void DifferentialStepper::
setVariableVelocity( boost::detail::multi_array::sub_array<Real, 1>
                     aVelocityBuffer )
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

        for ( VariableReferenceList::const_iterator
                anIterator( theVariableReferenceListVector[ i ].begin() );
                anIterator < theVariableReferenceListVector[ i ].end();
                anIterator++ )
        {
            ExprComponent const& aComponent = *anIterator;
            const RealMatrix::index anIndex(
                static_cast< RealMatrix::index >(
                    aComponent.first ) );
            aVelocityBuffer[ anIndex ] += aComponent.second * anActivity;
        }
    }
}

void DifferentialStepper::reset()
{
    // is this needed?
    for ( RealMatrix::index i( 0 ); i != getStage(); ++i )
        for ( RealMatrix::index j( 0 );
                j != getReadOnlyVariableOffset(); ++j )
        {
            theTaylorSeries[ i ][ j ] = 0.0;

            // RealMatrix::index_gen indices;
            // theTaylorSeries[ indices[ i ][ RealMatrix::index_range( 0, getReadOnlyVariableOffset() ) ] ].assign( 0.0 );
        }

    Stepper::reset();
}

void DifferentialStepper::resetAll()
{
    const VariableVector::size_type aSize( theVariableVector.size() );
    for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
        VariablePtr const aVariable( theVariableVector[ c ] );
        aVariable->loadValue( theValueBuffer[ c ] );
    }
}

void DifferentialStepper::interIntegrate()
{
    Real const aCurrentTime( getCurrentTime() );

    VariableVector::size_type c( theReadWriteVariableOffset );
    for ( ; c != theReadOnlyVariableOffset; ++c )
    {
        VariablePtr const aVariable( theVariableVector[ c ] );

        aVariable->interIntegrate( aCurrentTime );
    }

    // RealOnly Variables must be reset by the values in theValueBuffer
    // before interIntegrate().
    for ( ; c != theVariableVector.size(); ++c )
    {
        VariablePtr const aVariable( theVariableVector[ c ] );

        aVariable->loadValue( theValueBuffer[ c ] );
        aVariable->interIntegrate( aCurrentTime );
    }
}

void DifferentialStepper::interrupt( TimeParam aTime )
{
    const Real aCallerCurrentTime( aTime );

    const Real aCallerTimeScale( getModel()->getLastStepper()->getTimeScale() );
    const Real aStepInterval( getStepInterval() );

    // If the step size of this is less than caller's timescale,
    // ignore this interruption.
    if ( aCallerTimeScale >= aStepInterval )
    {
        return;
    }

    const Real aCurrentTime( getCurrentTime() );

    // aCallerTimeScale == 0 implies need for immediate reset
    if ( aCallerTimeScale != 0.0 )
    {
        // Shrink the next step size to that of caller's
        setNextStepInterval( aCallerTimeScale );

        const Real aNextStep( aCurrentTime + aStepInterval );
        const Real aCallerNextStep( aCallerCurrentTime + aCallerTimeScale );

        // If the next step of this occurs *before* the next step
        // of the caller, just shrink step size of this Stepper.
        if ( aNextStep <= aCallerNextStep )
        {
            return;
        }


        // If the next step of this will occur *after* the caller,
        // reschedule this Stepper, as well as shrinking the next step size.
        //    setStepInterval( aCallerCurrentTime + ( aCallerTimeScale * 0.5 )
        //       - aCurrentTime );
    }
    else
    {
        // reset step interval to the default
        setNextStepInterval( 0.001 );
    }

    const Real aNewStepInterval( aCallerCurrentTime - aCurrentTime );

    loadStepInterval( aNewStepInterval );
}


////////////////////////// AdaptiveDifferentialStepper

AdaptiveDifferentialStepper::AdaptiveDifferentialStepper()
        :
        theTolerance( 1.0e-6 ),
        theAbsoluteToleranceFactor( 1.0 ),
        theStateToleranceFactor( 1.0 ),
        theDerivativeToleranceFactor( 1.0 ),
        theEpsilonChecked( 0 ),
        theAbsoluteEpsilon( 0.1 ),
        theRelativeEpsilon( 0.1 ),
        safety( 0.9 ),
        theMaxErrorRatio( 1.0 )
{
    // use more narrow range
    setMinStepInterval( 1e-100 );
    setMaxStepInterval( 1e+10 );
}

AdaptiveDifferentialStepper::~AdaptiveDifferentialStepper()
{
    ; // do nothing
}


void AdaptiveDifferentialStepper::initialize()
{
    DifferentialStepper::initialize();

    //FIXME:!!
    //    theEpsilonChecked = ( theEpsilonChecked
    //     || ( theDependentStepperVector.size() > 1 ) );
}

void AdaptiveDifferentialStepper::step()
{
    theStateFlag = false;

    clearVariables();

    setStepInterval( getNextStepInterval() );
    //    setTolerableInterval( 0.0 );

    while ( !calculate() )
    {
        const Real anExpectedStepInterval( safety * getStepInterval()
                                           * pow( getMaxErrorRatio(),
                                                  -1.0 / getOrder() ) );
        // const Real anExpectedStepInterval( 0.5 * getStepInterval() );

        if ( anExpectedStepInterval > getMinStepInterval() )
        {
            // shrink it if the error exceeds 110%
            setStepInterval( anExpectedStepInterval );
        }
        else
        {
            setStepInterval( getMinStepInterval() );

            // this must return false,
            // so theTolerableStepInterval does NOT LIMIT the error.
            THROW_EXCEPTION( SimulationError,
                             "the error-limit step interval is too small." );

            calculate();
            break;
        }
    }

    // an extra calculation for resetting the activities of processes
    fireProcesses();

    setTolerableStepInterval( getStepInterval() );

    theStateFlag = true;

    // grow it if error is 50% less than desired
    const Real maxError( getMaxErrorRatio() );
    if ( maxError < 0.5 )
    {
        const Real aNewStepInterval( getTolerableStepInterval() * safety
                                     * pow( maxError ,
                                            -1.0 / ( getOrder() + 1 ) ) );
        // const Real aNewStepInterval( getStepInterval() * 2.0 );

        setNextStepInterval( aNewStepInterval );
    }
    else
    {
        setNextStepInterval( getTolerableStepInterval() );
    }

    /**
    // check the tolerances for Epsilon
    if ( isEpsilonChecked() )
      {
    const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

    for ( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
     VariablePtr const aVariable( theVariableVector[ c ] );
     
     const Real aTolerance( FMA( fabs( aVariable->getValue() ),
     theRelativeEpsilon,
     theAbsoluteEpsilon ) );

     const Real aVelocity( fabs( theVelocityBuffer[ c ] ) );
     
     if ( aTolerance < aVelocity * getStepInterval() )
       {
    setStepInterval( aTolerance / aVelocity );
       }
    }
      }
    */
}

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
