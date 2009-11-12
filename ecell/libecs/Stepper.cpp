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

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <functional>
#include <algorithm>
#include <limits>
#include <boost/bind.hpp>

#include "Util.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "Process.hpp"
#include "Model.hpp"
#include "FullID.hpp"
#include "Logger.hpp"
#include "SystemStepper.hpp"

#include "Stepper.hpp"

#include "libecs.hpp"

namespace libecs
{

LIBECS_DM_INIT_STATIC( Stepper, Stepper );

////////////////////////// Stepper

Stepper::Stepper() 
    : theReadWriteVariableOffset( 0 ),
      theReadOnlyVariableOffset( 0 ),
      theDiscreteProcessOffset( 0 ),
      theSchedulerIndex( -1 ),
      thePriority( 0 ),
      theCurrentTime( 0.0 ),
      theStepInterval( 0.001 ),
      theMinStepInterval( 1e-100 ),
      theMaxStepInterval( std::numeric_limits<Real>::infinity() )
{
    gsl_rng_env_setup();

    theRng = gsl_rng_alloc( gsl_rng_default );

    setRngSeed( "TIME" );
}

Stepper::~Stepper()
{
    gsl_rng_free( theRng );
}


void Stepper::initialize()
{
    //
    // Update theVariableVector.    This also calls updateInterpolantVector.
    //
    updateVariableVector();

    // size of the value buffer == the number of *all* variables.
    // (not just read or write variables)
    theValueBuffer.resize( theVariableVector.size() );
}


void Stepper::updateProcessVector()
{
    // lighter implementation of this method is 
    // to merge this into registerProcess() and unregisterProcess() and
    // find a position to insert/remove each time.

    // sort by memory address. this is an optimization.
    std::sort( theProcessVector.begin(), theProcessVector.end() );

    // sort by Process priority, conserving the partial order in memory
    std::stable_sort( theProcessVector.begin(), theProcessVector.end(),
                      Process::PriorityCompare() );

    // partition by isContinuous().
    ProcessVectorConstIterator aDiscreteProcessIterator(
        std::stable_partition( theProcessVector.begin(),
                               theProcessVector.end(),
                               std::mem_fun( &Process::isContinuous ) ) );

    theDiscreteProcessOffset = aDiscreteProcessIterator - theProcessVector.begin();
}

void Stepper::updateVariableVector()
{
    DECLARE_MAP( VariablePtr, VariableReference, std::less<VariablePtr>,
                             PtrVariableReferenceMap );

    PtrVariableReferenceMap aPtrVariableReferenceMap;

    for( ProcessVectorConstIterator i( theProcessVector.begin());
         i != theProcessVector.end() ; ++i )
    {
        VariableReferenceVector const& aVariableReferenceVector(
            (*i)->getVariableReferenceVector() );

        // for all the VariableReferences
        for( VariableReferenceVectorConstIterator j(
                aVariableReferenceVector.begin() );
             j != aVariableReferenceVector.end(); ++j )
        {
            VariableReference const& aNewVariableReference( *j );
            VariablePtr aVariablePtr( aNewVariableReference.getVariable() );

            PtrVariableReferenceMapIterator 
                anIterator( aPtrVariableReferenceMap.find( aVariablePtr ) );

            if( anIterator == aPtrVariableReferenceMap.end() )
            {
                aPtrVariableReferenceMap.insert(
                    PtrVariableReferenceMap::value_type(
                        aVariablePtr, aNewVariableReference ) );
            }
            else
            {
                VariableReferenceRef aVariableReference( anIterator->second );

                aVariableReference.setIsAccessor(
                    aVariableReference.isAccessor()
                    || aNewVariableReference.isAccessor() );
                
                aVariableReference.setCoefficient(
                    abs( aVariableReference.getCoefficient() )
                    + abs( aNewVariableReference.getCoefficient() ) );
            }
        }
    }

    VariableReferenceVector aVariableReferenceVector;
    aVariableReferenceVector.reserve( aPtrVariableReferenceMap.size() );

    // I want select2nd... but use a for loop for portability.
    for( PtrVariableReferenceMapConstIterator 
            i( aPtrVariableReferenceMap.begin() );
         i != aPtrVariableReferenceMap.end() ; ++i )
    {
        aVariableReferenceVector.push_back( i->second );
    }
    
    VariableReferenceVectorIterator aReadOnlyVariableReferenceIterator = 
        std::partition( aVariableReferenceVector.begin(),
                        aVariableReferenceVector.end(),
                        std::mem_fun_ref( &VariableReference::isMutator ) );

    VariableReferenceVectorIterator aReadWriteVariableReferenceIterator = 
        std::partition( aVariableReferenceVector.begin(),
                        aReadOnlyVariableReferenceIterator,
                        std::not1(
                            std::mem_fun_ref( &VariableReference::isAccessor )
                        ) );

    theVariableVector.clear();
    theVariableVector.reserve( aVariableReferenceVector.size() );

    std::transform( aVariableReferenceVector.begin(),
                    aVariableReferenceVector.end(),
                    std::back_inserter( theVariableVector ),
                    std::mem_fun_ref( &VariableReference::getVariable ) );

    theReadWriteVariableOffset = aReadWriteVariableReferenceIterator 
        - aVariableReferenceVector.begin();

    theReadOnlyVariableOffset = aReadOnlyVariableReferenceIterator 
        - aVariableReferenceVector.begin();

    VariableVectorIterator aReadWriteVariableIterator = 
        theVariableVector.begin() + theReadWriteVariableOffset;
    VariableVectorIterator aReadOnlyVariableIterator = 
        theVariableVector.begin() + theReadOnlyVariableOffset;

    // For each part of the vector, sort by memory address. 
    // This is an optimization.
    std::sort( theVariableVector.begin(),  aReadWriteVariableIterator );
    std::sort( aReadWriteVariableIterator, aReadOnlyVariableIterator );
    std::sort( aReadOnlyVariableIterator,  theVariableVector.end() );
}


void Stepper::updateIntegratedVariableVector()
{
    theIntegratedVariableVector.clear();

    // want copy_if()...
    for( VariableVectorConstIterator i( theVariableVector.begin() );
         i != theVariableVector.end(); ++i )
    {
        VariablePtr aVariablePtr( *i );

        if( aVariablePtr->isIntegrationNeeded() )
        {
            theIntegratedVariableVector.push_back( aVariablePtr );
        }
    }

    // optimization: sort by memory address.
    std::sort( theIntegratedVariableVector.begin(), 
               theIntegratedVariableVector.end() );
}


void Stepper::createInterpolants()
{
    // create Interpolants.
    for( VariableVector::size_type c( 0 );    
         c != theReadOnlyVariableOffset; ++c )
    {
        VariablePtr aVariablePtr( theVariableVector[ c ] );
        aVariablePtr->registerInterpolant( createInterpolant( aVariablePtr ) );
    }
}


const bool Stepper::isDependentOn( Stepper const* aStepper )
{
    // Every Stepper depends on the SystemStepper.
    // FIXME: UGLY -- reimplement SystemStepper in a different way
    if( typeid( *aStepper ) == typeid( SystemStepper ) )
        {
        return true;
    }

    VariableVector const& aTargetVector( aStepper->getVariableVector() );
    
    VariableVectorConstIterator aReadOnlyTargetVariableIterator(
        aTargetVector.begin() +
            aStepper->getReadOnlyVariableOffset() );

    VariableVectorConstIterator aReadWriteTargetVariableIterator(
        aTargetVector.begin() +
            aStepper->getReadWriteVariableOffset() );
    
    // if at least one Variable in this::readlist appears in
    // the target::write list.
    for( VariableVectorConstIterator i( getVariableVector().begin() +
                                        theReadWriteVariableOffset ); 
         i != getVariableVector().end(); ++i )
    {
        VariablePtr const aVariablePtr( *i );
        
        // search in target::write or readwrite lists.
        if( std::binary_search( aTargetVector.begin(),    // write-only
                                aReadWriteTargetVariableIterator,
                                aVariablePtr )
            || std::binary_search( aReadWriteTargetVariableIterator,
                                   aReadOnlyTargetVariableIterator,
                                   aVariablePtr ) )
        {
            return true;
        }
    }
    
    return false;
}


GET_METHOD_DEF( Polymorph, SystemList, Stepper )
{
    PolymorphVector aVector;
    aVector.reserve( theSystemVector.size() );

    for( SystemVectorConstIterator i( getSystemVector().begin() );
         i != getSystemVector().end() ; ++i )
    {
        aVector.push_back( ( *i )->getFullID().asString() );
    }

    return aVector;
}


void Stepper::registerSystem( System* aSystemPtr )
{ 
    if( std::find( theSystemVector.begin(), theSystemVector.end(), aSystemPtr )
            == theSystemVector.end() )
    {
        theSystemVector.push_back( aSystemPtr );
    }
}

void Stepper::unregisterSystem( System* aSystemPtr )
{ 
    SystemVectorIterator i( find( theSystemVector.begin(), 
                                  theSystemVector.end(),
                                  aSystemPtr ) );
    
    if( i == theSystemVector.end() )
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": Failed to dissociate [" +
                                aSystemPtr->asString() + "] (no such System is "
                                "associated to this stepper)" );
    }

    theSystemVector.erase( i );
}


void Stepper::unregisterAllSystem()
{
    theSystemVector.clear();
}

void Stepper::registerProcess( Process* aProcess )
{ 
    BOOST_ASSERT( aProcess != NULLPTR );

    if( std::find( theProcessVector.begin(), theProcessVector.end(), 
                   aProcess ) == theProcessVector.end() )
    {
        theProcessVector.push_back( aProcess );
    }

    updateProcessVector();
}

void Stepper::unregisterProcess( ProcessPtr aProcess )
{ 
    ProcessVectorIterator ip( find( theProcessVector.begin(), 
                                    theProcessVector.end(),
                                    aProcess ) );
    
    if( ip == theProcessVector.end() )
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                                asString() + ": Failed to dissociate [" +
                                aProcess->asString() + "] (no such Process is "
                                "associated to this stepper)" );
    }

    typedef std::set< Variable* > VariableSet;
    VariableSet aVarSet;
    VariableReferenceVector const& aVarRefVector(
        aProcess->getVariableReferenceVector() );
    std::transform( aVarRefVector.begin(), aVarRefVector.end(),
                    inserter( aVarSet, aVarSet.begin() ),
                    boost::bind( &VariableReference::getVariable, _1 ) );

    VariableVector aNewVector;
    VariableVector::size_type aReadWriteVariableOffset,
                             aReadOnlyVariableOffset;

    for( VariableVector::iterator i( theVariableVector.begin() ),
                                  e( theVariableVector.begin()
                                     + theReadWriteVariableOffset );
         i != e; ++i )
    {
        VariableSet::iterator j( aVarSet.find( *i ) );
        if ( j != aVarSet.end() )
            aVarSet.erase( j );
        else
            aNewVector.push_back( *i );
    }

    aReadWriteVariableOffset = aNewVector.size();

    for( VariableVector::iterator i( theVariableVector.begin()
                                     + theReadWriteVariableOffset ),
                                  e( theVariableVector.begin()
                                     + theReadOnlyVariableOffset );
         i != e; ++i )
    {
        VariableSet::iterator j( aVarSet.find( *i ) );
        if ( j != aVarSet.end() )
            aVarSet.erase( j );
        else
            aNewVector.push_back( *i );
    }

    aReadOnlyVariableOffset = aNewVector.size();

    for( VariableVector::iterator i( theVariableVector.begin()
                                     + theReadOnlyVariableOffset ),
                                  e( theVariableVector.end() );
         i != e; ++i )
    {
        VariableSet::iterator j( aVarSet.find( *i ) );
        if ( j != aVarSet.end() )
            aVarSet.erase( j );
        else
            aNewVector.push_back( *i );
    }

    theVariableVector.swap( aNewVector );
    theReadWriteVariableOffset = aReadWriteVariableOffset;
    theReadOnlyVariableOffset = aReadOnlyVariableOffset;

    theProcessVector.erase( ip );
}


void Stepper::log()
{
    for ( ProcessVector::const_iterator i( theProcessVector.begin() ),
                                        end( theProcessVector.end() );
          i != end; ++i )
    {
        LoggerBroker::LoggersPerFullID loggers( (*i)->getLoggers() );
        std::for_each( loggers.begin(), loggers.end(),
                       boost::bind( &Logger::log, _1, theCurrentTime ) );
    }

    for ( VariableVector::const_iterator i( theVariableVector.begin() ),
                                         end( theVariableVector.begin()
                                              + theReadOnlyVariableOffset );
          i != end; ++i )
    {
        LoggerBroker::LoggersPerFullID loggers( (*i)->getLoggers() );
        std::for_each( loggers.begin(), loggers.end(),
                       boost::bind( &Logger::log, _1, theCurrentTime ) );
    }

    for ( SystemVector::const_iterator i( theSystemVector.begin() ),
                                         end( theSystemVector.end() );
          i != end; ++i )
    {
        LoggerBroker::LoggersPerFullID loggers( (*i)->getLoggers() );
        std::for_each( loggers.begin(), loggers.end(),
                       boost::bind( &Logger::log, _1, theCurrentTime ) );
    }
}

GET_METHOD_DEF( Polymorph, WriteVariableList, Stepper )
{
    PolymorphVector aVector;
    aVector.reserve( theVariableVector.size() );

    for( VariableVector::size_type c( 0 ); 
         c != theReadOnlyVariableOffset; ++c )
    {
        aVector.push_back( theVariableVector[c]->getFullID().asString() );
    }
    
    return aVector;
}


GET_METHOD_DEF( Polymorph, ReadVariableList, Stepper )
{
    PolymorphVector aVector;
    aVector.reserve( theVariableVector.size() );
    
    for( VariableVector::size_type c( theReadWriteVariableOffset ); 
         c != theVariableVector.size(); ++c )
    {
        aVector.push_back( theVariableVector[c]->getFullID().asString() );
    }
    
    return aVector;
}

GET_METHOD_DEF( Polymorph, ProcessList, Stepper )
{
    PolymorphVector aVector;
    aVector.reserve( theProcessVector.size() );
    
    for( ProcessVectorConstIterator i( theProcessVector.begin() );
         i != theProcessVector.end() ; ++i )
    {
        aVector.push_back( (*i)->getFullID().asString() );
    }
    
    return aVector;
}

const Stepper::VariableVector::size_type 
Stepper::getVariableIndex( Variable const* aVariable )
{
    VariableVectorConstIterator
        anIterator( std::find( theVariableVector.begin(), 
                               theVariableVector.end(), 
                               aVariable ) ); 
    return anIterator - theVariableVector.begin();
}


void Stepper::clearVariables()
{
    const VariableVector::size_type aSize( theVariableVector.size() );
    for( VariableVector::size_type c( 0 ); c < aSize; ++c )
    {
        VariablePtr const aVariable( theVariableVector[ c ] );

        // save original value values
        theValueBuffer[ c ] = aVariable->getValue();
    }

}

void Stepper::fireProcesses()
{
    FOR_ALL( ProcessVector, theProcessVector )
    {
        (*i)->fire();
    }
}


void Stepper::integrate( RealParam aTime )
{
    //
    // Variable::integrate()
    //
    //        FOR_ALL( VariableVector, theVariableVector )
    FOR_ALL( VariableVector, theIntegratedVariableVector )
    {
        (*i)->integrate( aTime );
    }

    // this must be after Variable::integrate()
    setCurrentTime( aTime );
}


void Stepper::reset()
{
    // restore original values and clear velocity of all the *write* variables.
    for( VariableVector::size_type c( 0 ); 
         c < getReadOnlyVariableOffset(); ++c )
    {
        VariablePtr const aVariable( theVariableVector[ c ] );

        // restore x (original value) and clear velocity
        aVariable->setValue( theValueBuffer[ c ] );
    }
}


SET_METHOD_DEF( String, RngSeed, Stepper )
{
    UnsignedInteger aSeed( 0 );
    
    if( value == "TIME" )
    {
        // Using just time() still gives the same seeds to Steppers
        // in multi-stepper model.    Stepper index is added to prevent this.
        aSeed = static_cast<UnsignedInteger>( time( NULLPTR )
                + getSchedulerIndex() );
    }
    else if( value == "DEFAULT" )
    {
        aSeed = gsl_rng_default_seed;
    }
    else
    {
        aSeed = stringCast<UnsignedInteger>( value );
    }

    gsl_rng_set( getRng(), aSeed );
}

GET_METHOD_DEF( String, RngType, Stepper )
{
    return gsl_rng_name( getRng() );
}


String Stepper::asString() const
{
    return getPropertyInterface().getClassName() + "[" + getID() + "]";
}

SET_METHOD_DEF( Real, StepInterval, Stepper )
{
    if ( value < getMinStepInterval() )
    {
        loadStepInterval( getMinStepInterval() );
        THROW_EXCEPTION_INSIDE( SimulationError,
                "The error-limit step interval is too small for ["
                + asString() + "]" );
    }
    else
    {
        loadStepInterval( value );
    }
}

GET_METHOD_DEF( Real, TimeScale, Stepper )
{
    return getStepInterval();
}

SET_METHOD_DEF( Real, MaxStepInterval, Stepper )
{
    issueWarning( "MaxStepInterval is no longer supported." );
    theMaxStepInterval = value;
}

Interpolant* Stepper::createInterpolant( Variable* aVariable )
{
    return new Interpolant( aVariable );
}

} // namespace libecs
