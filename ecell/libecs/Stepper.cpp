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

#include <functional>
#include <algorithm>
#include <limits>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/if.hpp>
#include <boost/lambda/bind.hpp>

#include "Util.hpp"
#include "Variable.hpp"
#include "VariableValueIntegrator.hpp"
#include "Interpolant.hpp"
#include "Process.hpp"
#include "Model.hpp"
#include "FullID.hpp"
#include "LoggerManager.hpp"
#include "SystemStepper.hpp"

#include "Stepper.hpp"


namespace libecs
{

LIBECS_DM_INIT_STATIC( Stepper, Stepper );

struct FullIDExtracter: public std::unary_function< const Entity*, FullID >
{
    typedef const Entity* argument_type;
    typedef FullID result_type;

    FullIDExtracter( const Model* model )
        : model_( model ) {}

    FullID operator()( const Entity* ent )
    {
        return model_->getFullIDOf( ent );
    }

    const Model* model_;
};

void Stepper::startup()
{
    _LIBECS_BASE_CLASS_::startup();

    schedulerIndex_ = -1;
    priority_ = 0;
    currentTime_ =  0.0;
    stepInterval_ =  0.001;
    minStepInterval_ = 0.0;
    maxStepInterval_ =  std::numeric_limits<Real>::infinity();
}

void Stepper::initialize()
{
    _LIBECS_BASE_CLASS_::initialize();

    // Update theVariableVector.  This also calls updateInterpolantVector.
    updateVariableVector();

    createInterpolants();

    prepareValueBuffer();
}

void Stepper::updateVariableVector()
{
    typedef std::map<const Variable*, VariableReference> VarRefMap;
    VarRefMap varRefMap;

    VarRefMap::size_type numOfAccessors;
    VarRefMap::size_type numOfAffectedVars;

    for ( ProcessVectorCRange::const_iterator i( processes_.begin() );
            i != processes_.end() ; ++i )
    {
        Process::VarRefsCRange varRefs( (*i)->getVariableReferences() );

        // for all the VariableReferences
        for ( Process::VarRefsCRange::iterator j( varRefs.begin() );
                j != varRefs.end(); ++j )
        {
            const VariableReference& varRef( *j );
            const Variable* var( j->getVariable() );
            VarRefMap::iterator pos( varRefMap.find( var ) );

            if ( pos == varRefMap.end() )
            {
                varRefMap[ var ] = varRef;
                if ( varRef.isMutator() )
                {
                    numOfAffectedVars++;
                    if ( varRef.isAccessor() )
                    {
                        numOfAccessors++;
                    }
                }
                
            }
            else
            {
                if ( !pos->second.isMutator() && varRef.isMutator() )
                {
                    pos->second.setCoefficient( 1 );
                    numOfAffectedVars++;

                    if ( !pos->second.isAccessor() && varRef.isAccessor() )
                    {
                        pos->second.setAccessor( true );
                        numOfAccessors++;
                    }
                }
            }
        }
    }

    variables_.clear();
    variables_.resize( varRefMap.size() );
    variables_.partition( 0, numOfAffectedVars - numOfAccessors );
    variables_.partition( 1, numOfAffectedVars );

    for ( VarRefMap::const_iterator i( varRefMap.begin() );
            i != varRefMap.end(); ++i )
    {
        if ( i->second.isMutator() )
        {
            if ( i->second.isAccessor() )
            {
                variables_.push_back( 1, i->second.getVariable() );
            }
            else
            {
                variables_.push_back( 0, i->second.getVariable() );
            }
        }
        else
        {
            variables_.push_back( 2, i->second.getVariable() );
        }
    }
    // optimization: sort by memory address.
    std::sort( variables_.begin( 0 ), variables_.end( 0 ) );
    std::sort( variables_.begin( 1 ), variables_.end( 1 ) );
    std::sort( variables_.begin( 2 ), variables_.end( 2 ) );
}


Interpolant* createInterpolant()
{
    return 0;
}

void Stepper::createInterpolants()
{
    // create Interpolants.
    VariableVectorRange affecteds( getAffectedVariables() );
    for ( VariableVector::const_iterator i( affecteds.begin() );
            i != affecteds.end(); ++i )
    {
        VariableValueIntegrator* integrator( (*i)->getVariableValueIntegrator() );
        if ( !integrator )
        {
            integrator = new VariableValueIntegrator( *i );
            (*i)->setVariableValueIntegrator( integrator );
        }

        integrator->addInterpolant( createInterpolant() );
    }
}

void Stepper::postInitialize()
{
    using namespace boost::lambda;

    variablesToIntegrate_.clear();
    VariableVectorRange affecteds( getAffectedVariables() );

    std::for_each( affecteds.begin(), affecteds.end(),
            if_then( bind( &Variable::isIntegrationNeeded, _1 ),
                bind( &VariableVector::push_back,
                    &variablesToIntegrate_, _1 ) ) );

    // optimization: sort by memory address.
    std::sort( variablesToIntegrate_.begin(),
               variablesToIntegrate_.end() );
}

bool Stepper::isDependentOn( const Stepper* aStepper ) const
{
    const VariableVectorCRange affecteds( aStepper->getInvolvedVariables() );
    const VariableVectorCRange readers( getReadVariables() );

    // if at least one Variable in this::readlist appears in
    // the target::write list.
    for ( VariableVectorCRange::iterator i( readers.begin() );
            i != readers.end(); ++i )
    {
        Variable* const var( *i );

        // search in target::write or readwrite lists.
        if ( std::binary_search( affecteds.begin(), affecteds.end(), var ) )
        {
            return true;
        }
    }

    return false;
}


GET_METHOD_DEF( Polymorph, SystemList, Stepper )
{
    PolymorphVector aVector;
    aVector.reserve( systems_.size() );

    for ( SystemSet::const_iterator i( systems_.begin() );
            i != systems_.end() ; ++i )
    {
        aVector.push_back( (*i)->getFullID().asString() );
    }

    return aVector;
}

void Stepper::registerSystem( System* sys )
{
    systems_.push_back( sys );
}

void Stepper::removeSystem( System* sys )
{
    SystemSet::iterator i( 
            std::find( systems_.begin(), systems_.end(), sys ) );

    if ( i == systems_.end() )
    {
        THROW_EXCEPTION( NotFound,
                         String( "system not associated: " ) + sys->asString() );
    }

    systems_.erase( i );
}

void Stepper::registerProcess( Process* proc )
{
    if ( std::find( processes_.begin(), processes_.end(),
                    proc ) == processes_.end() )
    {
        Processes::partition_index_type part( proc->isContinuous() ? 0: 1 );
        processes_.push_back( part, proc );
        std::stable_sort( processes_.begin( part ), processes_.end( part ),
                          Process::PriorityCompare() );
    }
}

void Stepper::removeProcess( Process* proc )
{
    Processes::iterator pos(
            std::find( processes_.begin(), processes_.end(), proc ) );

    if ( pos == processes_.end() )
    {
        THROW_EXCEPTION( NotFound,
                proc->asString() + " not found in this stepper. Can't remove." );
    }

    processes_.erase( pos );
    Processes::partition_index_type part( proc->isContinuous() ? 0: 1 );
    std::stable_sort( processes_.begin( part ), processes_.end( part ),
                      Process::PriorityCompare() );
}

void Stepper::log()
{
    for ( ProcessVector::const_iterator i( processes_.begin() );
            i < processes_.end(); ++i )
    {
        loggerManager_->log( currentTime_, (*i) );
    }

    VariableVectorRange affecteds( getAffectedVariables() );
    for ( Variables::const_iterator i( affecteds.begin() );
            i < affecteds.end(); ++i )
    {
        loggerManager_->log( currentTime_, (*i) );
    }

    for ( SystemSet::const_iterator i( systems_.begin() );
            i < systems_.end(); ++i )
    {
        loggerManager_->log( currentTime_, (*i) );
    }
}

GET_METHOD_DEF( Polymorph, WriteVariableList, Stepper )
{
    PolymorphVector retval;
    VariableVectorCRange vars( getAffectedVariables() );
    retval.reserve( vars.size() );

    for ( VariableVectorCRange::iterator i( vars.begin() );
            i != vars.end(); ++i )
    {
        retval.push_back( model_->getFullIDOf( *i ).asString() );
    }

    return retval;
}

GET_METHOD_DEF( Polymorph, ReadVariableList, Stepper )
{
    PolymorphVector retval;
    VariableVectorCRange vars( getReadVariables() );
    retval.reserve( vars.size() );

    for ( VariableVectorCRange::iterator i( vars.begin() );
            i != vars.end(); ++i )
    {
        retval.push_back( model_->getFullIDOf( *i ).asString() );
    }

    return retval;
}

GET_METHOD_DEF( Polymorph, ProcessList, Stepper )
{
    PolymorphVector retval;
    ProcessVectorCRange vars( getProcesses() );
    retval.reserve( vars.size() );

    for ( ProcessVectorCRange::iterator i( vars.begin() );
            i != vars.end(); ++i )
    {
        retval.push_back( model_->getFullIDOf( *i ).asString() );
    }

    return retval;
}

void Stepper::fireProcesses()
{
    std::for_each( processes_.begin(), processes_.end(),
            boost::mem_fun( &Process::fire ) );
}

void Stepper::integrate( RealParam aTime )
{
    using namespace boost::lambda;

    std::for_each( variablesToIntegrate_.begin(), variablesToIntegrate_.end(),
            bind( &VariableValueIntegrator::integrate,
                bind( &Variable::getVariableValueIntegrator, _1 ), aTime ) );
}

void Stepper::reset()
{
    saveBufferToVariables();
}

void Stepper::prepareValueBuffer()
{
    // size of the value buffer == the number of *all* variables.
    // (not just read or write variables)
    valueBuffer_.resize( variables_.size() );
}

void Stepper::loadVariablesToBuffer()
{
    for ( RealVector::size_type i( 0 ); i < valueBuffer_.size(); ++i )
    {
        valueBuffer_[ i ] = variables_[ i ]->getValue();
    }
}

void Stepper::saveBufferToVariables( bool onlyAffected )
{
    VariableVectorRange range(
        onlyAffected ? getAffectedVariables():
                getInvolvedVariables() );

    for ( VariableVector::iterator i( range.begin() ); i < range.end(); ++i )
    {
        (*i)->setValue( valueBuffer_[ i - variables_.begin() ] );
    }
}

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

