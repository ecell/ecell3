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

#include "Util.hpp"
#include "EntityType.hpp"
#include "LoggerManager.hpp"
#include "Stepper.hpp"
#include "SystemStepper.hpp"
#include "System.hpp"
#include "SimulationContext.hpp"
#include "Model.hpp"
#include "PropertiedObjectMaker.hpp"

namespace libecs
{

void Model::EntityEventObserver::entityAdded( Descriptor desc )
{
    FullID fullID( desc.id, model_->getFullIDOf( desc.system ).getSystemPath() );

    switch ( fullID.getEntityType().code ) {
    case EntityType::_SYSTEM:
        model_->systems_.insert( std::make_pair( fullID, desc.entity ) );
        break;
    case EntityType::_PROCESS:
        model_->processes_.insert( std::make_pair( fullID, desc.entity ) );
        break;
    case EntityType::_VARIABLE:
        model_->variables_.insert( std::make_pair( fullID, desc.entity ) );
        break;
    }
}

void Model::EntityEventObserver::entityRemoved( Descriptor desc )
{
    FullID fullID( desc.id, model_->getFullIDOf( desc.system ).getSystemPath() );

    switch ( fullID.getEntityType().code ) {
    case EntityType::_SYSTEM:
        model_->systems_.erase( fullID );
        break;
    case EntityType::_PROCESS:
        model_->processes_.erase( fullID );
        break;
    case EntityType::_VARIABLE:
        model_->variables_.erase( fullID );
        break;
    }
}

Model::~Model()
{
    std::for_each( systems_.begin(), systems_.end(),
            compose1( deleter< EntityMap::mapped_type >(),
                    select2nd< EntityMap::value_type >() ) );

    std::for_each( variables_.begin(), variables_.end(),
            compose1( deleter< EntityMap::mapped_type >(),
                    select2nd< EntityMap::value_type >() ) );

    std::for_each( processes_.begin(), processes_.end(),
            compose1( deleter< EntityMap::mapped_type >(),
                    select2nd< EntityMap::value_type >() ) );

    std::for_each( steppers_.begin(), steppers_.end(),
            compose1( deleter< StepperMap::mapped_type >(),
                    select2nd< StepperMap::value_type >() ) );
}

void Model::startup()
{
    // initialize theRootSystem
    rootSystem_ = new System();
    rootSystem_->setModel( this );
    rootSystem_->setName( "The Root System" );
    rootSystem_->setEnclosingSystem( simulationContext_->getWorld() );
    rootSystem_->startup();
    simulationContext_->getWorld()->add( "/", rootSystem_ );
    observer_.setModel( this );
}

FullID Model::getFullIDOf( const Entity* ent ) const
{
    switch (ent->getEntityType().code) {
    case EntityType::_SYSTEM:
        {
            EntityMap::const_iterator pos(
                    std::lower_bound( systems_.begin(), systems_.end(),
                        const_cast< Entity* >( ent ),
                        compose_2u_to_b(
                            std::less< EntityMap::mapped_type >(),
                            select2nd< EntityMap::value_type >(),
                            empty_unary_function<
                                EntityMap::mapped_type >()
                        )
                    ) );
            if ( pos != systems_.end() )
            {
                return pos->first;
            }
        }
        break;
    case EntityType::_PROCESS:
        {
            EntityMap::const_iterator pos(
                    std::lower_bound(processes_.begin(), processes_.end(),
                        const_cast< Entity* >( ent ),
                        compose_2u_to_b(
                            std::less< EntityMap::mapped_type >(),
                            select2nd< EntityMap::value_type >(),
                            empty_unary_function<
                                EntityMap::mapped_type >()
                        )
                    ) );
            if ( pos != processes_.end() )
            {
                THROW_EXCEPTION(
                    NotFound, "No such entity" );
            }
        }
        break;
    case EntityType::_VARIABLE:
        {
            EntityMap::const_iterator pos(
                    std::lower_bound(variables_.begin(), variables_.end(),
                        const_cast< Entity* >( ent ),
                        compose_2u_to_b(
                            std::less< EntityMap::mapped_type >(),
                            select2nd< EntityMap::value_type >(),
                            empty_unary_function<
                                EntityMap::mapped_type >()
                        )
                    ) );

            if ( pos != variables_.end() )
            {
                return pos->first;
            }
        }
        break;
    }
    THROW_EXCEPTION( NotFound, "No such entity" );
}

template<typename T_>
T_* Model::createEntity( const String& className )
{
    T_* ent( propertiedObjectMaker_->make<T_>( className ) );
    ent->setModel( this );
    ent->startup();
    return ent;
}

void Model::addEntity( const FullID& fullID, Entity* ent )
{
    System* sys( getSystem( fullID.getSystemPath() ) );
    sys->add( fullID.getID(), ent );
}

System* Model::getSystem( const SystemPath& systemPath, bool throwIfNotFound )
{
    if ( ! systemPath.isAbsolute() || systemPath.isEmpty() )
    {
        THROW_EXCEPTION(
            BadFormat,
            String("\"") + systemPath + "\" is not an absolute SystemPath." );
    }

    std::pair<SystemPath, String> p = systemPath.splitAtLast();
    LocalID localID( EntityType::SYSTEM, p.second);
    FullID fullID( localID, p.first );

    EntityMap::const_iterator pos( systems_.find( fullID ) );
    if ( pos == systems_.end() )
    {
        if ( throwIfNotFound )
        {
            THROW_EXCEPTION( NotFound, String( "No such system: " ) + fullID );
        }
        return 0;
    }

    return reinterpret_cast<System*>( pos->second );
}

const System* Model::getSystem( const SystemPath& systemPath, bool throwIfNotFound ) const
{
    return const_cast<Model*>(this)->getSystem( systemPath, throwIfNotFound );
}

Entity* Model::getEntity( const FullID& fullID, bool throwIfNotFound )
{
    EntityMap::const_iterator pos;

    switch ( fullID.getEntityType().code ) {
    case EntityType::_SYSTEM:
        pos = systems_.find( fullID );
        if ( pos == systems_.end() )
        {
            if ( throwIfNotFound )
            {
                THROW_EXCEPTION( NotFound, "No such system: " + fullID );
            }
            return 0;
        }
        break;
    case EntityType::_PROCESS:
        pos = processes_.find( fullID );
        if ( pos == processes_.end() )
        {
            if ( throwIfNotFound )
            {
                THROW_EXCEPTION( NotFound, "No such process: " + fullID );
            }
            return 0;
        }
        break;
    case EntityType::_VARIABLE:
        pos = variables_.find( fullID );
        if ( pos == variables_.end() )
        {
            if ( throwIfNotFound )
            {
                THROW_EXCEPTION( NotFound, "No such variable: " + fullID );
            }
            return 0;
        }
        break;
    }

    return pos->second;
}

const Entity* Model::getEntity( const FullID& fullID, bool throwIfNotFound ) const
{
    return const_cast< Model* >( this )->getEntity( fullID, throwIfNotFound );
}

Stepper* Model::getStepper( const String& anID )
{
    StepperMap::const_iterator i( steppers_.find( anID ) );

    if ( i == steppers_.end() )
    {
        THROW_EXCEPTION( NotFound, "No such stepper: " + anID );
    }

    return i->second;
}

const Stepper* Model::getStepper( const String& anID ) const
{
    return const_cast< Model* >( this )->getStepper( anID );
}

Stepper* Model::createStepper( const String& className )
{
    Stepper* stepper( propertiedObjectMaker_->make< Stepper >( className ) );
    stepper->setModel( this );
    stepper->startup(); 

    return stepper;
}

void Model::addStepper( const String& id, Stepper* stepper )
{
    steppers_.insert( std::make_pair( id, stepper ) );
}

void Model::initialize()
{
}


} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
