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

#include <algorithm>

#include "Process.hpp"
#include "Model.hpp"
#include "Variable.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"


namespace libecs
{

LIBECS_DM_INIT_STATIC( System, System );

/////////////////////// System


// Property slots

GET_METHOD_DEF( Polymorph, SystemList, System )
{
    SystemsCRange items( getBelongings< System >() );
    PolymorphVector retval;

    retval.reserve( items.size() );

    for ( SystemsCRange::iterator i( items.begin() );
            i != items.end(); ++i )
    {
        retval.push_back( (*i).first.getID() );
    }

    return retval;
}

GET_METHOD_DEF( Polymorph, VariableList, System )
{
    VariablesCRange items( getBelongings< Variable >() );
    PolymorphVector retval;

    retval.reserve( items.size() );

    for ( VariablesCRange::iterator i( items.begin() );
            i != items.end(); ++i )
    {
        retval.push_back( (*i).first.getID() );
    }

    return retval;
}

GET_METHOD_DEF( Polymorph, ProcessList, System )
{
    ProcessesCRange items( getBelongings< Process >() );
    PolymorphVector retval;

    retval.reserve( items.size() );

    for ( ProcessesCRange::iterator i( items.begin() );
            i != items.end(); ++i )
    {
        retval.push_back( (*i).first.getID() );
    }

    return retval;
}

SET_METHOD_DEF( String, StepperID, System )
{
    stepper_ = getModel()->getStepper( value );
    stepper_->registerSystem( this );
}

GET_METHOD_DEF( String, StepperID, System )
{
    return getStepper()->getID();
}

void System::startup()
{
    stepper_ = 0;
    sizeVariable_ = 0;
    entityListChanged_ = false;
}

System::~System()
{
    if ( stepper_ )
    {
        stepper_->removeSystem( this );
    }
    
    delete entities_;
}


Variable* const System::findSizeVariable() const
{
    try
    {
        return getVariable( "SIZE" );
    }
    catch ( const NotFound& )
    {
        const System* enclosingSystem( getEnclosingSystem() );
        BOOST_ASSERT( enclosingSystem != this );
        return enclosingSystem->findSizeVariable();
    }
}

GET_METHOD_DEF( Real, Size, System )
{
    return sizeVariable_->getValue();
}

void System::configureSizeVariable()
{
    sizeVariable_ = findSizeVariable();
}

void System::initialize()
{
    // first initialize enclosed systems
    for ( SystemMap::iterator i( systems_.begin() ); i != systems_.end() ; ++i )
    {
        i->second->initialize();
    }

    if ( !getStepper() )
    {
        THROW_EXCEPTION( InitializationFailed,
                         "No stepper is associated" );
    }

    for ( VariableMap::iterator i( variables_.begin() );
            i != variables_.end() ; ++i )
    {
        i->second->initialize();
    }

    //
    // Set Process::stepper_.
    // Process::initialize() is called in Stepper::initialize()
    //
    for ( ProcessMap::iterator i( processes_.begin() );
            i != processes_.end() ; ++i )
    {
        i->second->setStepper( getStepper() );
        i->second->initialize();
    }
}

void System::postInitialize()
{
    configureSizeVariable();
}

Process* System::getProcess( const String& id ) const
{
    ProcessMap::const_iterator i( processes_.find(
            LocalID( EntityType::PROCESS, id ) ) );
    if ( i == processes_.end() )
    {
        return NULLPTR;
    }
    return i->second;
}

Variable* System::getVariable( const String& id ) const
{
    VariableMap::const_iterator i( variables_.find(
            LocalID( EntityType::VARIABLE, id ) ) );
    if ( i == variables_.end() )
    {
        return NULLPTR;
    }
    return i->second;
}

System* System::getSystem( const String& id ) const
{
    SystemMap::const_iterator i( systems_.find(
            LocalID( EntityType::SYSTEM, id ) ) );
    if ( i == systems_.end() )
    {
        return NULLPTR;
    }
    return i->second;
}

System* System::getSystem( const SystemPath& sysPath ) const
{
    if ( sysPath.isEmpty() )
    {
        return const_cast<System*>( this );
    }

    if ( sysPath.isAbsolute() )
    {
        return model_->getSystem( sysPath );
    }

    System* retval( const_cast< System* >( this ) );

    for ( SystemPath::const_iterator i( sysPath.begin() ); i != sysPath.end();
            ++i )
    {
        retval = retval->getSystem( *i );
    }

    return retval;
}

void System::notifyChangeOfEntityList()
{
    //    getStepper()->getMasterStepper()->setEntityListChanged();
}

const SystemPath System::getPath() const
{
    SystemPath retval( model_->getFullIDOf( this ).getSystemPath() );
    return retval;
}

void System::add( const String& id, Entity* ent )
{
    switch ( ent->getEntityType().code ) {
    case EntityType::_PROCESS:
        addProcess( id, reinterpret_cast<Process*>( ent ) );
        break;
    case EntityType::_VARIABLE:
        addVariable( id, reinterpret_cast<Variable*>( ent ) );
        break;
    case EntityType::_SYSTEM:
        addSystem( id, reinterpret_cast<System*>( ent ) );
        break;
    }
}

void System::addProcess( const String& id, Process* proc )
{
    LocalID localID( EntityType::PROCESS, id );
    if ( processes_.find( localID ) != processes_.end() )
    {
        delete proc;

        THROW_EXCEPTION( AlreadyExist,
                         "Process \"" + id + "\" already exists." );
    }

    proc->__setID( id );
    processes_[ localID ] = proc;
    proc->setEnclosingSystem( this );

    entityAdded( EntityEventDescriptor( this, proc, localID ) );
}

void System::addVariable( const String& id, Variable* var )
{
    LocalID localID( EntityType::VARIABLE, id );
    if ( variables_.find( localID ) != variables_.end() )
    {
        delete var;

        THROW_EXCEPTION( AlreadyExist,
                         "Variable \"" + id + "\" already exists." );
    }

    var->__setID( id );
    variables_[ localID ] = var;
    var->setEnclosingSystem( this );

    entityAdded( EntityEventDescriptor( this, var, localID ) );
}

void System::addSystem( const String& id, System* sys )
{
    LocalID localID( EntityType::SYSTEM, id );
    if ( systems_.find( localID ) != systems_.end() )
    {
        delete sys;

        THROW_EXCEPTION( AlreadyExist,
                         "System \"" + id + "\" already exists." );
    }

    sys->__setID( id );
    systems_[ localID ] = sys;
    sys->setEnclosingSystem( this );

    entityAdded( EntityEventDescriptor( this, sys, localID ) );
}

void System::setStepper( Stepper* obj )
{
    stepper_ = obj;
    setStepperID( obj ? obj->getID(): "" );
}

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
