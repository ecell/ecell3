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

#include <algorithm>
#include <boost/bind.hpp>

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
    PolymorphVector aVector;
    aVector.reserve( theSystemMap.size() );

    for( SystemMapConstIterator i = theSystemMap.begin() ;
         i != theSystemMap.end() ; ++i )
    {
        aVector.push_back( i->second->getID() );
    }

    return aVector;
}


GET_METHOD_DEF( Polymorph, VariableList, System )
{
    PolymorphVector aVector;
    aVector.reserve( theVariableMap.size() );

    for( VariableMapConstIterator i( theVariableMap.begin() );
         i != theVariableMap.end() ; ++i )
    {
        aVector.push_back( i->second->getID() );
    }

    return aVector;
}


GET_METHOD_DEF( Polymorph, ProcessList, System )
{
    PolymorphVector aVector;
    aVector.reserve( theProcessMap.size() );

    for( ProcessMapConstIterator i( theProcessMap.begin() );
         i != theProcessMap.end() ; ++i )
    {
        aVector.push_back( i->second->getID() );
    }

    return aVector;
}


SET_METHOD_DEF( String, StepperID, System )
{
    theStepperID = value;
    theStepper = NULLPTR;
}


GET_METHOD_DEF( String, StepperID, System )
{
    return theStepperID;
}


System::System()
    : theStepper( NULLPTR ),
      theSizeVariable( NULLPTR )
{
    ; // do nothing
}


System::~System()
{
}

void System::dispose()
{
    if ( !disposed_ )
    {
        if( getStepper() )
        {
            getStepper()->unregisterSystem( this );
        }

        std::for_each( theProcessMap.begin(), theProcessMap.end(),
                ComposeUnary(
                    boost::bind( &Process::setSuperSystem, _1,
                                 static_cast< System* >( NULLPTR ) ),
                    SelectSecond< ProcessMap::value_type >() ) );
        theProcessMap.clear();
        std::for_each( theVariableMap.begin(), theVariableMap.end(),
                ComposeUnary(
                    boost::bind( &Variable::setSuperSystem, _1,
                                 static_cast< System* >( NULLPTR ) ),
                    SelectSecond< VariableMap::value_type >() ) );
        theVariableMap.clear();
        std::for_each( theSystemMap.begin(), theSystemMap.end(),
                ComposeUnary(
                    boost::bind( &System::setSuperSystem, _1,
                                 static_cast< System* >( NULLPTR ) ),
                    SelectSecond< SystemMap::value_type >() ) );
        theSystemMap.clear();
    }

    Entity::dispose();
}

Variable const* System::findSizeVariable() const
{
    try
    {
        return getVariable( "SIZE" );
    }
    catch( NotFound const& )
    {
        SystemCptr const aSuperSystem( getSuperSystem() );

        // Prevent infinite looping.    But this shouldn't happen.
        if( aSuperSystem == this )
        {
            THROW_EXCEPTION_INSIDE( UnexpectedError, 
                                    asString() + ": while trying get a SIZE "
                                    "variable, supersystem == this. "
                                    "Probably a bug." );
        }

        return aSuperSystem->findSizeVariable();
    }
}

GET_METHOD_DEF( Real, Size, System )
{
    return getSizeVariable()->getValue();
}

void System::configureSizeVariable()
{
    theSizeVariable = findSizeVariable();
}

void System::preinitialize()
{
    // no need to call subsystems' initialize() -- the Model does this
    if ( !theStepper )
    {
        theStepper = theModel->getStepper( theStepperID );
        theStepper->registerSystem( this );
    }

    //
    // Set Process::theStepper.
    // 
    for ( ProcessMapConstIterator i( theProcessMap.begin() );
          i != theProcessMap.end() ; ++i )
    {
        Process* aProcess( i->second );

        if( aProcess->getStepper() == NULLPTR )
        {
            aProcess->setStepper( getStepper() );
        }
    }

    configureSizeVariable();
}


void System::initialize()
{
}


Process*
System::getProcess( String const& anID ) const
{
    ProcessMapConstIterator i( theProcessMap.find( anID ) );

    if ( i == theProcessMap.end() )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": Process [" + anID
                         + "] not found in this System" );
    }

    return i->second;
}


Variable*
System::getVariable( String const& anID ) const
{
    VariableMapConstIterator i( theVariableMap.find( anID ) );

    if ( i == theVariableMap.end() )
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                         asString() + ": Variable [" + anID
                         + "] not found in this System");
    }

    return i->second;
}


void System::registerEntity( System* aSystem )
{
    const String anID( aSystem->getID() );

    if ( theSystemMap.find( anID ) != theSystemMap.end() )
    {
        THROW_EXCEPTION_INSIDE( AlreadyExist, 
                         asString() + ": System " + aSystem->asString()
                         + " is already associated" );
    }

    theSystemMap[ anID ] = aSystem;
    aSystem->setSuperSystem( this );

    notifyChangeOfEntityList();
}


void System::unregisterEntity( System* aSystem )
{
    System const* aSuperSystem( aSystem->getSuperSystem() );

    if ( !aSuperSystem )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                        asString() + ": System is not associated to "
                        "any System" );
    }
    else if ( aSuperSystem != this )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                        asString() + ": System is already associated to "
                        "another system" );
    }

    SystemMap::iterator i( theSystemMap.find( aSystem->getID() ) );
    if ( i == theSystemMap.end() || (*i).second != aSystem )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": System is not associated" );
    }

    unregisterEntity( i );
}


void System::unregisterEntity( SystemMap::iterator const& i )
{
    (*i).second->setSuperSystem( NULLPTR );
    theSystemMap.erase( i ); 
    notifyChangeOfEntityList();    
}


System*
System::getSystem( SystemPath const& aSystemPath ) const
{
    if ( aSystemPath.empty() )
    {
        return const_cast<SystemPtr>( this );
    }
    
    if ( aSystemPath.isAbsolute() )
    {
        return theModel->getSystem( aSystemPath );
    }

    SystemPtr const aNextSystem( getSystem( aSystemPath.front() ) );

    SystemPath aSystemPathCopy( aSystemPath );
    aSystemPathCopy.pop_front();

    return aNextSystem->getSystem( aSystemPathCopy );
}
    

System*
System::getSystem( String const& anID ) const
{
    if ( anID[0] == '.' )
    {
        const String::size_type anIDSize( anID.size() );

        if ( anIDSize == 1 ) // == "."
        {
            return const_cast<SystemPtr>( this );
        }
        else if ( anID[1] == '.' && anIDSize == 2 ) // == ".."
        {
            if ( isRootSystem() )
            {
                THROW_EXCEPTION_INSIDE( NotFound,
                                 asString() + ": the root system has no super "
                                 "systems" );
            }
            return getSuperSystem();
        }
    }

    SystemMapConstIterator i( theSystemMap.find( anID ) );
    if ( i == theSystemMap.end() )
    {
        THROW_EXCEPTION_INSIDE( NotFound,
                         asString() + ": System [" + anID + 
                         "] not found in this System" );
    }

    return i->second;
}


void System::notifyChangeOfEntityList()
{
    if ( theModel )
        theModel->markDirty();
}


Variable const* System::getSizeVariable() const
{
    if ( !theSizeVariable )
    {
        THROW_EXCEPTION_INSIDE( IllegalOperation,
                         asString() + ": SIZE variable is not associated" );
    }
    return theSizeVariable;
}


const SystemPath System::getSystemPath() const
{
    return isRootSystem() ? SystemPath(): Entity::getSystemPath();
}


void System::registerEntity( Process* aProcess )
{
    const String anID( aProcess->getID() );

    if ( theProcessMap.find( anID ) != theProcessMap.end() )
    {
        THROW_EXCEPTION_INSIDE( AlreadyExist, 
                         asString() + ": Process [" + anID
                         + "] is already associated" );
    }

    theProcessMap[ anID ] = aProcess;
    aProcess->setSuperSystem( this );

    notifyChangeOfEntityList();
}


void System::unregisterEntity( Process* aProcess )
{
    System const* aSuperSystem( aProcess->getSuperSystem() );

    if ( !aSuperSystem )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": Process [" + aProcess->asString()
                         + "] is not associated t0 any System" );
    }
    if ( aSuperSystem != this )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": Process ["
                         + aProcess->asString()
                         + "] is associated to another system" );
    }

    ProcessMap::iterator i( theProcessMap.find( aProcess->getID() ) );
    if ( i == theProcessMap.end() || (*i).second != aProcess )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": Process ["
                         + aProcess->asString() + "] is not associated" );
    }

    unregisterEntity( i );
}


void System::unregisterEntity( ProcessMap::iterator const& i )
{
    (*i).second->setSuperSystem( NULLPTR );
    theProcessMap.erase( i ); 
    notifyChangeOfEntityList();    
}


void System::registerEntity( Variable* aVariable )
{
    const String anID( aVariable->getID() );

    if ( theVariableMap.find( anID ) != theVariableMap.end() )
    {
        THROW_EXCEPTION_INSIDE( AlreadyExist, 
                         asString() + ": Variable [" + anID
                         + "] is already associated" );
    }

    theVariableMap[ anID ] = aVariable;
    aVariable->setSuperSystem( this );

    notifyChangeOfEntityList();
}


void System::unregisterEntity( Variable* aVariable )
{
    System const* aSuperSystem( aVariable->getSuperSystem() );

    if ( !aSuperSystem )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": Variable [" + aVariable->asString()
                         + "] is not associated to any System" );
    }
    if ( aSuperSystem != this )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                        asString() + ": Variable [" + aVariable->asString()
                        + "] is associated to another system" );
    }

    VariableMap::iterator i( theVariableMap.find( aVariable->getID() ) );
    if ( i == theVariableMap.end() || (*i).second != aVariable )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": Variable [" + aVariable->asString()
                         + "] is not associated" );
    }

    unregisterEntity( i );
}


void System::unregisterEntity( VariableMap::iterator const& i )
{
    (*i).second->setSuperSystem( NULLPTR );
    theVariableMap.erase( i ); 
    notifyChangeOfEntityList();    
}


void System::registerEntity( Entity* anEntity )
{
    switch ( anEntity->getEntityType() )
    {
    case EntityType::VARIABLE:
        registerEntity( static_cast< Variable* >( anEntity ) );
        break;
    case EntityType::PROCESS:
        registerEntity( static_cast< Process* >( anEntity ) );
        break;
    case EntityType::SYSTEM:
        registerEntity( static_cast< System* >( anEntity ) );
        break;
    default:
        THROW_EXCEPTION_INSIDE( InvalidEntityType, "invalid EntityType specified [" + anEntity->getEntityType().asString() + "]" );
    }
}


void System::unregisterEntity( EntityType const& anEntityType, String const& anID )
{
    switch ( anEntityType )
    {
    case EntityType::VARIABLE:
        {
            VariableMap::iterator i( theVariableMap.find( anID ) );
            if ( i == theVariableMap.end() )
            {
                THROW_EXCEPTION_INSIDE( NotFound, 
                                 asString() + ": Variable [" + anID
                                 + "] is not associated." );
            }
            unregisterEntity( i );
        }
        break;

    case EntityType::PROCESS:
        {
            ProcessMap::iterator i( theProcessMap.find( anID ) );
            if ( i == theProcessMap.end() )
            {
                THROW_EXCEPTION_INSIDE( NotFound, 
                                 asString() + ": Process [" + anID
                                 + "] is not associated." );
            }
            unregisterEntity( i );
        }
        break;
    case EntityType::SYSTEM:
        {
            SystemMap::iterator i( theSystemMap.find( anID ) );
            if ( i == theSystemMap.end() )
            {
                THROW_EXCEPTION_INSIDE( NotFound, 
                                 asString() + ": System [" + anID
                                 + "] is not associated." );
            }
            unregisterEntity( i );
        }
        break;
    }
}

} // namespace libecs
