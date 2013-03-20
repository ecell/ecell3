//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
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

    for( SystemMap::const_iterator i = theSystemMap.begin() ;
         i != theSystemMap.end() ; ++i )
    {
        aVector.push_back( Polymorph( i->second->getID() ) );
    }

    return Polymorph( aVector );
}


GET_METHOD_DEF( Polymorph, VariableList, System )
{
    PolymorphVector aVector;
    aVector.reserve( theVariableMap.size() );

    for( VariableMap::const_iterator i( theVariableMap.begin() );
         i != theVariableMap.end() ; ++i )
    {
        aVector.push_back( Polymorph( i->second->getID() ) );
    }

    return Polymorph( aVector );
}


GET_METHOD_DEF( Polymorph, ProcessList, System )
{
    PolymorphVector aVector;
    aVector.reserve( theProcessMap.size() );

    for( ProcessMap::const_iterator i( theProcessMap.begin() );
         i != theProcessMap.end() ; ++i )
    {
        aVector.push_back( Polymorph( i->second->getID() ) );
    }

    return Polymorph( aVector );
}


SET_METHOD_DEF( String, StepperID, System )
{
    theStepperID = value;
    theStepper = 0;
}


GET_METHOD_DEF( String, StepperID, System )
{
    return theStepperID;
}


System::System()
    : theStepper( 0 ),
      theSizeVariable( 0 )
{
    ; // do nothing
}


System::~System()
{
}

Variable const* System::findSizeVariable() const
{
    try
    {
        return getVariable( "SIZE" );
    }
    catch( NotFound const& )
    {
        System const* const aSuperSystem( getSuperSystem() );

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

void System::prepreinitialize()
{
    if ( isRootSystem() )
    {
        try
        {
            getVariable( "SIZE" );
        }
        catch( NotFound const& )
        {
            Variable& v( *reinterpret_cast< Variable* >(
                    theModel->createEntity( "Variable", FullID( "Variable:/:SIZE" ) ) ) );
            v.setValue( 1.0 );
        }
    }
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
    for ( ProcessMap::const_iterator i( theProcessMap.begin() );
          i != theProcessMap.end() ; ++i )
    {
        Process* aProcess( i->second );

        if( !aProcess->getStepper() )
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
    ProcessMap::const_iterator i( theProcessMap.find( anID ) );

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
    VariableMap::const_iterator i( theVariableMap.find( anID ) );

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

void System::unregisterEntity( SystemMap::iterator const& i )
{
    (*i).second->setSuperSystem( 0 );
    theSystemMap.erase( i ); 
    notifyChangeOfEntityList();    
}


System*
System::getSystem( SystemPath const& aSystemPath ) const
{
    if ( aSystemPath.isModel() )
    {
        THROW_EXCEPTION_INSIDE( BadSystemPath, 
                                "Cannot retrieve the model" );
    }

    if ( aSystemPath.isAbsolute() )
    {
        return theModel->getSystem( aSystemPath );
    }

    System* aRetval( const_cast< System* >( this ) );
    for ( SystemPath::const_iterator i( aSystemPath.begin() );
          i != aSystemPath.end(); ++i )
    {
        if ( *i == "." )
        {
            continue;
        }
        aRetval = aRetval->getSystem( *i );
    }

    return aRetval;
}
    

System*
System::getSystem( String const& anID ) const
{
    if ( anID[0] == '.' )
    {
        const String::size_type anIDSize( anID.size() );

        if ( anIDSize == 1 ) // == "."
        {
            return const_cast<System*>( this );
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

    SystemMap::const_iterator i( theSystemMap.find( anID ) );
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


SystemPath System::getSystemPath() const
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


void System::unregisterEntity( ProcessMap::iterator const& i )
{
    (*i).second->setSuperSystem( 0 );
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


void System::unregisterEntity( VariableMap::iterator const& i )
{
    (*i).second->setSuperSystem( 0 );
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


void System::unregisterEntity( Entity* anEntity )
{
    System const* const aSuperSystem( anEntity->getSuperSystem() );
    if ( !aSuperSystem )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                         asString() + ": " + anEntity->asString()
                         + " is not associated to any System" );
    }
    if ( aSuperSystem != this )
    {
        THROW_EXCEPTION_INSIDE( NotFound, 
                        asString() + ": " + anEntity->asString()
                        + " is associated to another system" );
    }
    unregisterEntity( anEntity->getEntityType(), anEntity->getID() );
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

void System::detach()
{
    typedef std::vector< Entity* > EntityVector;
    EntityVector entitiesToDetach;

    for ( SystemMap::iterator i( theSystemMap.begin() ),
                              e( theSystemMap.end() );
          i != e; ++i )
    {
        entitiesToDetach.push_back( ( *i ).second );
    }

    for ( ProcessMap::iterator i( theProcessMap.begin() ),
                               e( theProcessMap.end() );
          i != e; ++i )
    {
        entitiesToDetach.push_back( ( *i ).second );
    }

    for ( VariableMap::iterator i( theVariableMap.begin() ),
                                e( theVariableMap.end() );
          i != e; ++i )
    {
        entitiesToDetach.push_back( ( *i ).second );
    }

    for ( EntityVector::iterator i( entitiesToDetach.begin() ),
                                 e( entitiesToDetach.end() );
          i != e; ++i )
    {
        ( *i )->detach();
    }

    if ( theStepper )
    {
        try { theStepper->unregisterSystem( this ); } catch ( NotFound const& ) {}
        theStepper = 0;
    }

    theSizeVariable = 0;

    Entity::detach();
}

} // namespace libecs
