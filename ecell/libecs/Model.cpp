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
#include "LoggerBroker.hpp"
#include "Stepper.hpp"
#include "SystemStepper.hpp"

#include "Model.hpp"

#include "DiscreteTimeStepper.hpp"
#include "DiscreteEventStepper.hpp"
#include "PassiveStepper.hpp"
#include "System.hpp"
#include "Variable.hpp"

namespace libecs
{

Model::Model( StaticModuleMaker< EcsObject >& maker )
    : theCurrentTime( 0.0 ),
      theNextHandleVal( 0 ),
      theLoggerBroker( *this ),
      theRootSystemPtr(0),
      theSystemStepper(),
      theEcsObjectMaker( maker ),
      theStepperMaker( theEcsObjectMaker ),
      theSystemMaker( theEcsObjectMaker ),
      theVariableMaker( theEcsObjectMaker ),
      theProcessMaker( theEcsObjectMaker )
{
    registerBuiltinModules();

    // initialize theRootSystem
    theRootSystemPtr = theSystemMaker.make( "System" );
    theRootSystemPtr->setModel( this );
    theRootSystemPtr->setID( "/" );
    theRootSystemPtr->setName( "The Root System" );
    // super system of the root system is itself.
    theRootSystemPtr->setSuperSystem( theRootSystemPtr );

 
    // initialize theSystemStepper
    theSystemStepper.setModel( this );
    theSystemStepper.setID( "___SYSTEM" );
    theScheduler.addEvent(
        StepperEvent( getCurrentTime() + theSystemStepper.getStepInterval(),
                      &theSystemStepper ) );

    theLastStepper = &theSystemStepper;
}

Model::~Model()
{
    delete theRootSystemPtr;
    for ( StepperMapConstIterator i( theStepperMap.begin() );
          i != theStepperMap.end(); ++i )
    {
        delete i->second;
    }
}


void Model::flushLoggers()
{
    theLoggerBroker.flush();
}


const PropertyInterfaceBase&
Model::getPropertyInterface( StringCref aClassname ) const
{
    return *(reinterpret_cast<const PropertyInterfaceBase*>(
        theEcsObjectMaker.getModule( aClassname ).getInfo() ) );
}


void Model::createEntity( StringCref aClassname, FullIDCref aFullID )
{
    if( aFullID.getSystemPath().empty() )
    {
        THROW_EXCEPTION( BadSystemPath, "Empty SystemPath." );
    }

    SystemPtr aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );

    switch( aFullID.getEntityType() )
    {
    case EntityType::VARIABLE:
        {
            Variable* aVariablePtr( theVariableMaker.make( aClassname ) );
            aVariablePtr->setID( aFullID.getID() );
            aVariablePtr->setModel( this );
            Handle nextHandle( generateNextHandle() );
            theObjectMap.insert(
                std::make_pair(
                    nextHandle,
                    boost::intrusive_ptr< EcsObject >(
                        aVariablePtr, true ) ) );
            aVariablePtr->setHandle( nextHandle );
            aContainerSystemPtr->registerVariable( aVariablePtr );
            return;
        }

    case EntityType::PROCESS:
        {
            Process* aProcessPtr( theProcessMaker.make( aClassname ) );
            aProcessPtr->setID( aFullID.getID() );
            aProcessPtr->setModel( this );
            Handle nextHandle( generateNextHandle() );
            theObjectMap.insert(
                std::make_pair(
                    nextHandle,
                    boost::intrusive_ptr< EcsObject >(
                        aProcessPtr, true ) ) );
            aProcessPtr->setHandle( nextHandle );
            aContainerSystemPtr->registerProcess( aProcessPtr );
            return;
        }

    case EntityType::SYSTEM:
        {
            System* aSystemPtr( theSystemMaker.make( aClassname ) );
            aSystemPtr->setID( aFullID.getID() );
            aSystemPtr->setModel( this );
            Handle nextHandle( generateNextHandle() );
            theObjectMap.insert(
                std::make_pair(
                    nextHandle,
                    boost::intrusive_ptr< EcsObject >(
                        aSystemPtr, true ) ) );
            aSystemPtr->setHandle( nextHandle );
            aContainerSystemPtr->registerSystem( aSystemPtr );
            return;
        }
    }

    THROW_EXCEPTION( InvalidEntityType, "Invalid EntityType specified." );
}


SystemPtr Model::getSystem( SystemPathCref aSystemPath ) const
{
    SystemPath aSystemPathCopy( aSystemPath );

    // 1. "" (empty) means Model itself, which is invalid for this method.
    // 2. Not absolute is invalid (not absolute implies not empty).
    if( ( ! aSystemPathCopy.isAbsolute() ) || aSystemPathCopy.empty() )
    {
        THROW_EXCEPTION( BadSystemPath, 
                         "[" + aSystemPath.getString() +
                         "] is not an absolute SystemPath." );
    }

    aSystemPathCopy.pop_front();

    return getRootSystem()->getSystem( aSystemPathCopy );
}


EntityPtr Model::getEntity( FullIDCref aFullID ) const
{
    EntityPtr anEntity( NULL );
    SystemPathCref aSystemPath( aFullID.getSystemPath() );
    StringCref         anID( aFullID.getID() );

    if( aSystemPath.empty() )
    {
        if( anID == "/" )
        {
            return getRootSystem();
        }
        else
        {
            THROW_EXCEPTION( BadID, 
                             "[" + aFullID.getString()
                             + "] is an invalid FullID" );
        }
    }

    SystemPtr aSystem ( getSystem( aSystemPath ) );

    switch( aFullID.getEntityType() )
    {
    case EntityType::VARIABLE:
        anEntity = aSystem->getVariable( aFullID.getID() );
        break;
    case EntityType::PROCESS:
        anEntity = aSystem->getProcess( aFullID.getID() );
        break;
    case EntityType::SYSTEM:
        anEntity = aSystem->getSystem( aFullID.getID() );
        break;
    default:
        THROW_EXCEPTION( InvalidEntityType, "bad EntityType specified." );
    }

    return anEntity;
}


StepperPtr Model::getStepper( StringCref anID ) const
{
    StepperMapConstIterator i( theStepperMap.find( anID ) );

    if( i == theStepperMap.end() )
    {
        THROW_EXCEPTION( NotFound, "Stepper [" + anID + "] not found in "
                                   "this model." );
    }

    return i->second;
}


void Model::createStepper( StringCref aClassName, StringCref anID )
{
    StepperPtr aStepper( theStepperMaker.make( aClassName ) );
    aStepper->setModel( this );
    aStepper->setID( anID );

    theStepperMap.insert( std::make_pair( anID, aStepper ) );

    theScheduler.addEvent(
        StepperEvent( getCurrentTime() + aStepper->getStepInterval(),
                      aStepper ) );
}


void Model::checkStepper( System const* const aSystem ) const
{
    if( aSystem->getStepper() == NULLPTR )
    {
        THROW_EXCEPTION( InitializationFailed,
                         "No stepper is connected with [" +
                         aSystem->getFullID().getString() + "]." );
    }

    for( System::SystemMapConstIterator i( aSystem->getSystemMap().begin() ) ;
         i != aSystem->getSystemMap().end() ; ++i )
    {
        // check it recursively
        checkStepper( i->second );
    }
}


void Model::initializeSystems( System* const aSystem )
{
    aSystem->initialize();

    for( System::SystemMapConstIterator i( aSystem->getSystemMap().begin() );
         i != aSystem->getSystemMap().end() ; ++i )
    {
        // initialize recursively
        initializeSystems( i->second );
    }
}

void Model::checkSizeVariable( System const* const aSystem )
{
    FullID aRootSizeFullID( "Variable:/:SIZE" );

    try
    {
        IGNORE_RETURN getEntity( aRootSizeFullID );
    }
    catch( NotFoundCref )
    {
        createEntity( "Variable", aRootSizeFullID );
        EntityPtr aRootSizeVariable( getEntity( aRootSizeFullID ) );

        aRootSizeVariable->setProperty( "Value", Polymorph( 1.0 ) );
    }
}

void Model::initialize()
{
    SystemPtr aRootSystem( getRootSystem() );

    checkSizeVariable( aRootSystem );

    initializeSystems( aRootSystem );

    checkStepper( aRootSystem );

    // initialization of Stepper needs four stages:
    // (1) update current times of all the steppers, and integrate Variables.
    // (2) call user-initialization methods of Processes.
    // (3) call user-defined initialize() methods.
    // (4) post-initialize() procedures:
    //         - construct stepper dependency graph and
    //         - fill theIntegratedVariableVector.

    FOR_ALL_SECOND( StepperMap, theStepperMap, initializeProcesses );
    FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );
    theSystemStepper.initialize();

    FOR_ALL_SECOND( StepperMap, theStepperMap, 
                    updateIntegratedVariableVector );

    theScheduler.updateEventDependency();

    for( EventIndex c( 0 ); c != theScheduler.getSize(); ++c )
    {
        theScheduler.getEvent(c).reschedule();
    }
}

void Model::setDMSearchPath( const std::string& path )
{
    theEcsObjectMaker.setSearchPath( path );
}

const std::string Model::getDMSearchPath() const
{
    return theEcsObjectMaker.getSearchPath();
}

void Model::registerBuiltinModules()
{
    DM_NEW_STATIC( &theEcsObjectMaker, EcsObject, DiscreteEventStepper );
    DM_NEW_STATIC( &theEcsObjectMaker, EcsObject, DiscreteTimeStepper );
    DM_NEW_STATIC( &theEcsObjectMaker, EcsObject, PassiveStepper );
    DM_NEW_STATIC( &theEcsObjectMaker, EcsObject, System );
    DM_NEW_STATIC( &theEcsObjectMaker, EcsObject, Variable );
}

void Model::step()
{
    StepperEventCref aNextEvent( theScheduler.getTopEvent() );
    theCurrentTime = aNextEvent.getTime();
    theLastStepper = aNextEvent.getStepper();

    theScheduler.step();
}

Handle Model::generateNextHandle()
{
    if ( Handle::INVALID_HANDLE_VALUE == ++theNextHandleVal )
        THROW_EXCEPTION( TooManyItems, "Too many entities or steppers created" );
    return Handle( theNextHandleVal );
}

} // namespace libecs
