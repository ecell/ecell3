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

#include <boost/bind.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/value_type.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "dmtool/SharedModuleMaker.hpp"

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

const char Model::PATH_SEPARATOR = SharedModuleMakerInterface::PATH_SEPARATOR;

Model::Model( ModuleMaker< EcsObject >& maker )
    : theCurrentTime( 0.0 ),
      theNextHandleVal( 0 ),
      theLoggerBroker( *this ),
      theRootSystem( 0 ),
      theSystemStepper(),
      theEcsObjectMaker( maker ),
      theStepperMaker( theEcsObjectMaker ),
      theSystemMaker( theEcsObjectMaker ),
      theVariableMaker( theEcsObjectMaker ),
      theProcessMaker( theEcsObjectMaker ),
      isDirty_( false )
{
}

void Model::setup()
{
    registerBuiltinModules();

    // initialize theRootSystem
    theRootSystem = createSystem( "System" );
    theRootSystem->setID( "/" );
    theRootSystem->setName( "The Root System" );
    // super system of the root system is itself.
    theRootSystem->setSuperSystem( theRootSystem );

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
    std::vector< EcsObject* > objects;
    for ( HandleToObjectMap::const_iterator i( theObjectMap.begin() ),
                                            e( theObjectMap.end() );
          i != e; ++i )
    {
        objects.push_back( ( *i ).second );
    }

    std::for_each( objects.begin(), objects.end(),
            boost::bind( &EcsObject::dispose, _1 ) );

    std::for_each( objects.begin(), objects.end(), DeletePtr< EcsObject >() );
}


void Model::flushLoggers()
{
    theLoggerBroker.flush();
}


const PropertyInterfaceBase&
Model::getPropertyInterface( String const& aClassname ) const
{
    return *(reinterpret_cast<const PropertyInterfaceBase*>(
        theEcsObjectMaker.getModule( aClassname ).getInfo() ) );
}


Variable* Model::createVariable( String const& aClassname )
{
    Variable* retval( theVariableMaker.make( aClassname ) );
    retval->setModel( this );
    Handle nextHandle( generateNextHandle() );
    theObjectMap.insert( std::make_pair( nextHandle, retval ) );
    retval->setHandle( nextHandle );
    return retval;
}

Process* Model::createProcess( String const& aClassname )
{
    Process* retval( theProcessMaker.make( aClassname ) );
    retval->setModel( this );
    Handle nextHandle( generateNextHandle() );
    theObjectMap.insert( std::make_pair( nextHandle, retval ) );
    retval->setHandle( nextHandle );
    return retval;
}

System* Model::createSystem( String const& aClassname )
{
    System* retval( theSystemMaker.make( aClassname ) );
    retval->setModel( this );
    Handle nextHandle( generateNextHandle() );
    theObjectMap.insert( std::make_pair( nextHandle, retval ) );
    retval->setHandle( nextHandle );
    return retval;
}

Entity* Model::createEntity( String const& aClassname, FullIDCref aFullID )
{
    if( aFullID.getSystemPath().isModel() )
    {
        THROW_EXCEPTION( BadSystemPath, "empty SystemPath" );
    }

    System* aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );
    Entity* retval( 0 );
    
    switch( aFullID.getEntityType() )
    {
    case EntityType::VARIABLE:
        {
            retval = createVariable( aClassname );
            retval->setID( aFullID.getID() );
            aContainerSystemPtr->registerEntity(
                    static_cast< Variable* >( retval ) );
        }
        break;

    case EntityType::PROCESS:
        {
            retval = createProcess( aClassname );
            retval->setID( aFullID.getID() );
            aContainerSystemPtr->registerEntity(
                    static_cast< Process* >( retval ) );
        }
        break;

    case EntityType::SYSTEM:
        {
            retval = createSystem( aClassname );
            retval->setID( aFullID.getID() );
            aContainerSystemPtr->registerEntity(
                    static_cast< System* >( retval ) );
        }
        break;

    default:
        THROW_EXCEPTION( InvalidEntityType, "invalid EntityType specified" );
    }

    return retval;
}


SystemPtr Model::getSystem( SystemPathCref aSystemPath ) const
{
    return getRootSystem()->getSystem(
        aSystemPath.toRelative( SystemPath( "/" ) ) );
}


Entity* Model::getEntity( FullIDCref aFullID ) const
{
    Entity* anEntity( NULL );
    SystemPathCref aSystemPath( aFullID.getSystemPath() );
    String const&         anID( aFullID.getID() );

    if( aSystemPath.isModel() )
    {
        if( anID == "/" )
        {
            return getRootSystem();
        }
        else
        {
            THROW_EXCEPTION( BadID, 
                             "[" + aFullID.asString()
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


void Model::removeVariableReferences( System* aSystem, Variable const* aVariable )
{
    System::Processes processes( aSystem->getProcesses() );
    std::for_each( boost::begin( processes ), boost::end( processes ),
            ComposeUnary(
                boost::bind(
                    (bool(Process::*)(Variable const*, bool))&Process::removeVariableReference,
                    _1, aVariable, false ),
                SelectSecond< boost::range_value< System::Processes >::type >() ) );
}

void Model::detachObject( EcsObject* anObject )
{
    {
        Stepper* aStepper( dynamic_cast< Stepper* >( anObject ) );
        if ( aStepper )
        {
            theStepperMap.erase( aStepper->getID() );
        }
    }
    {
        Variable* aVariable( dynamic_cast< Variable* >( anObject ) );
        if ( aVariable )
        {
            removeVariableReferences( theRootSystem, aVariable );
        }
    }
    theObjectMap.erase( anObject->getHandle() );
    markDirty();
}


EcsObject* Model::getObject( Handle const& handle ) const
{
    HandleToObjectMap::const_iterator i( theObjectMap.find( handle ) );

    if ( i == theObjectMap.end() )
    {
        THROW_EXCEPTION( NotFound, "entity not found");
    }

    return i->second;
}



Stepper* Model::getStepper( String const& anID ) const
{
    StepperMapConstIterator i( theStepperMap.find( anID ) );

    if( i == theStepperMap.end() )
    {
        THROW_EXCEPTION( NotFound, "Stepper [" + anID + "] not found in "
                                   "this model." );
    }

    return i->second;
}


Stepper* Model::createStepper( String const& aClassName )
{
    Stepper* retval( theStepperMaker.make( aClassName ) );
    retval->setModel( this );
    Handle nextHandle( generateNextHandle() );
    theObjectMap.insert( std::make_pair( nextHandle, retval ) );
    retval->setHandle( nextHandle );
    return retval;
}

Stepper* Model::createStepper( String const& aClassName, String const& anID )
{
    Stepper* aStepper( createStepper( aClassName ) );
    aStepper->setID( anID );
    registerStepper( aStepper );
    return aStepper;
}

void Model::registerStepper( Stepper* aStepper )
{
    theStepperMap.insert( std::make_pair( aStepper->getID(), aStepper ) );

    theScheduler.addEvent(
        StepperEvent( getCurrentTime() + aStepper->getStepInterval(),
                      aStepper ) );

    markDirty();
}


void Model::preinitializeEntities( System* aSystem )
{
    aSystem->preinitialize();

    System::Variables variables( aSystem->getVariables() );
    std::for_each( boost::begin( variables ), boost::end( variables ),
            ComposeUnary( boost::bind( &Variable::preinitialize, _1 ),
                          SelectSecond< boost::range_value< System::Variables >::type >() ) );

    System::Processes processes( aSystem->getProcesses() );
    std::for_each( boost::begin( processes ), boost::end( processes ),
            ComposeUnary( boost::bind( &Process::preinitialize, _1 ),
                          SelectSecond< boost::range_value< System::Processes >::type >() ) );

    System::Systems systems( aSystem->getSystems() );
    std::for_each( boost::begin( systems ), boost::end( systems ),
            ComposeUnary( boost::bind( &Model::preinitializeEntities, _1 ),
                          SelectSecond< boost::range_value< System::Systems >::type >() ) );
}


void Model::initializeEntities( System* const aSystem )
{
    aSystem->initialize();

    if( aSystem->getStepper() == NULLPTR )
    {
        THROW_EXCEPTION( InitializationFailed,
                         "No stepper is connected with [" +
                         aSystem->getFullID().asString() + "]." );
    }

    System::Variables variables( aSystem->getVariables() );
    std::for_each( boost::begin( variables ), boost::end( variables ),
            ComposeUnary( boost::bind( &Variable::initialize, _1 ),
                          SelectSecond< boost::range_value< System::Variables >::type >() ) );

    System::Processes processes( aSystem->getProcesses() );
    std::for_each( boost::begin( processes ), boost::end( processes ),
            ComposeUnary( boost::bind( &Process::initialize, _1 ),
                          SelectSecond< boost::range_value< System::Processes >::type >() ) );

    System::Systems systems( aSystem->getSystems() );
    std::for_each( boost::begin( systems ), boost::end( systems ),
            ComposeUnary( boost::bind( &Model::initializeEntities, _1 ),
                          SelectSecond< boost::range_value< System::Systems >::type >() ) );
}


void Model::checkSizeVariable( System const* const aSystem )
{
    try
    {
        getRootSystem()->getVariable( "SIZE" );
    }
    catch( NotFound const& )
    {
        Variable& v( *reinterpret_cast< Variable* >(
                createEntity( "Variable", FullID( "Variable:/:SIZE" ) ) ) );
        v.setValue( 1.0 );
    }
}

void Model::initialize()
{
    SystemPtr aRootSystem( getRootSystem() );

    checkSizeVariable( aRootSystem );

    preinitializeEntities( aRootSystem );

    // initialization of Stepper needs four stages:
    // (1) update current times of all the steppers, and integrate Variables.
    // (2) call user-initialization methods of Processes.
    // (3) call user-defined initialize() methods.
    // (4) post-initialize() procedures:
    //         - construct stepper dependency graph and
    //         - fill theIntegratedVariableVector.
    initializeEntities( aRootSystem );

    std::for_each( theStepperMap.begin(), theStepperMap.end(),
        ComposeUnary( boost::bind( &Stepper::initialize, _1 ),
                      SelectSecond< StepperMap::value_type >() ) );
    theSystemStepper.initialize();

    std::for_each( theStepperMap.begin(), theStepperMap.end(),
        ComposeUnary( boost::bind( &Stepper::updateIntegratedVariableVector, _1 ),
                      SelectSecond< StepperMap::value_type >() ) );

    theScheduler.updateEventDependency();

    for( EventIndex c( 0 ); c != theScheduler.getSize(); ++c )
    {
        theScheduler.getEvent(c).reschedule();
    }

    isDirty_ = false;
}

void Model::setDMSearchPath( const std::string& path )
{
    SharedModuleMakerInterface* smmbase(
        dynamic_cast< SharedModuleMakerInterface* >( &theEcsObjectMaker ) );
    if ( !smmbase )
    {
        THROW_EXCEPTION( IllegalOperation,
                         "the ModuleMaker assigned to this model is not a "
                         "SharedModuleMaker.");
    }
    smmbase->setSearchPath( path );
}

const std::string Model::getDMSearchPath() const
{
    SharedModuleMakerInterface const* smmbase(
        dynamic_cast< SharedModuleMakerInterface const* >( &theEcsObjectMaker ) );
    if ( !smmbase )
    {
        THROW_EXCEPTION( IllegalOperation,
                         "the ModuleMaker assigned to this model is not a "
                         "SharedModuleMaker.");
    }
    return smmbase->getSearchPath();
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
    if ( isDirty_ )
    {
        flushLoggers();
        initialize();
    }

    StepperEventCref aNextEvent( theScheduler.getTopEvent() );
    theCurrentTime = aNextEvent.getTime();
    theLastStepper = aNextEvent.getStepper();

    theScheduler.step();
}

void Model::markDirty()
{
    isDirty_ = true;
}

Handle Model::generateNextHandle()
{
    if ( Handle::INVALID_HANDLE_VALUE == ++theNextHandleVal )
    {
        THROW_EXCEPTION( TooManyItems,
                         "too many entities or steppers created" );
    }
    return Handle( theNextHandleVal );
}


void Model::deleteStepper( String const& anID )
{
    Stepper* aStepper( getStepper( anID ) );

    if ( !aStepper->getProcessVector().empty() )
    {
        THROW_EXCEPTION( IllegalOperation,
                "Stepper [" + anID + "] is relied on by one or more processes" );
    }

    aStepper->detach();
}


void Model::deleteEntity( FullID const& aFullID )
{
    getEntity( aFullID )->detach();
}


} // namespace libecs
