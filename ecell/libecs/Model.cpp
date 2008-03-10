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
#include "StepperMaker.hpp"
#include "VariableMaker.hpp"
#include "ProcessMaker.hpp"
#include "SystemMaker.hpp"
#include "LoggerBroker.hpp"
#include "Stepper.hpp"
#include "SystemStepper.hpp"

#include "Model.hpp"

namespace libecs
{
Model::Model( PropertiedObjectMaker& maker )
        :
        theCurrentTime( 0.0 ),
        theLoggerBroker(),
        thePropertiedObjectMaker( maker ),
        theStepperMaker( thePropertiedObjectMaker ),
        theSystemMaker( thePropertiedObjectMaker ),
        theVariableMaker( thePropertiedObjectMaker ),
        theProcessMaker( thePropertiedObjectMaker ),
        theNullModule( "NULL" ), 
        theWorld( reinterpret_cast<DynamicModuleBase<System>&>(theNullModule ) ),
        theRootSystem( NULLPTR )
{
    theLoggerBroker.setModel( this );

    // initialize theRootSystem
    theRootSystem = getSystemMaker().make( "System" );
    theRootSystem->setModel( this );
    theRootSystem->setID( "/" );
    theRootSystem->setName( "The Root System" );
    theRootSystem->setEnclosingSystem( theWorld );
    theRootSystem->__libecs_init__();
    theWorld.add( theRootSystem );

    // initialize theSystemStepper
    theSystemStepper.setModel( this );
    theSystemStepper.setID( "___SYSTEM" );
    theScheduler.addEvent(
        StepperEvent( getCurrentTime()
                      + theSystemStepper.getStepInterval(),
                      &theSystemStepper ) );
    theWorld.setStepper( new SystemStepper( theNullModule ) );

    theLastStepper = &theSystemStepper;
}

Model::~Model()
{
    delete theRootSystem*;
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

void Model::createEntity( const String& aClassname,
                          const FullID& aFullID )
{
    if ( aFullID.getSystemPath().empty() )
    {
        THROW_EXCEPTION( BadFormat, "empty SystemPath." );
    }

    System* aContainerSystem*( getSystem( aFullID.getSystemPath() ) );

    VariablePtr aVariablePtr( NULLPTR );

    switch ( aFullID.getEntityType().code )
    {
    case EntityType::_VARIABLE:
        ProcessPtr   aProcessPtr( getVariableMaker().make( aClassname ) );
        aVariablePtr->setID( aFullID.getID() );
        aContainerSystem*->registerEntity( aVariablePtr );
        break;
    case EntityType::_PROCESS:
        aProcessPtr = getProcessMaker().make( aClassname );
        aProcessPtr->setID( aFullID.getID() );
        aContainerSystem*->registerEntity( aProcessPtr );
        break;
    case EntityType::_SYSTEM:
        System* aSystem*( getSystemMaker().make( aClassname ) );
        aSystem*->setID( aFullID.getID() );
        aSystem*->setModel( this );
        aContainerSystem*->registerEntity( aSystem* );
        break;
    default:
        THROW_EXCEPTION( ValueEerror,
                         "bad EntityType specified." );
    }
}

System& Model::getSystem( const SystemPath& aSystemPath ) const
{
    // 1. "" (empty) means Model itself, which is invalid for this method.
    // 2. Not absolute is invalid (not absolute implies not empty).
    if ( ( ! aSystemPath.isAbsolute() ) || aSystemPath.empty() )
    {
        THROW_EXCEPTION( BadFormat,
                         "[" + aSystemPath.getString() +
                         "] is not an absolute SystemPath." );
    }

    aSystemPathCopy.pop_front();

    return getRootSystem()->getSystem( aSystemPathCopy );
}


Entity& Model::getEntity( const FullID& aFullID ) const
{
    const LocalID localID( aFullID.getLocalID() );
    System& aSystem( getSystem( aFullID.getSystemPath() ) );
    Entity* retval( aSystem.getEntity( localID ) );
    if ( !retval )
    {
        THROW_EXCEPTION( NotFound,
            "entity does not exist: " + aFullID );
    }
    return *retval;
}


StepperPtr Model::getStepper( const String& anID ) const
{
    StepperMapConstIterator i( theStepperMap.find( anID ) );

    if ( i == theStepperMap.end() )
    {
        THROW_EXCEPTION( NotFound,
                         "Stepper [" + anID + "] not found in this model." );
    }

    return ( *i ).second;
}


void Model::createStepper( const String& aClassName, const String& anID )
{
    StepperPtr aStepper( getStepperMaker().make( aClassName ) );
    aStepper->setModel( this );
    aStepper->setID( anID );

    theStepperMap.insert( std::make_pair( anID, aStepper ) );

    theScheduler.
    addEvent( StepperEvent( getCurrentTime() + aStepper->getStepInterval(),
                            aStepper ) );
}


void Model::checkStepper( SystemCptr const aSystem ) const
{
    if ( aSystem->getStepper() == NULLPTR )
    {
        THROW_EXCEPTION( InitializationFailed,
                         "No stepper is connected with [" +
                         aSystem->getFullID().getString() + "]." );
    }

    for ( SystemMap::const_iterator i( aSystem->getSystemMap().begin() ) ;
            i != aSystem->getSystemMap().end() ; ++i )
    {
        // check it recursively
        checkStepper( i->second );
    }
}


void Model::initializeSystems( System* const aSystem ) const
{
    aSystem->initialize();

    for ( SystemMap::const_iterator i( aSystem->getSystemMap().begin() );
            i != aSystem->getSystemMap().end() ; ++i )
    {
        // initialize recursively
        initializeSystems( i->second );
    }
}

void Model::checkSizeVariable( SystemCptr const aSystem )
{
    FullID aRootSizeFullID( "Variable:/:SIZE" );

    try
    {
        IGNORE_RETURN getEntity( aRootSizeFullID );
    }
    catch ( const NotFound& )
    {
        createEntity( "Variable", aRootSizeFullID );
        EntityPtr aRootSizeVariable( getEntity( aRootSizeFullID ) );

        aRootSizeVariable->setProperty( "Value", Polymorph( 1.0 ) );
    }
}

void Model::initialize()
{
    System* aRootSystem( getRootSystem() );

    checkSizeVariable( aRootSystem );

    initializeSystems( aRootSystem );

    checkStepper( aRootSystem );

    // initialization of Stepper needs four stages:
    // (1) update current times of all the steppers, and integrate Variables.
    // (2) call user-initialization methods of Processes.
    // (3) call user-defined initialize() methods.
    // (4) post-initialize() procedures:
    //     - construct stepper dependency graph and
    //     - fill theIntegratedVariableVector.
    FOR_ALL_SECOND( StepperMap, theStepperMap, initializeProcesses );
    FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );
    theSystemStepper.initialize();

    FOR_ALL_SECOND( StepperMap, theStepperMap,
                    updateIntegratedVariableVector );

    theScheduler.updateEventDependency();

    for ( EventIndex c( 0 ); c != theScheduler.getSize(); ++c )
    {
        theScheduler.getEvent( c ).reschedule();
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
