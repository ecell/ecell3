//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
#include <iostream>

namespace libecs
{

  ////////////////////////// Model

  Model::Model()
    :
    theCurrentTime( 0.0 ),
    theLoggerBroker(),
    theRootSystemPtr(0),
    theSystemStepper(),
    theStepperMaker(),
    theSystemMaker(),
    theVariableMaker(),
    theProcessMaker(),
    theRunningFlag( false ),
    theDirtyBit( false )
  {
    theLoggerBroker.setModel( this );
    // initialize theRootSystem
    theRootSystemPtr = getSystemMaker().make( "System" );
    theRootSystemPtr->setModel( this );
    theRootSystemPtr->setID( "/" );
    theRootSystemPtr->setName( "The Root System" );
    // super system of the root system is itself.
    theRootSystemPtr->setSuperSystem( theRootSystemPtr );

   
    // initialize theSystemStepper
    theSystemStepper.setModel( this );
    theSystemStepper.setID( "___SYSTEM" );
    theScheduler.addEvent(
      StepperEvent( getCurrentTime()
                    + theSystemStepper.getStepInterval(),
                    &theSystemStepper ) );

    theLastStepper = &theSystemStepper;
  }

  Model::~Model()
  {
    delete theRootSystemPtr;
  }


  void Model::flushLoggers()
  {
    theLoggerBroker.flush();
  }


  PolymorphMap Model::getClassInfo( StringCref aClassType, StringCref aClassname, Integer forceReload )
  {
	const void* (*InfoPtrFunc)();
     
    if ( aClassType == "Stepper" )
      {
        InfoPtrFunc = getStepperMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
      }
    else
      {
        EntityType anEntityType( aClassType );
        if ( anEntityType.getType() == EntityType::VARIABLE )
	      {    
		    InfoPtrFunc = getVariableMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
	      }
        else if ( anEntityType.getType() == EntityType::PROCESS )
	      {
		    InfoPtrFunc = getProcessMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
	      }
        else if ( anEntityType.getType() == EntityType::SYSTEM )
	      {
		    InfoPtrFunc = getSystemMaker().getModule( aClassname, forceReload != 0 ).getInfoLoader();
	      }
        else 
	      {
		    THROW_EXCEPTION( InvalidEntityType,
			  			     "bad ClassType specified." );
	      }
      }
      
	return *(reinterpret_cast<const PolymorphMap*>( InfoPtrFunc() ) );
  }

  void Model::recordUninitializedStepper( StepperPtr aStepperPtr )
  {
    theDirtyBit = true;
    // Duplication is checked when the object is originally created, therefore
    // we know this cannot be found in the uninitializedVector.

    uninitializedSteppers.push_back( aStepperPtr );
  }

  void Model::recordUninitializedVariable( VariablePtr aVariablePtr )
  {
    theDirtyBit = true;
    // Duplication is checked when the object is originally created, therefore
    // we know this cannot be found in the uninitializedVector.

    uninitializedVariables.push_back( aVariablePtr );
  }

  void Model::recordUninitializedSystem( SystemPtr aSystemPtr )
  {
    theDirtyBit = true;
    // Duplication is checked when the object is originally created, therefore
    // we know this cannot be found in the uninitializedVector.

    uninitializedSystems.push_back( aSystemPtr );
  }

  void Model::recordUninitializedProcess( ProcessPtr aProcessPtr )
  {
    theDirtyBit = true;
    // Duplication is checked when the object is originally created, therefore
    // we know this cannot be found in the uninitializedVector.

    uninitializedProcesses.push_back( aProcessPtr );
  }

  void Model::staticInitialize()
  {
    std::cout << "Model::staticInitialize" << std::endl;

    SystemPtr aRootSystem( getRootSystem() );

    checkRootSystemSizeVariable();

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
    //    theScheduler.updateAllEvents( getCurrentTime() );

    for( EventIndex c( 0 ); c != theScheduler.getSize(); ++c )
      {
	theScheduler.getEvent(c).reschedule();
      }

    clearUninitialized();
  }
  

  void Model::createEntity( StringCref aClassname,
			    FullIDCref aFullID )
  {
    std::cout << "Model::createEntity(" << aFullID.getID() << ")" << std::endl;

    if( aFullID.getSystemPath().empty() )
      {
	THROW_EXCEPTION( BadSystemPath, "Empty SystemPath." );
      }

    SystemPtr aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );

    ProcessPtr   aProcessPtr( NULLPTR );
    SystemPtr    aSystemPtr ( NULLPTR );
    VariablePtr aVariablePtr( NULLPTR );

    switch( aFullID.getEntityType() )
      {
      case EntityType::VARIABLE:
	aVariablePtr = getVariableMaker().make( aClassname );
	aVariablePtr->setID( aFullID.getID() );
        aVariablePtr->setCreationTime( getCurrentTime() );
	aContainerSystemPtr->registerVariable( aVariablePtr );
        recordUninitializedVariable( aVariablePtr );
	break;
      case EntityType::PROCESS:
	aProcessPtr = getProcessMaker().make( aClassname );
	aProcessPtr->setID( aFullID.getID() );
	aContainerSystemPtr->registerProcess( aProcessPtr );
        recordUninitializedProcess( aProcessPtr );
	break;
      case EntityType::SYSTEM:
	aSystemPtr = getSystemMaker().make( aClassname );
	aSystemPtr->setID( aFullID.getID() );
	aSystemPtr->setModel( this );
	aContainerSystemPtr->registerSystem( aSystemPtr );
        recordUninitializedSystem( aSystemPtr );
	break;

      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );
      }

    return;
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
    StringCref     anID( aFullID.getID() );

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
	anEntity = aSystem->getProcess(  aFullID.getID() );
	break;
      case EntityType::SYSTEM:
	anEntity = aSystem->getSystem(   aFullID.getID() );
	break;
      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );
      }

    return anEntity;
  }


  StepperPtr Model::getStepper( StringCref anID ) const
  {
    StepperMapConstIterator i( theStepperMap.find( anID ) );

    if( i == theStepperMap.end() )
      {
	THROW_EXCEPTION( NotFound, 
			 "Stepper [" + anID + "] not found in this model." );
      }

    return (*i).second;
  }


  void Model::createStepper( StringCref aClassName, StringCref anID )
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
    if( aSystem->getStepper() == NULLPTR )
      {
	THROW_EXCEPTION( InitializationFailed,
			 "No stepper is connected with [" +
			 aSystem->getFullID().getString() + "]." );
      }

    for( SystemMapConstIterator i( aSystem->getSystemMap().begin() ) ;
	 i != aSystem->getSystemMap().end() ; ++i )
      {
	// check it recursively
	checkStepper( i->second );
      }
  }


  void Model::initializeSystems( SystemPtr const aSystem )
  {
    aSystem->initialize();

    for( SystemMapConstIterator i( aSystem->getSystemMap().begin() );
	 i != aSystem->getSystemMap().end() ; ++i )
      {
	// initialize recursively
	initializeSystems( i->second );
      }
  }

  void Model::checkRootSystemSizeVariable()
  {
    // Changed from 
    // void Model::checkSizeVariable( SystemCptr const aSystem )
    // to void Model::checkRootSystemSize()
    // in order to eliminate potential confusion.

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

  void Model::runningInitialize()
  {
    std::cout<< "Model::runningInitialize() is being called" << std::endl;

    // Everything has been created at this point.  We may also assume that all property values that have 
    for(SystemVector::iterator i = uninitializedSystems.begin();
        i != uninitializedSystems.end();
        ++i)
      {
        // This ensures that a stepper has been set (if one has not, the system
        // is given to it's supersystem's stepper).
        (*i)->configureStepper();
        (*i)->configureSizeVariable();
      }

    for(VariableVector::iterator i = uninitializedVariables.begin();
        i != uninitializedVariables.end();
        ++i)
      {
        (*i)->initialize();
      }

    for(ProcessVector::iterator i = uninitializedProcesses.begin();
        i != uninitializedProcesses.end();
        ++i)
      {
        if ( (*i)->getStepper() == NULLPTR )
          {
            (*i)->setStepper( (*i)->getSuperSystem()->getStepper() );
          }

        (*i)->initialize();
      }

    // Now do all the stepper initialization stuff.....
    // corresponds to Model::staticInitialize() lines 197-

    {
      // Here we must do what corresponds to initializing each of the steppers.
      // We can reinitialize *ALL* of them, but this is quite slow.  What is needed
      // here (desperately) is a method to tell which Steppers have been fucked with.

      FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );
      FOR_ALL_SECOND( StepperMap, theStepperMap, 
                      updateIntegratedVariableVector );

      theScheduler.updateEventDependency();
      for( EventIndex c( 0 ); c != theScheduler.getSize(); ++c )
        {
          theScheduler.getEvent(c).reschedule();
        }
    }
    
    clearUninitialized();
    return;
  }

  void Model::initialize()
  {
    if ( getRunningFlag() )
      {
        this->runningInitialize();
      }
    else
      {
        this->staticInitialize();
      }

    return;
  }

} // namespace libecs





/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
