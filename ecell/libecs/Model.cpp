//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2004 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

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

  ////////////////////////// Model

  Model::Model()
    : 
    theLoggerBroker(     *new LoggerBroker( *this ) ),
    theRootSystemPtr(    NULL ),
    theSystemStepperPtr(  new SystemStepper ),
    theStepperMaker(     *new StepperMaker          ),
    theSystemMaker(      *new SystemMaker           ),
    theVariableMaker(    *new VariableMaker        ),
    theProcessMaker(     *new ProcessMaker          )
  {
    // initialize theRootSystem
    theRootSystemPtr = getSystemMaker().make( "System" );
    theRootSystemPtr->setModel( this );
    theRootSystemPtr->setID( "/" );
    theRootSystemPtr->setName( "The Root System" );
    // super system of the root system is itself.
    theRootSystemPtr->setSuperSystem( theRootSystemPtr );

   
    // initialize theSystemStepper
    theSystemStepperPtr->setModel( this );
    theSystemStepperPtr->setID( "___SYSTEM" );
    // don't insert this to the stepper map.
    theScheduler.registerStepper( theSystemStepperPtr );
  }

  Model::~Model()
  {
    delete theRootSystemPtr;
    delete &theProcessMaker;
    delete &theVariableMaker;
    delete &theSystemMaker;
    delete &theStepperMaker;
    delete theSystemStepperPtr;
    delete &theLoggerBroker;
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


  void Model::createEntity( StringCref aClassname,
			    FullIDCref aFullID )
  {
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
	aContainerSystemPtr->registerVariable( aVariablePtr );
	break;
      case EntityType::PROCESS:
	aProcessPtr = getProcessMaker().make( aClassname );
	aProcessPtr->setID( aFullID.getID() );
	aContainerSystemPtr->registerProcess( aProcessPtr );

	break;
      case EntityType::SYSTEM:
		aSystemPtr = getSystemMaker().make( aClassname );
		aSystemPtr->setID( aFullID.getID() );
		aSystemPtr->setModel( this );
		aContainerSystemPtr->registerSystem( aSystemPtr );
	break;

      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );

      }
	

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

    theScheduler.registerStepper( aStepper );
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

  void Model::checkSizeVariable( SystemCptr const aSystem )
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
    //     - construct stepper dependency graph and
    //     - fill theIntegratedVariableVector.

    /*
    const Real aCurrentTime( getCurrentTime() );
    for( StepperMapConstIterator i( theStepperMap.begin() );
    	 i != theStepperMap.end(); ++i )
      {
    	(*i).second->integrate( aCurrentTime );
      }
    */
    
    FOR_ALL_SECOND( StepperMap, theStepperMap, initializeProcesses );
    FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );
    FOR_ALL_SECOND( StepperMap, theStepperMap, updateDependentStepperVector );
    FOR_ALL_SECOND( StepperMap, theStepperMap, 
		    updateIntegratedVariableVector );

    theSystemStepperPtr->initialize();

  }


} // namespace libecs





/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
