//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include <iostream>

#include "Util.hpp"
#include "StepperMaker.hpp"
#include "VariableMaker.hpp"
#include "ProcessMaker.hpp"
#include "SystemMaker.hpp"
//#include "AccumulatorMaker.hpp"
#include "LoggerBroker.hpp"
#include "Stepper.hpp"

#include "Model.hpp"


namespace libecs
{

  ////////////////////////// Model

  Model::Model()
    : 
    theRootSystemPtr( NULL ),
    theLoggerBroker(     *new LoggerBroker( *this ) ),
    theStepperMaker(     *new StepperMaker          ),
    theSystemMaker(      *new SystemMaker           ),
    theVariableMaker(   *new VariableMaker        ),
    theProcessMaker(     *new ProcessMaker          )
    //    ,theAccumulatorMaker( *new AccumulatorMaker      )
  {
    theRootSystemPtr = getSystemMaker().make( "System" );
    theRootSystemPtr->setModel( this );
    theRootSystemPtr->setID( "/" );
    theRootSystemPtr->setName( "The Root System" );
    // super system of the root system is itself.
    theRootSystemPtr->setSuperSystem( theRootSystemPtr );
  }

  Model::~Model()
  {
    delete theRootSystemPtr;
    //    delete &theAccumulatorMaker;
    delete &theProcessMaker;
    delete &theVariableMaker;
    delete &theSystemMaker;
    delete &theStepperMaker;
    delete &theLoggerBroker;
  }


  void Model::flushLogger()
  {
    theLoggerBroker.flush();
  }


  void Model::createEntity( StringCref aClassname,
			    FullIDCref aFullID )
  {
    if( aFullID.getSystemPath().empty() )
      {
	THROW_EXCEPTION( BadSystemPath, "Empty SystemPath." );
      }

    SystemPtr aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );

    ProcessPtr   aProcessPtr  ( NULLPTR );
    SystemPtr    aSystemPtr   ( NULLPTR );
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
    SystemPtr aSystem( getRootSystem() );
    SystemPath aSystemPathCopy( aSystemPath );


    // 1. "" (empty) means Model itself, which is invalid for this method.
    // 2. Not absolute is invalid.
    // (not absolute implies not empty.)
    if( aSystemPathCopy.isAbsolute() && ! aSystemPathCopy.empty() )
      {
	aSystemPathCopy.pop_front();
      }
    else
      {
	THROW_EXCEPTION( BadSystemPath, 
			 "[" + aSystemPath.getString() +
			 "] is not an absolute SystemPath." );
      }

    // root system
    if( aSystemPathCopy.size() == 0 )
      {
	return aSystem;
      }

    // looping is faster than recursive search
    while( ! aSystemPathCopy.empty() )
      {
	aSystem = aSystem->getSystem( aSystemPathCopy.front() );
	aSystemPathCopy.pop_front();
      }


    return aSystem;  
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
	anEntity = aSystem->getProcess(   aFullID.getID() );
	break;
      case EntityType::SYSTEM:
	anEntity = aSystem->getSystem(    aFullID.getID() );
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

  void Model::initialize()
  {
    initializeSystems( getRootSystem() );

    checkStepper( getRootSystem() );

    // initialization of Stepper needs two stages:
    // (1) initialize
    // (2) construct stepper dependency graph
    FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );
    FOR_ALL_SECOND( StepperMap, theStepperMap, updateDependentStepperVector );

  }


} // namespace libecs





/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
