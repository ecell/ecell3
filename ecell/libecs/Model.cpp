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
#include "SubstanceMaker.hpp"
#include "ReactorMaker.hpp"
#include "SystemMaker.hpp"
#include "AccumulatorMaker.hpp"
#include "LoggerBroker.hpp"
#include "Stepper.hpp"

#include "Model.hpp"


namespace libecs
{

  ////////////////////////// Model

  Model::Model()
    : 
    theCurrentTime( 0.0 ),
    theStepperMap(),
    theScheduleQueue(),
    theRootSystemPtr( NULL ),
    theLoggerBroker(     *new LoggerBroker( *this ) ),
    theStepperMaker(     *new StepperMaker          ),
    theSystemMaker(      *new SystemMaker           ),
    theSubstanceMaker(   *new SubstanceMaker        ),
    theReactorMaker(     *new ReactorMaker          ),
    theAccumulatorMaker( *new AccumulatorMaker      )
  {
    theRootSystemPtr = getSystemMaker().make( "System" );
    theRootSystemPtr->setID( "/" );
    theRootSystemPtr->setName( "The Root System" );
    theRootSystemPtr->setModel( this );
    theRootSystemPtr->setSuperSystem( theRootSystemPtr );
  }

  Model::~Model()
  {
    delete theRootSystemPtr;
    delete &theAccumulatorMaker;
    delete &theReactorMaker;
    delete &theSubstanceMaker;
    delete &theSystemMaker;
    delete &theStepperMaker;
    delete &theLoggerBroker;
  }


  void Model::flushLogger()
  {
    theLoggerBroker.flush();
  }


  void Model::createEntity( StringCref aClassname,
			    FullIDCref aFullID,
			    StringCref aName )
  {
    if( aFullID.getSystemPath().empty() )
      {
	THROW_EXCEPTION( BadSystemPath, "Empty SystemPath." );
      }

    SystemPtr aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );

    ReactorPtr   aReactorPtr  ( NULLPTR );
    SystemPtr    aSystemPtr   ( NULLPTR );
    SubstancePtr aSubstancePtr( NULLPTR );

    switch( aFullID.getEntityType() )
      {
      case EntityType::SUBSTANCE:
	aSubstancePtr = getSubstanceMaker().make( aClassname );
	aSubstancePtr->setID( aFullID.getID() );
	aSubstancePtr->setName( aName );
	aContainerSystemPtr->registerSubstance( aSubstancePtr );
	break;
      case EntityType::REACTOR:
	aReactorPtr = getReactorMaker().make( aClassname );
	aReactorPtr->setID( aFullID.getID() );
	aReactorPtr->setName( aName );
	aContainerSystemPtr->registerReactor( aReactorPtr );
	break;
      case EntityType::SYSTEM:
	aSystemPtr = getSystemMaker().make( aClassname );
	aSystemPtr->setID( aFullID.getID() );
	aSystemPtr->setName( aName );
	aContainerSystemPtr->registerSystem( aSystemPtr );
	break;

      default:
	THROW_EXCEPTION( InvalidEntityType,
			 "bad EntityType specified." );

      }

  }



  SystemPtr Model::getSystem( SystemPathCref aSystemPath )
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


  EntityPtr Model::getEntity( FullIDCref aFullID )
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
      case EntityType::SUBSTANCE:
	anEntity = aSystem->getSubstance( aFullID.getID() );
	break;
      case EntityType::REACTOR:
	anEntity = aSystem->getReactor(   aFullID.getID() );
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


  StepperPtr Model::getStepper( StringCref anID )
  {

    StepperMapIterator i( theStepperMap.find( anID ) );

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
    aStepper->setID( anID );

    theStepperMap.insert( std::make_pair( anID, aStepper ) );
    theScheduleQueue.push( Event( getCurrentTime(), aStepper ) );
  }


  void Model::resetScheduleQueue()
  {
    //FIXME: slow! :  no theScheduleQueue.clear() ?
    while( ! theScheduleQueue.empty() )
      {
	theScheduleQueue.pop();
      }

    for( StepperMapConstIterator i( theStepperMap.begin() );
	 i != theStepperMap.end(); i++ )
      {
	theScheduleQueue.push( Event( getCurrentTime(), (*i).second ) );
      }
  }


  void Model::checkStepper( SystemCptr aSystem)
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

  void Model::initialize()
  {
    checkStepper( getRootSystem() );

    FOR_ALL_SECOND( StepperMap, theStepperMap, initialize );

    theCurrentTime = ( theScheduleQueue.top() ).first;
  }


  void Model::step()
  {
    EventCref aTopEvent( theScheduleQueue.top() );

    StepperPtr aStepper( aTopEvent.second );

    // three-phase progression of the step
    // 1. sync:  synchronize with proxies of the PropertySlots
    aStepper->sync();
    // 2. step:  do the computation
    aStepper->step();
    // 3. push:  re-sync with the proxies, and push new values to Loggers
    //           this need to be placed here after the event re-scheduling
    //           so that Loggers get the new time
    aStepper->push();

    // schedule a new event
    theScheduleQueue.changeTopKey( Event( aStepper->getCurrentTime(),
					  aStepper ) );
    // update theCurrentTime, which is scheduled time of the Event on the top
    theCurrentTime = ( theScheduleQueue.top() ).first;

  }



} // namespace libecs





/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
