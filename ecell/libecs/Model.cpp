//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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

  void Model::createEntity( StringCref aClassname,
			    FullIDCref aFullID,
			    StringCref aName )
  {
    if( aFullID.getSystemPath().empty() )
      {
	throw BadSystemPath( __PRETTY_FUNCTION__, "Empty SystemPath." );
      }

    SystemPtr aContainerSystemPtr( getSystem( aFullID.getSystemPath() ) );

    ReactorPtr   aReactorPtr  ( NULLPTR );
    SystemPtr    aSystemPtr   ( NULLPTR );
    SubstancePtr aSubstancePtr( NULLPTR );

    switch( aFullID.getPrimitiveType() )
      {
      case PrimitiveType::SUBSTANCE:
	aSubstancePtr = getSubstanceMaker().make( aClassname );
	aSubstancePtr->setID( aFullID.getID() );
	aSubstancePtr->setName( aName );
	aContainerSystemPtr->registerSubstance( aSubstancePtr );
	break;
      case PrimitiveType::REACTOR:
	aReactorPtr = getReactorMaker().make( aClassname );
	aReactorPtr->setID( aFullID.getID() );
	aReactorPtr->setName( aName );
	aContainerSystemPtr->registerReactor( aReactorPtr );
	break;
      case PrimitiveType::SYSTEM:
	aSystemPtr = getSystemMaker().make( aClassname );
	aSystemPtr->setID( aFullID.getID() );
	aSystemPtr->setName( aName );
	aContainerSystemPtr->registerSystem( aSystemPtr );
	break;

      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );

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
	throw BadSystemPath( __PRETTY_FUNCTION__, "[" + 
			     aSystemPath.getString() +
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
	    throw BadID( __PRETTY_FUNCTION__, "[" + aFullID.getString()
			 + "] is an invalid FullID" );
	  }
      }

    SystemPtr aSystem ( getSystem( aSystemPath ) );

    switch( aFullID.getPrimitiveType() )
      {
      case PrimitiveType::SUBSTANCE:
	anEntity = aSystem->getSubstance( aFullID.getID() );
	break;
      case PrimitiveType::REACTOR:
	anEntity = aSystem->getReactor(   aFullID.getID() );
	break;
      case PrimitiveType::SYSTEM:
	anEntity = aSystem->getSystem(    aFullID.getID() );
	break;
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );
      }

    return anEntity;
  }


  StepperPtr Model::getStepper( StringCref anID )
  {
    StepperMapIterator i( theStepperMap.find( anID ) );
    if( i == theStepperMap.end() )
      {
	throw NotFound( __PRETTY_FUNCTION__, "Stepper [" + anID + 
			"] not found in this model." );
      }

    return (*i).second;
  }


  void Model::createStepper( StringCref aClassName,
			     StringCref anID,
			     UVariableVectorCref data )
  {
    StepperPtr aStepper( getStepperMaker().make( aClassName ) );
    aStepper->setName( anID );

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
	throw InitializationFailed( __PRETTY_FUNCTION__, 
				    "No stepper is connected with System [" +
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


    FOR_ALL_SECOND( StepperMap, theStepperMap, 
		    initialize );

    theCurrentTime = ( theScheduleQueue.top() ).first;
  }


  void Model::step()
  {
    EventCref aTopEvent( theScheduleQueue.top() );

    StepperPtr aStepper( aTopEvent.second );

    // three-phase progression of the step
    // 1. sync:  synchronize with proxies of the PropertySlots
    aStepper->sync();
    // 2. step:  do the computation, returning a length of the time progression
    const Real aStepSize( aStepper->step() );
    // 3. push:  re-sync with the proxies, and push new values to Loggers
    aStepper->push();


    // the time must be memorized before the Event is deleted by the pop
    const Real aTopTime( aTopEvent.first );

    // equivalent to these two lines.
    // theScheduleQueue.pop();
    // theScheduleQueue.push( Event( aTopTime + aStepSize, aStepper ) );
    theScheduleQueue.changeTopKey( Event( aTopTime + aStepSize, aStepper ) );

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
