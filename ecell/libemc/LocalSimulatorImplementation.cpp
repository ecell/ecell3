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

#include <algorithm>

#include "libecs/libecs.hpp"
#include "libecs/Message.hpp"
#include "libecs/RootSystem.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/LoggerBroker.hpp"

#include "LocalSimulatorImplementation.hpp"

namespace libemc
{

  using namespace libecs;

  LocalSimulatorImplementation::LocalSimulatorImplementation()
    :
    theRootSystem( *new RootSystem ),
    theLoggerBroker( *new LoggerBroker( theRootSystem ) ),
    theRunningFlag( false ),
    thePendingEventChecker( defaultPendingEventChecker ),
    theEventHandler( NULL )
  {
    ; // do nothing
  }

  LocalSimulatorImplementation::~LocalSimulatorImplementation()
  {
    delete &theRootSystem;
    delete &theLoggerBroker;
  }

  void LocalSimulatorImplementation::createEntity( StringCref    classname, 
						   PrimitiveType type,
						   StringCref    systempath,
						   StringCref    id,
						   StringCref    name )
  {
    getRootSystem().createEntity( classname, 
				  FullID( type, systempath, id ),
				  name );
  }
    
  void LocalSimulatorImplementation::setProperty( PrimitiveType type,
						  StringCref    systempath,
						  StringCref    id,
						  StringCref    property,
						  UVariableVectorCref data )
  {
    EntityPtr anEntityPtr( getRootSystem().getEntity( FullID( type, 
							      systempath, 
							      id ) ) );
    // this new does not cause memory leak since Message will get it as a RCPtr
    anEntityPtr->setMessage( Message( property, 
				      new UVariableVector( data ) ) );
  }


  const UVariableVectorRCPtr
  LocalSimulatorImplementation::getProperty( PrimitiveType type,
					     StringCref    systempath,
					     StringCref    id,
					     StringCref    propertyname )
  {
    EntityPtr anEntityPtr( getRootSystem().getEntity( FullID( type, 
							      systempath, 
							      id ) ) );
    return anEntityPtr->getMessage( propertyname ).getBody();
  }


  void LocalSimulatorImplementation::step()
  {
    theRootSystem.getStepperLeader().step();  
  }

  void LocalSimulatorImplementation::initialize()
  {
    theRootSystem.initialize();
  }

  LoggerPtr LocalSimulatorImplementation::
  getLogger(libecs::PrimitiveType type,
	    libecs::StringCref    systempath,
	    libecs::StringCref    id,
	    libecs::StringCref    propertyname )
  {
    return theLoggerBroker.getLogger( FullPN( type,
					      SystemPath(systempath),
					      id,
					      propertyname ) );
  }

  StringVectorRCPtr LocalSimulatorImplementation::getLoggerList()
  {
    StringVectorRCPtr aLoggerListPtr( new StringVector );
    aLoggerListPtr->reserve( theLoggerBroker.getLoggerMap().size() );

    LoggerBroker::LoggerMapCref aLoggerMap( theLoggerBroker.getLoggerMap() );

    for( LoggerBroker::LoggerMapConstIterator i( aLoggerMap.begin() );
	 i != aLoggerMap.end(); ++i )
      {
	aLoggerListPtr->push_back( i->first );
      }

    return aLoggerListPtr;
  }


  void LocalSimulatorImplementation::run()
  {
    assert( thePendingEventChecker != NULLPTR );
    assert( theEventHandler != NULLPTR );

    theRunningFlag = true;

    do
      {

	for( int i = 0 ; i < 20 ; i++ )
	  {
	    step();
	  }

	while( (*thePendingEventChecker)() )
        {
	  (*theEventHandler)();

	  if( !theRunningFlag )
	    break;
	}

      }	while( theRunningFlag );

  }

  void LocalSimulatorImplementation::run( libecs::Real aDuration )
  {
    assert( thePendingEventChecker != NULLPTR );
    assert( theEventHandler != NULLPTR );

    theRunningFlag = true;

    libecs::Real aStopTime( theRootSystem.getStepperLeader().getCurrentTime() + aDuration );

    do
      {
	for( int i = 0 ; i < 20 ; i++ )
	  {

	    if( theRootSystem.getStepperLeader().getCurrentTime() 
		>= aStopTime )
	      {
		theRunningFlag = false;
		break;
	      }

	    step();
	  }

	while( (*thePendingEventChecker)() )
        {
	  (*theEventHandler)();

	  if( !theRunningFlag )
	    break;
	}

      }	while( theRunningFlag );

  }

  void LocalSimulatorImplementation::stop()
  {
    theRunningFlag = false;
  }

  void LocalSimulatorImplementation::
  setPendingEventChecker( PendingEventCheckerFuncPtr aPendingEventChecker )
  {
    thePendingEventChecker = aPendingEventChecker;
  }

  void LocalSimulatorImplementation::
  setEventHandler( EventHandlerFuncPtr anEventHandler )
  {
    theEventHandler = anEventHandler;
  }

  void LocalSimulatorImplementation::clearPendingEventChecker()
  {
    thePendingEventChecker = defaultPendingEventChecker;
  }

  bool LocalSimulatorImplementation::defaultPendingEventChecker()
  {
    return false;
  }



} // namespace libemc,


