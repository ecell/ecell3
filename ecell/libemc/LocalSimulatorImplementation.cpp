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

#include <algorithm>

#include "libecs/libecs.hpp"
#include "libecs/Message.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/LoggerBroker.hpp"

#include "LocalSimulatorImplementation.hpp"

namespace libemc
{

  using namespace libecs;

  LocalSimulatorImplementation::LocalSimulatorImplementation()
    :
    theModel( *new Model ),
    theRunningFlag( false ),
    thePendingEventChecker( defaultPendingEventChecker ),
    theEventHandler( NULL )
  {
    ; // do nothing
  }

  LocalSimulatorImplementation::~LocalSimulatorImplementation()
  {
    delete &theModel;
  }

  void LocalSimulatorImplementation::
  createStepper( libecs::StringCref          aClassname, 
		 libecs::StringCref          anId,
		 libecs::UVariableVectorCref aData )
  {
    getModel().createStepper( aClassname, anId, aData );
  }

  void LocalSimulatorImplementation::createEntity( StringCref aClassname,
						   StringCref aFullIDString,
						   StringCref aName )
  {
    getModel().createEntity( aClassname, FullID( aFullIDString ), aName );
  }
    
  void LocalSimulatorImplementation::
  setProperty( StringCref aFullPNString,
	       UVariableVectorCref aData )
  {
    FullPN aFullPN( aFullPNString );
    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    // this new does not cause memory leak since Message will get it as a RCPtr
    anEntityPtr->setMessage( Message( aFullPN.getPropertyName(), 
				      new UVariableVector( aData ) ) );
  }


  const UVariableVectorRCPtr
  LocalSimulatorImplementation::getProperty( StringCref aFullPNString )
  {
    FullPN aFullPN( aFullPNString );
    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );
    return anEntityPtr->getMessage( aFullPN.getPropertyName() ).getBody();
  }

  void LocalSimulatorImplementation::step()
  {
    getModel().step();  
  }

  void LocalSimulatorImplementation::initialize()
  {
    getModel().initialize();
  }

  const libecs::Real LocalSimulatorImplementation::getCurrentTime()
  {
    return getModel().getCurrentTime();
  }

  LoggerPtr LocalSimulatorImplementation::
  getLogger( libecs::StringCref aFullPNString )
  {
    FullPN aFullPN( aFullPNString );

    return getModel().getLoggerBroker().getLogger( aFullPN );
  }

  StringVectorRCPtr LocalSimulatorImplementation::getLoggerList()
  {
    StringVectorRCPtr aLoggerListPtr( new StringVector );
    aLoggerListPtr->
      reserve( getModel().getLoggerBroker().getLoggerMap().size() );

    LoggerBroker::LoggerMapCref 
      aLoggerMap( getModel().getLoggerBroker().getLoggerMap() );

    for( LoggerBroker::LoggerMapConstIterator i( aLoggerMap.begin() );
	 i != aLoggerMap.end(); ++i )
      {
	FullPNCref aFullPN( i->first );
	aLoggerListPtr->push_back( aFullPN.getString() );
      }

    return aLoggerListPtr;
  }


  void LocalSimulatorImplementation::run()
  {
    if( ! ( thePendingEventChecker != NULLPTR && theEventHandler != NULLPTR ) )
      {
	THROW_EXCEPTION( libecs::Exception,
			 "Both EventChecker and EventHandler must be "
			 "set before run without duration." ) ;
      }

    theRunningFlag = true;

    do
      {

	for( int i( 0 ) ; i < 20 ; i++ )
	  {
	    step();
	  }

	while( (*thePendingEventChecker)() )
	  {
	    (*theEventHandler)();
	  }

      }	while( theRunningFlag );

  }

  void LocalSimulatorImplementation::run( libecs::Real aDuration )
  {
    if( thePendingEventChecker != NULLPTR && theEventHandler != NULLPTR )
      {
	runWithEvent( aDuration );
      }
    else
      {
	runWithoutEvent( aDuration );
      }
  }

  void LocalSimulatorImplementation::runWithEvent( libecs::Real aDuration )
  {
    theRunningFlag = true;

    libecs::Real aStopTime( getModel().getCurrentTime() + aDuration );

    do
      {
	for( int i( 0 ) ; i < 20 ; i++ )
	  {
	    if( getModel().getCurrentTime() >= aStopTime )
	      {
		theRunningFlag = false;
		break;
	      }

	    step();
	  }

	while( (*thePendingEventChecker)() )
	  {
	    (*theEventHandler)();
	  }

      }	while( theRunningFlag );

  }

  void LocalSimulatorImplementation::runWithoutEvent( libecs::Real aDuration )
  {
    theRunningFlag = true;

    libecs::Real aStopTime( getModel().getCurrentTime() + aDuration );

    do
      {
	for( int i( 0 ) ; i < 20 ; i++ )
	  {
	    if( getModel().getCurrentTime() >= aStopTime )
	      {
		theRunningFlag = false;
		break;
	      }

	    step();
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


