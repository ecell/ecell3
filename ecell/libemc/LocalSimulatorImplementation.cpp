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
#include "libecs/Stepper.hpp"
#include "libecs/LoggerBroker.hpp"

#include "EmcLogger.hpp"

#include "LocalSimulatorImplementation.hpp"

namespace libemc
{

  using namespace libecs;

  LocalSimulatorImplementation::LocalSimulatorImplementation()
    :
    theModel( *new Model ),
    theRunningFlag( false ),
    thePendingEventChecker( NULLPTR ),
    theEventHandler( NULLPTR )
  {
    clearPendingEventChecker();
  }

  LocalSimulatorImplementation::~LocalSimulatorImplementation()
  {
    delete &theModel;
  }

  void LocalSimulatorImplementation::
  createStepper( libecs::StringCref          aClassname, 
		 libecs::StringCref          anId )
  {
    getModel().createStepper( aClassname, anId );
  }

  const libecs::Polymorph LocalSimulatorImplementation::getStepperList()
  {
    StepperMapCref aStepperMap( getModel().getStepperMap() );

    PolymorphVector aPolymorphVector; 
    aPolymorphVector.reserve( aStepperMap.size() );
    
    for( StepperMapConstIterator i( aStepperMap.begin() );
	 i != aStepperMap.end(); ++i )
      {
	aPolymorphVector.push_back( String( (*i).first ) );
      }

    return aPolymorphVector;
  }

  void LocalSimulatorImplementation::
  setStepperProperty( libecs::StringCref          aStepperID,
		      libecs::StringCref          aPropertyName,
		      libecs::PolymorphCref aValue )
  {
    StepperPtr aStepperPtr( getModel().getStepper( aStepperID ) );
    
    aStepperPtr->setProperty( aPropertyName, aValue );
  }

  const libecs::Polymorph
  LocalSimulatorImplementation::
  getStepperProperty( libecs::StringCref aStepperID,
		      libecs::StringCref aPropertyName )
  {
    StepperPtr aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->getProperty( aPropertyName );
  }


  void LocalSimulatorImplementation::createEntity( StringCref aClassname,
						   StringCref aFullIDString,
						   StringCref aName )
  {
    getModel().createEntity( aClassname, FullID( aFullIDString ), aName );
  }

  bool LocalSimulatorImplementation::
  isEntityExist( libecs::StringCref aFullIDString )
  {
    try
      {
	getModel().getEntity( FullID( aFullIDString ) );
      }
    catch( const libecs::NotFound& )
      {
	return false;
      }

    return true;
  }


  void LocalSimulatorImplementation::
  setProperty( StringCref aFullPNString, PolymorphCref aValue )
  {
    FullPN aFullPN( aFullPNString );
    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    anEntityPtr->setProperty( aFullPN.getPropertyName(), aValue );
  }


  const Polymorph
  LocalSimulatorImplementation::getProperty( StringCref aFullPNString )
  {
    FullPN aFullPN( aFullPNString );
    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    return anEntityPtr->getProperty( aFullPN.getPropertyName() );
  }

  void LocalSimulatorImplementation::step()
  {
    getModel().initialize();  
    getModel().step();  
    getModel().flushLogger();
  }

  void LocalSimulatorImplementation::initialize()
  {
    getModel().initialize();
  }

  const libecs::Real LocalSimulatorImplementation::getCurrentTime()
  {
    return getModel().getCurrentTime();
  }

  EmcLogger LocalSimulatorImplementation::
  getLogger( libecs::StringCref aFullPNString )
  {
    FullPN aFullPN( aFullPNString );
    LoggerPtr aLoggerPtr( getModel().getLoggerBroker().getLogger( aFullPN ) );

    return EmcLogger( aLoggerPtr );
  }

  const Polymorph LocalSimulatorImplementation::getLoggerList()
  {
    PolymorphVector aLoggerList;
    aLoggerList.reserve( getModel().getLoggerBroker().getLoggerMap().size() );

    LoggerBroker::LoggerMapCref 
      aLoggerMap( getModel().getLoggerBroker().getLoggerMap() );

    for( LoggerBroker::LoggerMapConstIterator i( aLoggerMap.begin() );
	 i != aLoggerMap.end(); ++i )
      {
	FullPNCref aFullPN( (*i).first );
	aLoggerList.push_back( aFullPN.getString() );
      }

    return aLoggerList;
  }


  void LocalSimulatorImplementation::run()
  {
    getModel().initialize();

    if( ! ( thePendingEventChecker != NULLPTR && theEventHandler != NULLPTR ) )
      {
	THROW_EXCEPTION( libecs::Exception,
			 "Both EventChecker and EventHandler must be "
			 "set before run without duration." ) ;
      }

    theRunningFlag = true;

    do
      {
	unsigned int i( 20 );
	do 
	  {
	    getModel().step();

	    --i;
	  } while( i != 0 );

	while( (*thePendingEventChecker)() )
	  {
	    (*theEventHandler)();
	  }

      }	while( theRunningFlag );

    getModel().flushLogger();
  }

  void LocalSimulatorImplementation::run( libecs::Real aDuration )
  {
    getModel().initialize();

    if( thePendingEventChecker != NULLPTR && theEventHandler != NULLPTR )
      {
	runWithEvent( aDuration );
      }
    else
      {
	runWithoutEvent( aDuration );
      }

    getModel().flushLogger();
  }

  void LocalSimulatorImplementation::runWithEvent( libecs::Real aDuration )
  {
    theRunningFlag = true;

    const libecs::Real aStopTime( getModel().getCurrentTime() + aDuration );

    do
      {
	unsigned int i( 20 );
	do
	  {
	    if( getModel().getCurrentTime() > aStopTime )
	      {
		theRunningFlag = false;
		break;
	      }
	    
	    getModel().step();

	    --i;
	  } while( i != 0 );


	while( (*thePendingEventChecker)() )
	  {
	    (*theEventHandler)();
	  }

      }	while( theRunningFlag );

  }

  void LocalSimulatorImplementation::runWithoutEvent( libecs::Real aDuration )
  {
    theRunningFlag = true;

    const libecs::Real aStopTime( getModel().getCurrentTime() + aDuration );

    do
      {
	if( getModel().getCurrentTime() > aStopTime )
	  {
	    theRunningFlag = false;
	    return;  // the only exit
	  }

	getModel().step();

      }	while( 1 );

  }

  void LocalSimulatorImplementation::stop()
  {
    theRunningFlag = false;
  }

  void LocalSimulatorImplementation::
  setPendingEventChecker( PendingEventCheckerPtr aPendingEventChecker )
  {
    delete thePendingEventChecker;
    thePendingEventChecker = aPendingEventChecker;
  }

  void LocalSimulatorImplementation::
  setEventHandler( EventHandlerPtr anEventHandler )
  {
    delete theEventHandler;
    theEventHandler = anEventHandler;
  }

  void LocalSimulatorImplementation::clearPendingEventChecker()
  {
    setPendingEventChecker( new DefaultPendingEventChecker() );
  }


} // namespace libemc,


