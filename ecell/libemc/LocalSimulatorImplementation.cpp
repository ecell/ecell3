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

#include <algorithm>

#include "libecs/libecs.hpp"
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
    theEventChecker( NULLPTR ),
    theEventHandler( NULLPTR )
  {
    clearEventChecker();
  }

  LocalSimulatorImplementation::~LocalSimulatorImplementation()
  {
    delete &theModel;
  }

  inline LoggerPtr LocalSimulatorImplementation::
  getLogger( libecs::StringCref aFullPNString ) const
  {
    return getModel().getLoggerBroker().getLogger( aFullPNString );
  }

  void LocalSimulatorImplementation::
  createStepper( libecs::StringCref          aClassname, 
		 libecs::StringCref          anId )
  {
    getModel().createStepper( aClassname, anId );
  }

  void LocalSimulatorImplementation::deleteStepper( libecs::StringCref anID )
  {
    std::cerr << "deleteStepper() method is not supported yet." << std::endl;
  }

  const libecs::Polymorph LocalSimulatorImplementation::getStepperList() const
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


  const libecs::Polymorph LocalSimulatorImplementation::
  getStepperPropertyList( StringCref aStepperID ) const
  {
    StepperPtr aStepperPtr( getModel().getStepper( aStepperID ) );
    
    return aStepperPtr->getPropertyList();
  }
  
  const libecs::Polymorph LocalSimulatorImplementation::
  getStepperPropertyAttributes( libecs::StringCref aStepperID, 
				libecs::StringCref aPropertyName ) const
  {
    StepperPtr aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->getPropertyAttributes( aPropertyName );
  }
  

  void LocalSimulatorImplementation::
  setStepperProperty( libecs::StringCref          aStepperID,
		      libecs::StringCref          aPropertyName,
		      libecs::PolymorphCref aValue )
  {
    StepperPtr aStepperPtr( getModel().getStepper( aStepperID ) );
    
    aStepperPtr->setProperty( aPropertyName, aValue );
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  getStepperProperty( libecs::StringCref aStepperID,
		      libecs::StringCref aPropertyName ) const
  {
    StepperCptr aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->getProperty( aPropertyName );
  }

  const libecs::String LocalSimulatorImplementation::
  getStepperClassName( libecs::StringCref aStepperID ) const
  {
    StepperCptr aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->getClassNameString();
  }


  void LocalSimulatorImplementation::createEntity( StringCref aClassname,
						   StringCref aFullIDString )
  {
    getModel().createEntity( aClassname, FullID( aFullIDString ) );
  }

  void LocalSimulatorImplementation::deleteEntity( StringCref aFullIDString )
  {
    std::cerr << "deleteEntity() method is not supported yet." << std::endl;
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  getEntityList( libecs::Int anEntityTypeNumber,
		 libecs::StringCref aSystemPathString ) const
  {
    const libecs::EntityType anEntityType( anEntityTypeNumber );
    const libecs::SystemPath aSystemPath( aSystemPathString );
    const libecs::SystemPtr aSystemPtr( getModel().getSystem( aSystemPath ) );

    switch( anEntityType )
      {
      case libecs::EntityType::VARIABLE:
	return aSystemPtr->getVariableList();
      case libecs::EntityType::PROCESS:
	return aSystemPtr->getProcessList();
      case libecs::EntityType::SYSTEM:
	return aSystemPtr->getSystemList();
      }

    NEVER_GET_HERE;
  }


  const libecs::Polymorph LocalSimulatorImplementation::
  getEntityPropertyList( libecs::StringCref aFullIDString ) const
  {
    FullID aFullID( aFullIDString );
    EntityCptr anEntityPtr( getModel().getEntity( aFullID ) );

    return anEntityPtr->getPropertyList();
  }


  const bool LocalSimulatorImplementation::
  isEntityExist( libecs::StringCref aFullIDString ) const
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
  setEntityProperty( StringCref aFullPNString, PolymorphCref aValue )
  {
    FullPN aFullPN( aFullPNString );
    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    anEntityPtr->setProperty( aFullPN.getPropertyName(), aValue );
  }


  const Polymorph LocalSimulatorImplementation::
  getEntityProperty( StringCref aFullPNString ) const
  {
    FullPN aFullPN( aFullPNString );
    EntityCptr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    return anEntityPtr->getProperty( aFullPN.getPropertyName() );
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  getEntityPropertyAttributes( libecs::StringCref aFullPNString ) const
  {
    FullPN aFullPN( aFullPNString );
    EntityCptr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    return anEntityPtr->getPropertyAttributes( aFullPN.getPropertyName() );
  }

  const libecs::String LocalSimulatorImplementation::
  getEntityClassName( libecs::StringCref aFullIDString ) const
  {
    FullID aFullID( aFullIDString );
    EntityCptr anEntityPtr( getModel().getEntity( aFullID ) );

    return anEntityPtr->getClassNameString();
  }


  void LocalSimulatorImplementation::
  createLogger( libecs::StringCref aFullPNString )
  {
    FullPN aFullPN( aFullPNString );
    getModel().getLoggerBroker().createLogger( aFullPN );
  }

  const Polymorph LocalSimulatorImplementation::getLoggerList() const
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


  const libecs::DataPointVectorRCPtr LocalSimulatorImplementation::
  getLoggerData( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getData();
  }

  const libecs::DataPointVectorRCPtr LocalSimulatorImplementation::
  getLoggerData( libecs::StringCref aFullPNString, 
		 libecs::RealCref aStartTime, 
		 libecs::RealCref anEndTime ) const
  {
    return getLogger( aFullPNString )->getData( aStartTime, anEndTime );
  }


  const libecs::DataPointVectorRCPtr LocalSimulatorImplementation::
  getLoggerData( libecs::StringCref aFullPNString, 
		 libecs::RealCref aStartTime, 
		 libecs::RealCref anEndTime,
		 libecs::RealCref anInterval ) const
  {
    return getLogger( aFullPNString )->getData( aStartTime, anEndTime, 
						anInterval );
  }

  const libecs::Real LocalSimulatorImplementation::
  getLoggerStartTime( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getStartTime();
  }

  const libecs::Real LocalSimulatorImplementation::
  getLoggerEndTime( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getEndTime();
  }

  void LocalSimulatorImplementation::
  setLoggerMinimumInterval( libecs::StringCref aFullPNString, 
			    libecs::RealCref anInterval )
  {
    getLogger( aFullPNString )->setMinimumInterval( anInterval );
  }

  const libecs::Real LocalSimulatorImplementation::
  getLoggerMinimumInterval( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getMinimumInterval();
  }


  const libecs::Int LocalSimulatorImplementation::
  getLoggerSize( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getSize();
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

  const libecs::Real LocalSimulatorImplementation::getCurrentTime() const
  {
    return getModel().getCurrentTime();
  }


  void LocalSimulatorImplementation::run()
  {
    getModel().initialize();

    if( ! ( theEventChecker != NULLPTR && theEventHandler != NULLPTR ) )
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

	while( (*theEventChecker)() )
	  {
	    (*theEventHandler)();
	  }

      }	while( theRunningFlag );

    getModel().flushLogger();
  }

  void LocalSimulatorImplementation::run( libecs::Real aDuration )
  {
    getModel().initialize();

    if( theEventChecker != NULLPTR && theEventHandler != NULLPTR )
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


	while( (*theEventChecker)() )
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
  setEventChecker( EventCheckerRCPtrCref aEventChecker )
  {
    theEventChecker = aEventChecker;
  }

  void LocalSimulatorImplementation::
  setEventHandler( EventHandlerRCPtrCref anEventHandler )
  {
    theEventHandler = anEventHandler;
  }

  void LocalSimulatorImplementation::clearEventChecker()
  {
    setEventChecker( EventCheckerRCPtr( new DefaultEventChecker() ) );
  }


} // namespace libemc,


