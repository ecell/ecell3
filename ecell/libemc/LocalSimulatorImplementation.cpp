//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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

#include <iostream>
#include <algorithm>

#include "libecs/libecs.hpp"
#include "libecs/Model.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/System.hpp"
#include "libecs/Process.hpp"
#include "libecs/Variable.hpp"
#include "libecs/LoggerBroker.hpp"
#include "libecs/StepperMaker.hpp"
#include "libecs/ProcessMaker.hpp"
#include "libecs/SystemMaker.hpp"
#include "libecs/VariableMaker.hpp"

#include "LocalSimulatorImplementation.hpp"


namespace libemc
{

  using namespace libecs;

  LocalSimulatorImplementation::LocalSimulatorImplementation()
    :
    theRunningFlag( false ),
    theMutatedFlag( true ),
    theEventCheckInterval( 20 ),
    theModel( *new Model ),
    theEventChecker(),
    theEventHandler()
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
    if( theRunningFlag )
      {
	THROW_EXCEPTION( libecs::Exception, 
			 "Cannot create a Stepper while running." );
      }

    getModel().createStepper( aClassname, anId );
  }

  void LocalSimulatorImplementation::deleteStepper( libecs::StringCref anID )
  {
    THROW_EXCEPTION( libecs::NotImplemented,
		     "deleteStepper() method is not supported yet." );
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

  void LocalSimulatorImplementation::
  loadStepperProperty( libecs::StringCref          aStepperID,
		       libecs::StringCref          aPropertyName,
		       libecs::PolymorphCref aValue )
  {
    StepperPtr aStepperPtr( getModel().getStepper( aStepperID ) );
    
    aStepperPtr->loadProperty( aPropertyName, aValue );
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  saveStepperProperty( libecs::StringCref aStepperID,
		       libecs::StringCref aPropertyName ) const
  {
    StepperCptr aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->saveProperty( aPropertyName );
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
    if( theRunningFlag )
      {
	THROW_EXCEPTION( libecs::Exception, 
			 "Cannot create an Entity while running." );
      }

    getModel().createEntity( aClassname, FullID( aFullIDString ) );
  }

  void LocalSimulatorImplementation::deleteEntity( StringCref aFullIDString )
  {
    THROW_EXCEPTION( libecs::NotImplemented,
		     "deleteEntity() method is not supported yet." );
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  getEntityList( libecs::StringCref anEntityTypeString,
		 libecs::StringCref aSystemPathString ) const
  {
    const libecs::EntityType anEntityType( anEntityTypeString );
    const libecs::SystemPath aSystemPath( aSystemPathString );

    if( aSystemPath.size() == 0 )
      {
	PolymorphVector aVector;
	if( anEntityType == libecs::EntityType::SYSTEM )
	  {
	    aVector.push_back( Polymorph( "/" ) );
	  }
	return aVector;
      }

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
	IGNORE_RETURN getModel().getEntity( FullID( aFullIDString ) );
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

  void LocalSimulatorImplementation::
  loadEntityProperty( StringCref aFullPNString, PolymorphCref aValue )
  {
    FullPN aFullPN( aFullPNString );
    EntityPtr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    anEntityPtr->loadProperty( aFullPN.getPropertyName(), aValue );
  }

  const Polymorph LocalSimulatorImplementation::
  saveEntityProperty( StringCref aFullPNString ) const
  {
    FullPN aFullPN( aFullPNString );
    EntityCptr anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    return anEntityPtr->saveProperty( aFullPN.getPropertyName() );
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
    libecs::PolymorphVector aVector;
    aVector.push_back(Integer(1));
    aVector.push_back(Real(0.0));
    aVector.push_back(Integer(0));
    aVector.push_back(Integer(0));
    createLogger( aFullPNString, libecs::Polymorph(aVector) );
  }


  void LocalSimulatorImplementation::
  createLogger( libecs::StringCref aFullPNString, 
		libecs::Polymorph aParamList )
  {
    if( theRunningFlag )
      {
	THROW_EXCEPTION( libecs::Exception, 
			 "Cannot create a Logger while running." );
      }

    if ( aParamList.getType() != libecs::Polymorph::POLYMORPH_VECTOR )
      {
	THROW_EXCEPTION( libecs::Exception,
			 "2nd argument of createLogger must be a list.");
      }

    FullPN aFullPN( aFullPNString );
    getModel().getLoggerBroker().
      createLogger( aFullPN, aParamList.asPolymorphVector() );
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


  const libecs::DataPointVectorSharedPtr LocalSimulatorImplementation::
  getLoggerData( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getData();
  }

  const libecs::DataPointVectorSharedPtr LocalSimulatorImplementation::
  getLoggerData( libecs::StringCref aFullPNString, 
		 libecs::RealCref aStartTime, 
		 libecs::RealCref anEndTime ) const
  {
    return getLogger( aFullPNString )->getData( aStartTime, anEndTime );
  }


  const libecs::DataPointVectorSharedPtr LocalSimulatorImplementation::
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


  void LocalSimulatorImplementation::
  setLoggerPolicy( libecs::StringCref aFullPNString, 
		   libecs::Polymorph aParamList )
  {
    if ( aParamList.getType() != libecs::Polymorph::POLYMORPH_VECTOR )
      {
	THROW_EXCEPTION( libecs::Exception,
			 "2nd parameter of logger policy must be a list.");
      }

    getLogger( aFullPNString )->setLoggerPolicy( aParamList );
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  getLoggerPolicy( libecs::StringCref aFullPNString ) const
  {

    return getLogger( aFullPNString )->getLoggerPolicy();
  }

  const libecs::Integer LocalSimulatorImplementation::
  getLoggerSize( libecs::StringCref aFullPNString ) const
  {
    return getLogger( aFullPNString )->getSize();
  }

  const libecs::Polymorph LocalSimulatorImplementation::
  getNextEvent() const
  {
    libecs::SchedulerEventCref aNextEvent( getModel().getNextEvent() );

    PolymorphVector aVector;
    aVector.push_back( static_cast<Real>( aNextEvent.getTime() ) );
    aVector.push_back( aNextEvent.getStepper()->getID() );
    return aVector;
  }


  void LocalSimulatorImplementation::step( const libecs::Integer aNumSteps )
  {
    getModel().initialize();  

    theRunningFlag = true;

    libecs::Integer aCounter( aNumSteps );
    do
      {
	getModel().step();
	
	--aCounter;
	
	if( aCounter == 0 )
	  {
	    theRunningFlag = false;
	    break;
	  }

	if( aCounter % 20 == 0 )
	  {
	    while( (*theEventChecker)() )
	      {
		(*theEventHandler)();
	      }
	    if( ! theRunningFlag )
	      {
		break;
	      }
	  }
      }	while( 1 );

    getModel().flushLoggers();
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
	for( unsigned int i( 20 ); i != 0; --i )
	  {
	    getModel().step();
	  }

	while( (*theEventChecker)() )
	  {
	    (*theEventHandler)();
	  }

      }	while( theRunningFlag );

    getModel().flushLoggers();
  }

  void LocalSimulatorImplementation::run( const libecs::Real aDuration )
  {
    if( aDuration <= 0.0 )
      {
	THROW_EXCEPTION( libecs::Exception,
			 "run( l ): l must be > 0.0. (" + 
			 libecs::stringCast( aDuration ) + " given.)" );
      }

    getModel().initialize();

    if( theEventChecker != NULLPTR && theEventHandler != NULLPTR )
      {
	runWithEvent( aDuration );
      }
    else
      {
	runWithoutEvent( aDuration );
      }

    getModel().flushLoggers();
  }

  void LocalSimulatorImplementation::runWithEvent( libecs::Real aDuration )
  {
    theRunningFlag = true;

    const libecs::Real aStopTime( getModel().getCurrentTime() + aDuration );

    do
      {
	for( unsigned int i( 20 ); i != 0; --i )
	  {
	    if( getModel().getCurrentTime() > aStopTime )
	      {
		theRunningFlag = false;
		return;  // the exit
	      }
	    
	    getModel().step();
	  }

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
  setEventChecker( EventCheckerSharedPtrCref aEventChecker )
  {
    theEventChecker = aEventChecker;
  }

  void LocalSimulatorImplementation::
  setEventHandler( EventHandlerSharedPtrCref anEventHandler )
  {
    theEventHandler = anEventHandler;
  }

  void LocalSimulatorImplementation::clearEventChecker()
  {
    setEventChecker( EventCheckerSharedPtr( new DefaultEventChecker() ) );
  }

  const libecs::Polymorph LocalSimulatorImplementation::getDMInfo()
  {
    libecs::PolymorphVector aVector;

    
    // ugly hack...
    // ModuleMaker should be reconstructed to make this cleanly.

    ProcessMakerRef aProcessMaker( getModel().getProcessMaker() );
    for( libecs::ProcessMaker::ModuleMap::const_iterator
	   i( aProcessMaker.getModuleMap().begin() ); 
	 i != aProcessMaker.getModuleMap().end(); ++i )
      {
	libecs::PolymorphVector anInnerVector;

	anInnerVector.push_back( Polymorph( "Process" ) );
	anInnerVector.push_back( Polymorph( i->first ) );
	anInnerVector.push_back( Polymorph( i->second->getFileName() ) );

	aVector.push_back( anInnerVector );
      }

    StepperMakerRef aStepperMaker( getModel().getStepperMaker() );
    for( libecs::StepperMaker::ModuleMap::const_iterator
	   i( aStepperMaker.getModuleMap().begin() ); 
	 i != aStepperMaker.getModuleMap().end(); ++i )
      {
	libecs::PolymorphVector anInnerVector;

	anInnerVector.push_back( Polymorph( "Stepper" ) );
	anInnerVector.push_back( Polymorph( i->first ) );
	anInnerVector.push_back( Polymorph( i->second->getFileName() ) );

	aVector.push_back( anInnerVector );
      }

    SystemMakerRef aSystemMaker( getModel().getSystemMaker() );
    for( libecs::SystemMaker::ModuleMap::const_iterator
	   i( aSystemMaker.getModuleMap().begin() ); 
	 i != aSystemMaker.getModuleMap().end(); ++i )
      {
	libecs::PolymorphVector anInnerVector;

	anInnerVector.push_back( Polymorph( "System" ) );
	anInnerVector.push_back( Polymorph( i->first ) );
	anInnerVector.push_back( Polymorph( i->second->getFileName() ) );

	aVector.push_back( anInnerVector );
      }

    VariableMakerRef aVariableMaker( getModel().getVariableMaker() );
    for( libecs::VariableMaker::ModuleMap::const_iterator
	   i( aVariableMaker.getModuleMap().begin() ); 
	 i != aVariableMaker.getModuleMap().end(); ++i )
      {
	libecs::PolymorphVector anInnerVector;

	anInnerVector.push_back( Polymorph( "Variable" ) );
	anInnerVector.push_back( Polymorph( i->first ) );
	anInnerVector.push_back( Polymorph( i->second->getFileName() ) );

	aVector.push_back( anInnerVector );
      }

    return aVector;
  }


} // namespace libemc,


