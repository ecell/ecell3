//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
#ifdef DLL_EXPORT
#undef DLL_EXPORT
#define _DLL_EXPORT
#endif /* DLL_EXPORT */

#include "libecs/libecs.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/System.hpp"
#include "libecs/Process.hpp"
#include "libecs/Variable.hpp"
#include "libecs/LoggerBroker.hpp"
#include "libecs/SystemStepper.hpp"

#ifdef _DLL_EXPORT
#define DLL_EXPORT
#undef _DLL_EXPORT
#endif /* _DLL_EXPORT */

#include "LocalSimulatorImplementation.hpp"

namespace libemc
{

LocalSimulatorImplementation::LocalSimulatorImplementation()
    : theRunningFlag( false ),
      theDirtyFlag( true ),
      theEventCheckInterval( 20 ),
      theEventChecker(),
      theEventHandler(),
      thePropertiedObjectMaker( libecs::createDefaultModuleMaker() ),
      theModel( *thePropertiedObjectMaker )
{
    theModel.setup();
    clearEventChecker();
}


LocalSimulatorImplementation::~LocalSimulatorImplementation()
{
    delete thePropertiedObjectMaker;
}


libecs::Logger* LocalSimulatorImplementation::getLogger(
        libecs::String const& aFullPNString ) const
{
    return getModel().getLoggerBroker().getLogger( aFullPNString );
}


void LocalSimulatorImplementation::createStepper(
        libecs::String const& aClassname, 
        libecs::String const& anId )
{
    if( theRunningFlag )
    {
        THROW_EXCEPTION( libecs::Exception, 
                         "cannot create a Stepper during simulation" );
    }

    setDirty();
    getModel().createStepper( aClassname, anId );
}


inline libecs::Polymorph LocalSimulatorImplementation::buildPolymorph( const libecs::Logger::Policy& pol )
{
    return libecs::Polymorph(
        boost::make_tuple( pol.getMinimumStep(), pol.getMinimumTimeInterval(),
                           static_cast< libecs::Integer >(
                               pol.doesContinueOnError() ),
                           pol.getMaxSpace() ) );
}


inline libecs::Polymorph LocalSimulatorImplementation::buildPolymorph( const libecs::PropertyAttributes& attrs )
{
    using namespace libecs;
    return libecs::Polymorph(
        boost::make_tuple( static_cast< Integer >( attrs.isSetable() ),
                           static_cast< Integer >( attrs.isGetable() ),
                           static_cast< Integer >( attrs.isLoadable() ),
                           static_cast< Integer >( attrs.isSavable() ),
                           static_cast< Integer >( attrs.isDynamic() ),
                           static_cast< Integer >( attrs.getType() ) ) ); 
}


void LocalSimulatorImplementation::deleteStepper( libecs::String const& anID )
{
    getModel().deleteStepper( anID );
    setDirty();
}


libecs::Polymorph LocalSimulatorImplementation::getStepperList() const
{
    libecs::Model::StepperMap const& aStepperMap( getModel().getStepperMap() );

    libecs::PolymorphVector aPolymorphVector; 
    aPolymorphVector.reserve( aStepperMap.size() );
    
    for( libecs::Model::StepperMap::const_iterator i( aStepperMap.begin() );
         i != aStepperMap.end(); ++i )
    {
        aPolymorphVector.push_back( libecs::Polymorph( (*i).first ) );
    }

    return libecs::Polymorph( aPolymorphVector );
}


libecs::Polymorph
LocalSimulatorImplementation::getStepperPropertyList(
        libecs::String const& aStepperID ) const
{
    libecs::Stepper* const aStepperPtr( getModel().getStepper( aStepperID ) );
    
    return libecs::Polymorph( aStepperPtr->getPropertyList() );
}

libecs::Polymorph
LocalSimulatorImplementation::getStepperPropertyAttributes(
        libecs::String const& aStepperID, 
        libecs::String const& aPropertyName ) const
{
    libecs::Stepper* const aStepperPtr( getModel().getStepper( aStepperID ) );
    return buildPolymorph( aStepperPtr->getPropertyAttributes( aPropertyName ) );
}


void LocalSimulatorImplementation::setStepperProperty(
        libecs::String const& aStepperID,
        libecs::String const& aPropertyName,
        libecs::Polymorph const& aValue )
{
    libecs::Stepper* const aStepperPtr( getModel().getStepper( aStepperID ) );
    
    setDirty();
    aStepperPtr->setProperty( aPropertyName, aValue );
}

libecs::Polymorph LocalSimulatorImplementation::getStepperProperty(
        libecs::String const& aStepperID,
        libecs::String const& aPropertyName ) const
{
    libecs::Stepper const* const aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->getProperty( aPropertyName );
}

void LocalSimulatorImplementation::loadStepperProperty(
        libecs::String const& aStepperID,
        libecs::String const& aPropertyName,
        libecs::Polymorph const& aValue )
{
    libecs::Stepper* const aStepperPtr( getModel().getStepper( aStepperID ) );
    
    setDirty();
    aStepperPtr->loadProperty( aPropertyName, aValue );
}

libecs::Polymorph LocalSimulatorImplementation::saveStepperProperty(
        libecs::String const& aStepperID,
        libecs::String const& aPropertyName ) const
{
    libecs::Stepper const* const aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->saveProperty( aPropertyName );
}

libecs::String LocalSimulatorImplementation::getStepperClassName(
        libecs::String const& aStepperID ) const
{
    libecs::Stepper const* const aStepperPtr( getModel().getStepper( aStepperID ) );

    return aStepperPtr->getPropertyInterface().getClassName();
}


void LocalSimulatorImplementation::createEntity(
        libecs::String const& aClassname, libecs::String const& aFullIDString )
{
    if( theRunningFlag )
    {
        THROW_EXCEPTION( libecs::Exception, 
                         "cannot create an Entity during simulation" );
    }

    setDirty();
    getModel().createEntity( aClassname, libecs::FullID( aFullIDString ) );
}

void LocalSimulatorImplementation::deleteEntity(
        libecs::String const& aFullIDString )
{
    getModel().deleteEntity( libecs::FullID( aFullIDString ) );
}

libecs::Polymorph LocalSimulatorImplementation::getEntityList(
        libecs::String const& anEntityTypeString,
        libecs::String const& aSystemPathString ) const
{
    const libecs::EntityType anEntityType( anEntityTypeString );
    const libecs::SystemPath aSystemPath( aSystemPathString );

    if( aSystemPath.size() == 0 )
    {
        libecs::PolymorphVector aVector;
        if( anEntityType == libecs::EntityType::SYSTEM )
        {
            aVector.push_back( libecs::Polymorph( "/" ) );
        }
        return libecs::Polymorph( aVector );
    }

    libecs::System const* const aSystemPtr( getModel().getSystem( aSystemPath ) );

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


libecs::Polymorph LocalSimulatorImplementation::getEntityPropertyList(
        libecs::String const& aFullIDString ) const
{
    libecs::Entity const* anEntityPtr( getModel().getEntity( libecs::FullID( aFullIDString ) ) );

    return libecs::Polymorph( anEntityPtr->getPropertyList() );
}


bool LocalSimulatorImplementation::entityExists(
        libecs::String const& aFullIDString ) const
{
    try
    {
        (void)getModel().getEntity( libecs::FullID( aFullIDString ) );
    }
    catch( const libecs::NotFound& )
    {
        return false;
    }

    return true;
}


void LocalSimulatorImplementation::setEntityProperty(
        libecs::String const& aFullPNString, libecs::Polymorph const& aValue )
{
    libecs::FullPN aFullPN( aFullPNString );
    libecs::Entity* const anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    setDirty();
    anEntityPtr->setProperty( aFullPN.getPropertyName(), aValue );
}


libecs::Polymorph LocalSimulatorImplementation::getEntityProperty(
        libecs::String const& aFullPNString ) const
{
    libecs::FullPN aFullPN( aFullPNString );
    libecs::Entity const* anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );
            
    return anEntityPtr->getProperty( aFullPN.getPropertyName() );
}

void LocalSimulatorImplementation::loadEntityProperty(
        libecs::String const& aFullPNString, libecs::Polymorph const& aValue )
{
    libecs::FullPN aFullPN( aFullPNString );
    libecs::Entity* const anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    setDirty();
    anEntityPtr->loadProperty( aFullPN.getPropertyName(), aValue );
}

libecs::Polymorph LocalSimulatorImplementation::saveEntityProperty(
        libecs::String const& aFullPNString ) const
{
    libecs::FullPN aFullPN( aFullPNString );
    libecs::Entity const* const anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    return anEntityPtr->saveProperty( aFullPN.getPropertyName() );
}

libecs::Polymorph LocalSimulatorImplementation::
getEntityPropertyAttributes( libecs::String const& aFullPNString ) const
{
    libecs::FullPN aFullPN( aFullPNString );
    libecs::Entity const* const anEntityPtr( getModel().getEntity( aFullPN.getFullID() ) );

    return buildPolymorph( anEntityPtr->getPropertyAttributes( aFullPN.getPropertyName() ) );
}

libecs::String LocalSimulatorImplementation::
getEntityClassName( libecs::String const& aFullIDString ) const
{
    libecs::FullID aFullID( aFullIDString );
    libecs::Entity const* const anEntityPtr( getModel().getEntity( aFullID ) );

    return anEntityPtr->getPropertyInterface().getClassName();
}


void LocalSimulatorImplementation::
createLogger( libecs::String const& aFullPNString )
{
    libecs::PolymorphVector aVector;
    createLogger( aFullPNString, boost::make_tuple( 1l, 0.0, 0l, 0l ) );
}


void LocalSimulatorImplementation::createLogger(
        libecs::String const& aFullPNString, 
        libecs::Polymorph aParamList )
{
    typedef libecs::PolymorphValue::Tuple Tuple;

    if( theRunningFlag )
    {
        THROW_EXCEPTION( libecs::Exception, 
                         "cannot create a Logger during simulation" );
    }

    if ( aParamList.getType() != libecs::PolymorphValue::TUPLE
         || aParamList.as< Tuple const& >().size() != 4 )
    {
        THROW_EXCEPTION( libecs::Exception,
                         "second argument must be a tuple of 4 items");
    }

    getModel().getLoggerBroker().createLogger(
        libecs::FullPN( aFullPNString ),
        libecs::Logger::Policy(
            aParamList.as< Tuple const& >()[ 0 ].as< libecs::Integer >(),
            aParamList.as< Tuple const& >()[ 1 ].as< libecs::Real >(),
            aParamList.as< Tuple const& >()[ 2 ].as< libecs::Integer >(),
            aParamList.as< Tuple const& >()[ 3 ].as< libecs::Integer >() != 0 ? true: false ) );

    setDirty();
}

libecs::Polymorph LocalSimulatorImplementation::getLoggerList() const
{
    libecs::PolymorphVector aLoggerList;

    libecs::LoggerBroker const& aLoggerBroker( getModel().getLoggerBroker() );

    for( libecs::LoggerBroker::const_iterator
            i( aLoggerBroker.begin() ), end( aLoggerBroker.end() );
         i != end; ++i )
    {
        aLoggerList.push_back( libecs::Polymorph( (*i).first.asString() ) );
    }

    return libecs::Polymorph( aLoggerList );
}


boost::shared_ptr< libecs::DataPointVector >
LocalSimulatorImplementation::getLoggerData(
        libecs::String const& aFullPNString ) const
{
    return getLogger( aFullPNString )->getData();
}

boost::shared_ptr< libecs::DataPointVector >
LocalSimulatorImplementation::getLoggerData(
        libecs::String const& aFullPNString, 
        libecs::Real aStartTime, 
        libecs::Real anEndTime ) const
{
    return getLogger( aFullPNString )->getData( aStartTime, anEndTime );
}


boost::shared_ptr< libecs::DataPointVector >
LocalSimulatorImplementation::getLoggerData(
        libecs::String const& aFullPNString, 
        libecs::Real aStartTime, 
        libecs::Real anEndTime,
        libecs::Real anInterval ) const
{
    return getLogger( aFullPNString )->getData( aStartTime, anEndTime, anInterval );
}

libecs::Real
LocalSimulatorImplementation::getLoggerStartTime(
        libecs::String const& aFullPNString ) const
{
    return getLogger( aFullPNString )->getStartTime();
}

libecs::Real LocalSimulatorImplementation::getLoggerEndTime(
        libecs::String const& aFullPNString ) const
{
    return getLogger( aFullPNString )->getEndTime();
}

void LocalSimulatorImplementation::setLoggerPolicy(
        libecs::String const& aFullPNString, 
        libecs::Polymorph aParamList )
{
    typedef libecs::PolymorphValue::Tuple Tuple;

    if( aParamList.getType() != libecs::PolymorphValue::TUPLE
        || aParamList.as< Tuple const& >().size() != 4 )
    {
        THROW_EXCEPTION( libecs::Exception,
                         "second parameter must be a tuple of 4 items");
    }

    getLogger( aFullPNString )->setLoggerPolicy(
        libecs::Logger::Policy(
            aParamList.as< Tuple const& >()[ 0 ].as< libecs::Integer >(),
            aParamList.as< Tuple const& >()[ 1 ].as< libecs::Real >(),
            aParamList.as< Tuple const& >()[ 2 ].as< libecs::Integer >(),
            aParamList.as< Tuple const& >()[ 3 ].as< libecs::Integer >() != 0 ? true: false ) );
}


libecs::Polymorph
LocalSimulatorImplementation::getLoggerPolicy(
        libecs::String const& aFullPNString ) const
{
    return buildPolymorph( getLogger( aFullPNString )->getLoggerPolicy() );
}


libecs::Logger::size_type LocalSimulatorImplementation::
getLoggerSize( libecs::String const& aFullPNString ) const
{
    return getLogger( aFullPNString )->getSize();
}


libecs::Polymorph
LocalSimulatorImplementation::getNextEvent() const
{
    libecs::StepperEvent const& aNextEvent( getModel().getTopEvent() );

    return boost::make_tuple(
        static_cast< libecs::Real >( aNextEvent.getTime() ),
        aNextEvent.getStepper()->getID() );
}


void LocalSimulatorImplementation::initialize() const
{
    // calling the model's initialize(), which is non-const,
    // is semantically a const operation at the simulator level.
    const_cast< LocalSimulatorImplementation* >( this )->getModel().initialize();
}


libecs::Real LocalSimulatorImplementation::getCurrentTime() const
{
    return getModel().getCurrentTime();
}



void LocalSimulatorImplementation::step( const libecs::Integer aNumSteps )
{
    if( aNumSteps <= 0 )
    {
        THROW_EXCEPTION( libecs::Exception,
                         "step( n ): n must be 1 or greater ("
                         + libecs::stringCast( aNumSteps ) + " given)" );
    }

    start();

    libecs::Integer aCounter( aNumSteps );
    do
    {
        getModel().step();
        
        --aCounter;
        
        if( aCounter == 0 )
        {
            stop();
            break;
        }

        if( aCounter % theEventCheckInterval == 0 )
        {
            handleEvent();

            if( ! theRunningFlag )
            {
                break;
            }
        }
    }
    while( 1 );

}

void LocalSimulatorImplementation::run()
{
    if( ! ( typeid( *theEventChecker ) != 
            typeid( DefaultEventChecker ) && 
            theEventHandler.get() ) )
    {
        THROW_EXCEPTION( libecs::Exception,
                         "both EventChecker and EventHandler must be "
                         "set before run without duration" ) ;
    }

    start();

    do
    {
        unsigned int aCounter( theEventCheckInterval );
        do
        {
            getModel().step();
            --aCounter;
        }
        while( aCounter != 0 );
        
        handleEvent();

    }
    while( theRunningFlag );
}

void LocalSimulatorImplementation::run( const libecs::Real aDuration )
{
    if( aDuration <= 0.0 )
    {
        THROW_EXCEPTION( libecs::Exception,
                         "duration must be greater than 0 ("
                         + libecs::stringCast( aDuration ) + " given)" );
    }

    start();

    const libecs::Real aCurrentTime( getModel().getCurrentTime() );
    const libecs::Real aStopTime( aCurrentTime + aDuration );

    // setup SystemStepper to step at aStopTime

    //FIXME: dirty, ugly!
    libecs::Stepper* const aSystemStepper( getModel().getSystemStepper() );
    aSystemStepper->setCurrentTime( aCurrentTime );
    aSystemStepper->setNextTime( aStopTime );

    getModel().getScheduler().updateEvent( 0, aStopTime );


    if( typeid( *theEventChecker ) != typeid( DefaultEventChecker ) &&
        theEventHandler.get() )
    {
        runWithEvent( aStopTime );
    }
    else
    {
        runWithoutEvent( aStopTime );
    }

}

void LocalSimulatorImplementation::runWithEvent( libecs::Real const aStopTime )
{
    do
    {
        unsigned int aCounter( theEventCheckInterval );
        do 
        {
            if( getModel().getTopEvent().getTime() > aStopTime )
            {
                stop();
                return;
            }
            
            getModel().step();

            --aCounter;
        }
        while( aCounter != 0 );

        handleEvent();

    }
    while( theRunningFlag );
}

void LocalSimulatorImplementation::runWithoutEvent( libecs::Real const aStopTime )
{
    libecs::Stepper const* const aSystemStepper( getModel().getSystemStepper() );

    do
    {
        if( getModel().getTopEvent().getTime() > aStopTime )
        {
            stop();
            return;    // the only exit
        }

        getModel().step();

    }
    while( 1 );
}

void LocalSimulatorImplementation::stop()
{
    theRunningFlag = false;

    getModel().flushLoggers();
}

void LocalSimulatorImplementation::
setEventChecker( boost::shared_ptr< EventChecker > const& anEventChecker )
{
    theEventChecker = anEventChecker;
}

void LocalSimulatorImplementation::
setEventHandler( boost::shared_ptr< EventHandler > const& anEventHandler )
{
    theEventHandler = anEventHandler;
}

void LocalSimulatorImplementation::clearEventChecker()
{
    setEventChecker( boost::shared_ptr< EventChecker >( new DefaultEventChecker() ) );
}

SimulatorImplementation::PolymorphMap 
LocalSimulatorImplementation::getClassInfo( libecs::String const& aClassname ) const
{
    SimulatorImplementation::PolymorphMap aBuiltInfoMap;
    for ( DynamicModuleInfo::EntryIterator* anInfo(
          getModel().getPropertyInterface( aClassname ).getInfoFields() );
          anInfo->next(); )
    {
        aBuiltInfoMap.insert( std::make_pair( anInfo->current().first,
                              *reinterpret_cast< const libecs::Polymorph* >(
                                anInfo->current().second ) ) );
    }
    return aBuiltInfoMap;
}

SimulatorImplementation::PolymorphMap
LocalSimulatorImplementation::getPropertyInfo( libecs::String const& aClassname ) const
{
    typedef libecs::PropertyInterfaceBase::PropertySlotMap PropertySlotMap;
    SimulatorImplementation::PolymorphMap retval;
    const PropertySlotMap& slots( getModel().getPropertyInterface( aClassname ).getPropertySlotMap() );

    for( PropertySlotMap::const_iterator i( slots.begin() );
         i != slots.end(); ++i)
    {
        retval.insert( std::make_pair( i->first,
                buildPolymorph( *i->second ) ) );
    }

    return retval;
}

libecs::PolymorphVector LocalSimulatorImplementation::getDMInfo() const
{
    typedef ModuleMaker< libecs::EcsObject >::ModuleMap ModuleMap;
    libecs::PolymorphVector aVector;
    const ModuleMap& modules( thePropertiedObjectMaker->getModuleMap() );

    for( ModuleMap::const_iterator i( modules.begin() );
                i != modules.end(); ++i )
    {
        libecs::PolymorphVector anInnerVector;
        const libecs::PropertyInterfaceBase* info(
            reinterpret_cast< const libecs::PropertyInterfaceBase *>(
                i->second->getInfo() ) );
        const char* aFilename( i->second->getFileName() );

        aVector.push_back( boost::make_tuple(
            info->getTypeName(), i->second->getModuleName(),
            libecs::String( aFilename ? aFilename: "" ) ) );
    }

    return aVector;
}

char LocalSimulatorImplementation::getDMSearchPathSeparator() const
{
    return libecs::Model::PATH_SEPARATOR;
}

libecs::String LocalSimulatorImplementation::getDMSearchPath() const
{
    return theModel.getDMSearchPath();
}

void LocalSimulatorImplementation::setDMSearchPath( libecs::String const& aDMSearchPath )
{
    theModel.setDMSearchPath( aDMSearchPath );
}

} // namespace libemc,
