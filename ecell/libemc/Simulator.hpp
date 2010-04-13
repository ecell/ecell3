//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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
#ifndef __SIMULATOR_HPP
#define __SIMULATOR_HPP

#ifdef DLL_EXPORT
#undef DLL_EXPORT
#define _DLL_EXPORT
#endif /* DLL_EXPORT */

#include "libecs/libecs.hpp"
#include "libecs/EntityType.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/DataPointVector.hpp"
#include "libecs/Logger.hpp"

#ifdef _DLL_EXPORT
#define DLL_EXPORT
#undef _DLL_EXPORT
#endif /* _DLL_EXPORT */

#include "libemc.hpp"
#include "SimulatorImplementation.hpp"

namespace libemc
{

/**
   The interface to the simulator.

   Simulator class provides a unified API to the libecs, 
   C++ library for cell modeling and simulation.

   Unlike libecs::Model class, this API does involve only standard
   C++ types/classes, and doesn't depend on libecs classes.    An only
   exception is Polymorph class.

   The public API methods are classified into these four groups:

   - Entity methods
   - Stepper methods
   - Logger methods, and
   - Simulator methods
   
   @see libecs
   @see Model
   @see SimulatorImplementation
*/
class LIBEMC_API Simulator
{
public:
    typedef std::map< libecs::String, libecs::Polymorph > PolymorphMap;

public:
    Simulator();
    virtual ~Simulator();


    /**
       @name Stepper methods.
    */

    //@{

    /**
       Create a new Stepper in the model.

       @param aClassname a classname of the Stepper to create.
       @param anID             an ID of the Stepper.
       @param aParameterList a list of parameters to give to the Stepper
    */
    void createStepper( libecs::String const& aClassname,
                        libecs::String const& anId )
    {
        theSimulatorImplementation->createStepper( aClassname, anId );
    }

    /**
       Delete a Stepper.
       This method is not supported yet.
    */
    void deleteStepper( libecs::String const& anID )
    {
        theSimulatorImplementation->deleteStepper( anID );
    }

    /**
       List Steppers in the model.

       @returh a list of Steppers.
    */
    libecs::Polymorph getStepperList() const
    {
        return theSimulatorImplementation->getStepperList();
    }

    /**
       List names of properties of a Stepper.
    
       @return a list of properties of a Stepper.
    */
    libecs::Polymorph 
    getStepperPropertyList( libecs::String const& aStepperID ) const
    {
        return theSimulatorImplementation->getStepperPropertyList( aStepperID );
    }

    /**
       Get attributes of a property of a Stepper.

       The attributes are returned as a form of boolean 2-tuple 
       ( setable, getable ).    ( 1, 0 ) means that the property is setable but
       not getable,,, and so on.

       @return Stepper property attributes.
    */
    libecs::Polymorph 
    getStepperPropertyAttributes( libecs::String const& aStepperID, 
                                  libecs::String const& aPropertyName ) const
    {
        return theSimulatorImplementation->
            getStepperPropertyAttributes( aStepperID, aPropertyName );
    }

    /**
       Set a property value of a Stepper.

       @param aStepperID the Stepper ID.
       @param aValue     the value to set as a Polymorph.
    */
    void setStepperProperty( libecs::String const&    aStepperID,
                             libecs::String const&    aPropertyName,
                             libecs::Polymorph const& aValue )
    {
        theSimulatorImplementation->setStepperProperty( aStepperID,
                                                        aPropertyName,
                                                        aValue );
    }

    /**
       Get a value of a property from a Stepper.

       @param aStepperID the Stepper ID.
       @param aPropertyName the name of the property.
       @return the property value as a reference counted pointor of a 
       Polymorph.
    */
    libecs::Polymorph
    getStepperProperty( libecs::String const& aStepperID,
                        libecs::String const& aPropertyName ) const
    {
        return theSimulatorImplementation->getStepperProperty( aStepperID,
                                                               aPropertyName );
    }


    /**
       Load a property value of a Stepper.

       @param aStepperID the Stepper ID.
       @param aValue     the value to set as a Polymorph.
    */
    void loadStepperProperty( libecs::String const&    aStepperID,
                              libecs::String const&    aPropertyName,
                              libecs::Polymorph const& aValue )
    {
        theSimulatorImplementation->loadStepperProperty( aStepperID,
                                                         aPropertyName,
                                                         aValue );
    }

    /**
       Get a value of a property from an Stepper.

       @param aStepperID the Stepper ID.
       @param aPropertyName the name of the property.
       @return the property value as a reference counted pointer of a 
               Polymorph.
    */
    libecs::Polymorph
    saveStepperProperty( libecs::String const& aStepperID,
                         libecs::String const& aPropertyName ) const
    {
        return theSimulatorImplementation->saveStepperProperty( aStepperID,
                                                                aPropertyName );
    }

    /**
       Get class name of a Stepper.

       @param aStepperID the Stepper ID.
       @return the class name.
    */

    libecs::String
    getStepperClassName( libecs::String const& aStepperID ) const
    {
        return theSimulatorImplementation->getStepperClassName( aStepperID );
    }


    //@}

    
    /**
       @name Entity methods.
       @{
     */


    /**
       Create a new Entity in the model.

       @param aClassname    a classname of the Entity to create.
       @param aFullIDString FullID of the Entity.
       @param aName         a name of the Entity.
    */
    void createEntity( libecs::String const& aClassname, 
                       libecs::String const& aFullIDString )
    {
        theSimulatorImplementation->createEntity( aClassname, aFullIDString );
    }

    /**
       Delete an Entity. This method is not supported yet.
    */
    void deleteEntity( libecs::String const& aFullIDString )
    {
        theSimulatorImplementation->deleteEntity( aFullIDString );
    }

    /**
       Get a list of Entities in a System.

       @param anEntityTypeString an EntityType as a string
       @param aSystemPathString a SystemPath of the System.
       @return the list of IDs of Entities.
    */

    const libecs::Polymorph 
    getEntityList( libecs::String const& anEntityTypeString,
                   libecs::String const& aSystemPathString ) const
    {
        return theSimulatorImplementation->getEntityList( anEntityTypeString,
                                                          aSystemPathString );
    }


    /**
       List names of properties of an Entity. 
    
       @return a list of properties of an Entity.
    */
    libecs::Polymorph 
    getEntityPropertyList( libecs::String const& aFullIDString ) const
    {
        return theSimulatorImplementation->getEntityPropertyList( aFullIDString );
    }

    /**
       Check if an Entity object specified by a FullID exists in the model.

       @param aFullIDString a FullID string to be checked.
       @return true if the Entity exists, false if not.
    */

    bool entityExists( libecs::String const& aFullIDString ) const
    {
        return theSimulatorImplementation->entityExists( aFullIDString );
    }

    /**
       Set a property value of an Entity.

       @param aFullPNString a FullPN of the Property to set.
       @param aValue        the value to be set.
    */

    void setEntityProperty( libecs::String const&    aFullPNString,
                            libecs::Polymorph const& aValue )
    {
        theSimulatorImplementation->setEntityProperty( aFullPNString, aValue );
    }

    /**
       Get a value of a property from an Entity.

       @param aFullPNString a FullPN of the property.
       @return the property value.
    */
    libecs::Polymorph
    getEntityProperty( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->getEntityProperty( aFullPNString );
    }


    /**
       Load a property value of an Entity.

       @param aFullPNString a FullPN of the Property to set.
       @param aValue        the value to be set.
    */

    void loadEntityProperty( libecs::String const& aFullPNString,
                             libecs::Polymorph const& aValue )
    {
        theSimulatorImplementation->loadEntityProperty( aFullPNString, aValue );
    }

    /**
       Save a value of a property from an Entity.

       @param aFullPNString a FullPN of the property.
       @return the property value.
    */
    const libecs::Polymorph
    saveEntityProperty( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->saveEntityProperty( aFullPNString );
    }

    /**
       Get attributes of a property of an Entity.

       The attributes are returned as a form of boolean 2-tuple 
       ( setable, getable ).    ( 1, 0 ) means that the property is setable but
       not getable,,, and so on.

       @return Entity property attributes.
    */
    libecs::Polymorph
    getEntityPropertyAttributes( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->getEntityPropertyAttributes( aFullPNString );
    }

    /**
       Get class name of an Entity.

       @param aFullIDString a FullID of the Entity.
       @return the class name.
    */
    libecs::String
    getEntityClassName( libecs::String const& aFullIDString ) const
    {
        return theSimulatorImplementation->getEntityClassName( aFullIDString );
    }

    /** @} */


    /**
       @name Logger methods.
       @{
    */

    /**
       Create a Logger.

       If the Logger already exists, this method does nothing.

       @param aFullPNString a FullPN of the PropertySlot which the Logger is
       observing, as a String 

       @return a borrowed pointer to the Logger
    */
    void createLogger( libecs::String const& aFullPNString ) 
    {
                            
        return theSimulatorImplementation->createLogger( aFullPNString );
    }

    /**
         Create a Logger with parameters.
         - First parameter: minimum log interval dimension
           0 - none, 1 - by step, 2 - by time
         - Second parameter: behaviour when run out of disk
           0 - throw exception, 1 - overwrite data
         - Third parameter: minimum log interval

         If the Logger already exists, this method does nothing.

         @param aFullPNString the FullPN of the PropertySlot which the Logger is
                observing, as a String 
         @param aParamLsit the parameters
         @return a borrowed pointer to the Logger
    */

    void createLogger( libecs::String const& aFullPNString,
                       libecs::Polymorph aParamList ) 
    {
        return theSimulatorImplementation->createLogger( aFullPNString, aParamList );
    }

    /**
       List Loggers in the simulator.

       @return a list of Loggers.
    */

    libecs::Polymorph getLoggerList() const
    {
        return theSimulatorImplementation->getLoggerList();
    }

    boost::shared_ptr< libecs::DataPointVector > 
    getLoggerData( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->getLoggerData( aFullPNString );
    }

    boost::shared_ptr< libecs::DataPointVector >
    getLoggerData( libecs::String const& aFullPNString, 
                   libecs::Real aStartTime, 
                   libecs::Real anEndTime ) const 
    {
        return theSimulatorImplementation->getLoggerData( aFullPNString,
                                                          aStartTime, anEndTime );
    }

    boost::shared_ptr< libecs::DataPointVector >
    getLoggerData( libecs::String const& aFullPNString,
                   libecs::Real aStartTime, libecs::Real anEndTime,
                   libecs::Real anInterval ) const
    {
        return theSimulatorImplementation->getLoggerData( aFullPNString,
                                                          aStartTime,
                                                          anEndTime, 
                                                          anInterval );
    }

    libecs::Real 
    getLoggerStartTime( libecs::String const& aFullPNString ) const 
    {
        return theSimulatorImplementation->getLoggerStartTime( aFullPNString );
    }

    libecs::Real 
    getLoggerEndTime( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->getLoggerEndTime( aFullPNString );
    }

    void setLoggerPolicy( libecs::String const& aFullPNString, 
                          libecs::Polymorph aParamList )
    {
        return theSimulatorImplementation->setLoggerPolicy( aFullPNString,
                                                            aParamList );
    }

    libecs::Polymorph
    getLoggerPolicy( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->getLoggerPolicy( aFullPNString );
    }

    libecs::Logger::size_type getLoggerSize( libecs::String const& aFullPNString ) const
    {
        return theSimulatorImplementation->getLoggerSize( aFullPNString );
    }

    /** @} */

    /**
       @name Simulator methods.
       @{
    */

    /**
       Advance the simulation a step forward.
    */
    void step( libecs::Integer aNumSteps = 1 )
    {

        theSimulatorImplementation->step( aNumSteps );
    }

    /**
       Get a tuple of the time and the stepper ID of the next event

       @return a tuple of the time and the stepper ID of the next event
     */
    libecs::Polymorph getNextEvent() const
    {
        return theSimulatorImplementation->getNextEvent();
    }

    /**
       Get the current time of the simulator.

       @return current time of the simulator
    */
    libecs::Real getCurrentTime() const
    {
        return theSimulatorImplementation->getCurrentTime();
    }

    /**
       Run the simulation.

       @note Both the EventChecker and the EventHandler must be set 
       before calling this method.

       @see setEventChecker
       @see setEventHandler
    */
    void run()
    {
        theSimulatorImplementation->run();
    }

    /**
       Run the simulation with a duration.

       @param a duration of the simulation run.
    */
    void run( libecs::Real aDuration )
    {
        theSimulatorImplementation->run( aDuration );
    }

    /**
       Stop the simulation.

       Usually this is called from the EventHandler.
    */
    void stop()
    {
        theSimulatorImplementation->stop();
    }

    /**
       Set a pending event checker.

       The event checker must be a subclass of EventChecker class.

       This is usually used to set to form a mainloop of GUI toolkit.
       If you are using gtk, the event checker would call gtk_events_pending()
       function.

       While the simulation is running by the run() method, the function
       object given by this method is called once in several simulation 
       steps.    If it returns true, the EventHandler given by setEventHandler()
       method is called.

       @param aEventChecker a function object of the event checker
       @see EventChecker
       @see setEventHandler
    */
    void setEventChecker( boost::shared_ptr< EventChecker > const& aEventChecker )
    {
         theSimulatorImplementation->setEventChecker( aEventChecker );
    }

    /**
       Set an event handler.

       The event handler must be a subclass of EventHandler class.

       If you are using gtk, it would call gtk_main_iteration() function.

       @param anEventHandler a function object of the event handler
       @see EventHandler
       @see setEventChecker
    */
    void setEventHandler( boost::shared_ptr< EventHandler > const& anEventHandler )
    {
         theSimulatorImplementation->setEventHandler( anEventHandler );
    }

    /**
       Return a list of tuples which contain the information of loaded DMs.

       @return a list of tuples.
     */
    libecs::PolymorphVector getDMInfo() const
    {
        return theSimulatorImplementation->getDMInfo();
    }

    /**
       Retrieve the list of the specified DM's properties.
       @param aClassname the name of the DM from which the property list is
                         retrieved.
       @return a tuple of the properties.
     */
    PolymorphMap getPropertyInfo( libecs::String const& aClassname ) const
    {
        return theSimulatorImplementation->getPropertyInfo( aClassname );
    }

    /**
       Retrieve the meta information of a specific DM.
       @param aClassname the name of the DM to retrieve the information of.
       @return the meta information keyed by strings.
     */
    PolymorphMap getClassInfo( libecs::String const& aClassname ) const 
    {

        return theSimulatorImplementation->getClassInfo( aClassname );
    }

    char getDMSearchPathSeparator() const
    {
        return theSimulatorImplementation->getDMSearchPathSeparator();
    }

    void setDMSearchPath( const std::string& dmSearchPath )
    {
        theSimulatorImplementation->setDMSearchPath( dmSearchPath );
    }

    std::string getDMSearchPath() const
    {
        return theSimulatorImplementation->getDMSearchPath();
    }

    /** @} */

private:

    SimulatorImplementation* theSimulatorImplementation;

};


} // namespace libemc

#endif   /* __SIMULATOR_HPP */
