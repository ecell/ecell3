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

#ifndef __MODEL_HPP
#define __MODEL_HPP

#include "dmtool/ModuleMaker.hpp"

#include "libecs/Defs.hpp"
#include "libecs/AssocVector.h"
#include "libecs/EcsObjectMaker.hpp"
#include "libecs/EventScheduler.hpp"
#include "libecs/StepperEvent.hpp"
#include "libecs/LoggerBroker.hpp"
#include "libecs/Stepper.hpp"
#include "libecs/SystemStepper.hpp"
#include "libecs/Handle.hpp"

namespace libecs
{

class Entity;

/**
   Model class represents a simulation model.

   Model has a list of Steppers and a pointer to the root system.
*/
class LIBECS_API Model
{
public:
    typedef Loki::AssocVector<String, Stepper*, std::less<String> > StepperMap;

protected:
    typedef EventScheduler< StepperEvent > StepperEventScheduler;
    typedef StepperEventScheduler::EventIndex EventIndex;
    typedef EcsObjectMaker< Stepper > StepperMaker;
    typedef EcsObjectMaker< System > SystemMaker;
    typedef EcsObjectMaker< Variable > VariableMaker;
    typedef EcsObjectMaker< Process > ProcessMaker;
    typedef std::map< Handle, EcsObject* > HandleToObjectMap;

public:
    Model( ModuleMaker< EcsObject >& maker );

    void setup();

    virtual ~Model();
    /**
       Initialize the whole model.

       This method must be called before running the model, and when
       structure of the model is changed.

       Procedure of the initialization is as follows:

       1. Initialize Systems recursively starting from theRootSystem.
              ( System::initialize() )
       2. Check if all the Systems have a Stepper each.
       3. Initialize Steppers. ( Stepper::initialize() )
       4. Construct Stepper interdependency graph 
              ( Stepper::updateDependentStepperVector() )


       @throw InitializationFailed
    */
    void initialize();


    /**
       Conduct a step of the simulation.

       @see Scheduler
    */
    void step();


    /**
       Get the next event to occur on the scheduler.
     */
    StepperEvent const& getTopEvent() const
    {
        return theScheduler.getTopEvent();
    }


    /**
       Returns the current time.

       @return time elasped since start of the simulation.
    */
    Real getCurrentTime() const
    {
        return theCurrentTime;
    }


    Stepper* getLastStepper() const
    {
        return theLastStepper;
    }


    /**
       Get the property interface for the specified class.

       @param aClassname
    */
    PropertyInterfaceBase const& getPropertyInterface(
            String const& aClassname ) const;

    /**
       Retrieves an object by the handle.

       @param handle the handle of the requested object.
       @return A borrowed pointer to the object.
    */
    EcsObject* getObject( Handle const& handle ) const;


    /**
       Creates a new, unbound variable object. 
       One should register the object to an appropriate system to use it in
       the model.

       @param aClassname the class name of the variable.
     */
    Variable* createVariable( String const& aClassname );


    /**
       Creates a new, unbound procss object. 
       One should register the object to an appropriate system to use it in
       the model.

       @param aClassname the class name of the variable.
     */
    Process* createProcess( String const& aClassname );


    /**
       Creates a new, unbound system object. 
       One should register the object to an appropriate system to use it in
       the model.

       @param aClassname the class name of the variable.
     */
    System* createSystem( String const& aClassname );


    /**
       Creates a new Entity object and register it in an appropriate System
       in the Model.

       @param aClassname
       @param aFullID
       @param aName
    */
    Entity* createEntity( String const& aClassname, FullID const& aFullID );


    /**
       detach an EcsObject from the Model
    */
    void detachObject( EcsObject* anObject );

    /**
       Retrieves an Entity object pointed by the FullID.

       @param aFullID a FullID of the requested Entity.
       @return A borrowed pointer to the Entity specified by the FullID.
    */
    Entity* getEntity( FullID const& aFullID ) const;


    /**
       Delete a Entity with aFullID.

       @param aFullID      a FullID of the Entity to delete.
     */
    void deleteEntity( FullID const& aFullID );

    /**
       Retrieves a System object pointed by the SystemPath.    

       @param aSystemPath a SystemPath of the requested System.
       @return A borrowed pointer to the System.
    */
    System* getSystem( SystemPath const& aSystemPath ) const;

    /**
       Create a new, unbound stepper of aClassname.
       One should register the object to use it in the model.

       @param aClassname  a classname of the Stepper to create.    
    */
    Stepper* createStepper( String const& aClassname );


    /**
       Register a stepper to the model
     */
    void registerStepper( Stepper* aStepper );

    /**
       Create a stepper with an ID and a classname. 

       @param aClassname  a classname of the Stepper to create.    
       @param anID        a Stepper ID string of the Stepper to create.    
    */
    Stepper* createStepper( String const& aClassname, String const& anID );


    /**
       Delete a stepper with anID.
       Removal of a stepper connected to any Entity is not allowed
       and such an attempt causes exception.

       @param anID        a Stepper ID string of the Stepper to delete.
     */
    void deleteStepper( String const& aClassname );

    /**
       Get a stepper by an ID.

       @param anID a Stepper ID string of the Stepper to get.
       @return a borrowed pointer to the Stepper.
    */
    Stepper* getStepper( String const& anID ) const;

    /**
       Get the StepperMap of this Model.

       @return the const reference of the StepperMap.
    */
    StepperMap const& getStepperMap() const
    {
        return theStepperMap;
    }


    /**
       Flush the data in all Loggers immediately.

       Usually Loggers record data with logging intervals.    This method
       orders every Logger to write the data immediately ignoring the
       logging interval.
    */
    void flushLoggers();


    /**
       Get the RootSystem.

       @return a borrowed pointer to the RootSystem.
    */
    System* getRootSystem() const
    {
        return theRootSystem;
    }


    SystemStepper* getSystemStepper()
    {
        return &theSystemStepper;
    }

    /**
       Get the LoggerBroker.

       @return a borrowed pointer to the LoggerBroker.
    */
    LoggerBroker& getLoggerBroker()
    { 
        return theLoggerBroker; 
    }


    LoggerBroker const& getLoggerBroker() const
    { 
        return theLoggerBroker; 
    }


    StepperEventScheduler& getScheduler()
    {
        return theScheduler;
    }


    StepperEventScheduler const& getScheduler() const
    {
        return theScheduler;
    }

    void setDMSearchPath( String const& path );

    String getDMSearchPath() const;

    void markDirty();

    bool isDirty() const
    {
        return this->isDirty_;
    }

private:
    /** @internal */
    void registerBuiltinModules();

    void checkSizeVariable( System const* aSystem );

    Handle generateNextHandle();

    /**
       This method checks recursively if all systems have Steppers
       connected.

       @param aSystem a root node to start recursive search.
       
       @throw InitializationFailed if the check is failed.
    */
    static void preinitializeEntities( System* aSystem );

    static void initializeEntities( System* aSystem );

    static void removeVariableReferences( System* aSystem, Variable const* aVariable );

public:
    static const char PATH_SEPARATOR;

protected:

    Time                            theCurrentTime;
    Stepper*                        theLastStepper;

    StepperEventScheduler           theScheduler;

    LoggerBroker                    theLoggerBroker;

    HandleToObjectMap               theObjectMap;
    unsigned int                    theNextHandleVal;

    System*                         theRootSystem;

    SystemStepper                   theSystemStepper;

    StepperMap                      theStepperMap;

    ModuleMaker< EcsObject >&       theEcsObjectMaker;
    StepperMaker                    theStepperMaker;
    SystemMaker                     theSystemMaker;
    VariableMaker                   theVariableMaker;
    ProcessMaker                    theProcessMaker;
    bool                            isDirty_;
};

} // namespace libecs

#endif /* __MODEL_HPP */
