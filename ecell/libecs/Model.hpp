//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

#include "AssocVector.h"

#include "libecs.hpp"

#include "EventScheduler.hpp"
#include "StepperEvent.hpp"
#include "StepperMaker.hpp"
#include "VariableMaker.hpp"
#include "ProcessMaker.hpp"
#include "SystemMaker.hpp"
#include "LoggerBroker.hpp"
#include "Stepper.hpp"
#include "SystemStepper.hpp"

namespace libecs
{

  /** @addtogroup model The Model.

      The model.

      @ingroup libecs
      @{ 
   */ 

  /** @file */


  DECLARE_ASSOCVECTOR( String, StepperPtr, std::less< const String >,
		       StepperMap ); 


  /**
     Model class represents a simulation model.

     Model has a list of Steppers and a pointer to the root system.

  */

  class Model
  {

  protected:

    typedef EventScheduler<StepperEvent> StepperEventScheduler;
    typedef StepperEventScheduler::EventIndex EventIndex;

  public:

    LIBECS_API Model();
    LIBECS_API ~Model();

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

    LIBECS_API void initialize();

    /**
       Conduct a step of the simulation.

       @see Scheduler
    */

    bool getRunningFlag() const
    {
      return theRunningFlag;
    }

    // MINE

    bool getGlobalDirtyState() const
    {
      return (this->getDirtyBit() || this->getLoggerBroker().getDirtyBit() );
    }

    void clearGlobalDirtyState()
    {
      clearUninitialized();
      clearDirtyBit();
      getLoggerBroker().clearDirtyBit();
      
    }

    bool getDirtyBit() const
    {
      return this->theDirtyBit;
    }

    void setDirtyBit()
    {
      theDirtyBit = true;
    }

    void clearDirtyBit()
    {
      theDirtyBit = false;
    }

    // ENDM

    void step()
    {

      if (!theRunningFlag)
        {
          this->initialize();
          theRunningFlag = true;
        }
      
      if ( this->getGlobalDirtyState() )
        {
          this->initialize();
        }
      
      StepperEventCref aNextEvent( theScheduler.getTopEvent() );
      theCurrentTime = aNextEvent.getTime();
      theLastStepper = aNextEvent.getStepper();

      theScheduler.step();
    }


    /**
       Get the next event to occur on the scheduler.

     */

    const StepperEvent& getTopEvent() const
    {
      return theScheduler.getTopEvent();
    }


    /**
       Returns the current time.

       @return time elasped since start of the simulation.
    */

    const Real getCurrentTime() const
    {
      return theCurrentTime;
    }


    const StepperPtr getLastStepper() const
    {
      return theLastStepper;
    }

    /**
       Creates a new Entity object and register it in an appropriate System
       in  the Model.

       @param aClassname
       @param aClassType
    */

    LIBECS_API PolymorphMap getClassInfo( StringCref aClassType, StringCref aClassname, Integer forceReload );

    /**
       Creates a new Entity object and register it in an appropriate System
       in  the Model.

       @param aClassname
       @param aFullID
       @param aName
    */

    LIBECS_API void createEntity( StringCref aClassname, FullIDCref aFullID );

    /**
       This method finds an Entity object pointed by the FullID.

       @param aFullID a FullID of the requested Entity.
       @return A borrowed pointer to an Entity specified by the FullID.
    */

    LIBECS_API EntityPtr getEntity( FullIDCref aFullID ) const;

    /**
       This method finds a System object pointed by the SystemPath.  


       @param aSystemPath a SystemPath of the requested System.
       @return A borrowed pointer to a System.
    */


    LIBECS_API SystemPtr getSystem( SystemPathCref aSystemPath ) const;;


    /**
       Create a stepper with an ID and a classname. 

       @param aClassname  a classname of the Stepper to create.  

       @param anID        a Stepper ID string of the Stepper to create.  

    */

    LIBECS_API void createStepper( StringCref aClassname, StringCref anID );


    /**
       Get a stepper by an ID.

       @param anID a Stepper ID string of the Stepper to get.
       @return a borrowed pointer to the Stepper.
    */

    LIBECS_API StepperPtr getStepper( StringCref anID ) const;


    /**
       Get the StepperMap of this Model.

       @return the const reference of the StepperMap.
    */

    StepperMapCref getStepperMap() const
    {
      return theStepperMap;
    }




    /**
       Flush the data in all Loggers immediately.

       Usually Loggers record data with logging intervals.  This method
       orders every Logger to write the data immediately ignoring the
       logging interval.

    */

    LIBECS_API void flushLoggers();


    /**
       Get the RootSystem.

       @return a borrowed pointer to the RootSystem.
    */

    SystemPtr getRootSystem() const
    {
      return theRootSystemPtr;
    }

    SystemStepperPtr getSystemStepper()
    {
      return &theSystemStepper;
    }


    /**
       Get the LoggerBroker.

       @return a borrowed pointer to the LoggerBroker.
    */

    LoggerBrokerRef getLoggerBroker()
    { 
      return theLoggerBroker; 
    }

    LoggerBrokerCref getLoggerBroker() const
    { 
      return theLoggerBroker; 
    }


    StepperEventScheduler&   getScheduler() { return theScheduler; }

    /// @internal
    StepperMakerRef     getStepperMaker()     { return theStepperMaker; }

    /// @internal

    ProcessMakerRef     getProcessMaker()     { return theProcessMaker; }

    /// @internal

    VariableMakerRef    getVariableMaker()    { return theVariableMaker; }

    /// @internal

    SystemMakerRef      getSystemMaker()      { return theSystemMaker; }


  private:

    void clearUninitialized()
    {
      uninitializedSteppers.clear();
      uninitializedSystems.clear();
      uninitializedVariables.clear();
      uninitializedProcesses.clear();
    }

    void recordUninitializedVariable( VariablePtr aVariablePtr );
    void recordUninitializedSystem( SystemPtr aSystemPtr );
    void recordUninitializedProcess( ProcessPtr aProcessPtr );
    void recordUninitializedStepper( StepperPtr aStepperPtr );

    void staticInitialize();
    void runningInitialize();
    
    void constructEntity( StringCref aClassname, FullIDCref aFullID );

    /**
       This method checks recursively if all systems have Steppers
       connected.

       @param aSystem a root node to start recursive search.
       
       @throw InitializationFailed if the check is failed.
    */

    void checkStepper( SystemCptr const aSystem ) const;

    void checkRootSystemSizeVariable();

    static void initializeSystems( SystemPtr const aSystem );

  private:

    Time                theCurrentTime;
    StepperPtr          theLastStepper;

    StepperEventScheduler theScheduler;

    LoggerBroker        theLoggerBroker;

    System              *theRootSystemPtr;
 
    SystemStepper       theSystemStepper;

    StepperMap          theStepperMap;

    StepperMaker        theStepperMaker;
    SystemMaker         theSystemMaker;
    VariableMaker       theVariableMaker;
    ProcessMaker        theProcessMaker;

    bool theRunningFlag;
    bool theDirtyBit;

    StepperVector uninitializedSteppers;
    SystemVector uninitializedSystems;
    VariableVector uninitializedVariables;
    ProcessVector uninitializedProcesses;
    

  };

  
  /*@}*/

} // namespace libecs




#endif /* __STEPPERLEADER_HPP */




/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

