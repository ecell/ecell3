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


#ifndef __SIMULATOR_HPP
#define __SIMULATOR_HPP

#include "libecs/libecs.hpp"
#include "libecs/EntityType.hpp"

#include "libemc.hpp"
#include "EmcLogger.hpp"
#include "SimulatorImplementation.hpp"

namespace libemc
{
  
  /** @defgroup libemc_module The Libemc Module 
   * This is the libemc module 
   * @{ 
   */ 
  

  /**
     The interface to the simulator.

     Simulator class provides a unified interface to the libecs, 
     C++ library for cell modeling and simulation.
     
     @see libecs
     @see Model
     @see SimulatorImplementation
  */

  class Simulator
  {

  public:

    Simulator();
    virtual ~Simulator() {}

    /**
       Create a new Stepper in the model.

       @param aClassname a classname of the Stepper to create.
       @param anID       an ID of the Stepper.
       @param aParameterList a list of parameters to give to the Stepper
    */

    void createStepper( libecs::StringCref          aClassname,
			libecs::StringCref          anId )
    {
      theSimulatorImplementation->createStepper( aClassname, anId );
    }


    /**
       Set a property value of an Stepper.

       @param aStepperID    the Stepper ID.
       @param aValue        the value to set as a UVariableVector.
    */

    void setStepperProperty( libecs::StringCref          aStepperID,
			     libecs::StringCref          aPropertyName,
			     libecs::UVariableVectorCref aValue )
    {
      theSimulatorImplementation->setStepperProperty( aStepperID,
						      aPropertyName,
						      aValue );
    }

    /**
       Get a value of a property from an Stepper.

       @param aStepperID the Stepper ID.
       @param aPropertyName the name of the property.
       @return the property value as a reference counted pointor of a 
       UVariableVector.
    */

    const libecs::UVariableVectorRCPtr
    getStepperProperty( libecs::StringCref aStepperID,
			libecs::StringCref aPropertyName )
    {
      return theSimulatorImplementation->getStepperProperty( aStepperID,
							     aPropertyName );
    }

    /**
       Create a new Entity in the model.

       @param aClassname a classname of the Entity to create.
       @param aFullIDString FullID of the Entity as a String.
       @param aName      a name of the Entity.
    */

    void createEntity( libecs::StringCref           aClassname, 
		       libecs::StringCref           aFullIDString,
		       libecs::StringCref           aName )
    {
      theSimulatorImplementation->createEntity( aClassname,
						aFullIDString,
						aName );
    }

    /**
       Set a property value of an Entity.

       @param aFullPNString a FullPN of the Property to set as a String.
       @param aValue        the value to set as a UVariableVector.
    */

    void setProperty( libecs::StringCref            aFullPNString,
		      libecs::UVariableVectorCref   aValue )
    {
      theSimulatorImplementation->setProperty( aFullPNString,
					       aValue );
    }

    /**
       Get a value of a property from an Entity.

       @param aFullPNString a FullPN of the property as a String.
       @return the property value as a reference counted pointor of a 
       UVariableVector
    */

    const libecs::UVariableVectorRCPtr
    getProperty( libecs::StringCref aFullPNString )
    {
      return theSimulatorImplementation->getProperty( aFullPNString );
    }

    /**
       Get or create a Logger.

       @param aFullPNString a FullPN of the PropertySlot which the Logger is
       observing, as a String 

       @return a borrowed pointer to the Logger
    */

    EmcLogger getLogger( libecs::StringCref aFullPNString )
    {
      return theSimulatorImplementation->getLogger( aFullPNString );
    }

    /**
       Conduct a step of the simulation.

    */

    void step()
    {
      theSimulatorImplementation->step();
    }

    /**
       Initialize the simulator.

       This method is automatically called before step() and run().
       Usually no need to call this explicitly.
    */

    void initialize()
    {
      theSimulatorImplementation->initialize();
    }


    /**
       Get current time of the simulator.

       @return current time of the simulator
    */

    const libecs::Real getCurrentTime()
    {
      return theSimulatorImplementation->getCurrentTime();
    }

    /**
       List Loggers in the simulator.

       @return a list of Loggers in a reference counted pointer 
       to a StringVector
    */

    libecs::StringVectorRCPtr getLoggerList()
    {
      return theSimulatorImplementation->getLoggerList();
    }

    /**
       Run the simulation.

       @note Both the PendingEventChecker and the EventHandler must be set 
       before calling this method.

       @see setPendingEventChecker
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

       The event checker must be a subclass of PendingEventChecker class.

       This is usually used to set to form a mainloop of GUI toolkit.
       If you are using gtk, the event checker would call gtk_events_pending()
       function.

       While the simulation is running by the run() method, the function
       object given by this method is called once in several simulation 
       steps.  If it returns true, the EventHandler given by setEventHandler()
       method is called.

       @param aPendingEventChecker a function object of the event checker
       @see PendingEventChecker
       @see setEventHandler
    */

    void setPendingEventChecker( PendingEventCheckerPtr aPendingEventChecker )
    {
       theSimulatorImplementation->
	 setPendingEventChecker( aPendingEventChecker );
     }

    /**
       Set an event handler.

       The event handler must be a subclass of EventHandler class.

       If you are using gtk, it would call gtk_main_iteration() function.

       @param anEventHandler a function object of the event handler
       @see EventHandler
       @see setPendingEventChecker
    */

    void setEventHandler( EventHandlerPtr anEventHandler )
    {
       theSimulatorImplementation->setEventHandler( anEventHandler );
    }

  private:

    SimulatorImplementation* theSimulatorImplementation;

  };

  /** @} */ //end of libemc_module 

} // namespace libemc

#endif   /* __SIMULATOR_HPP */





