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

#ifndef __MODEL_HPP
#define __MODEL_HPP

#include <map>
#include "libecs.hpp"

#include "Scheduler.hpp"

namespace libecs
{

  /** @addtogroup model The Model.

      The model.

      @ingroup libecs
      @{ 
   */ 

  /** @file */


  DECLARE_MAP( const String, StepperPtr, std::less< const String >,
	       StepperMap ); 

  /**
     Model class represents a simulation model.

     Model has a list of Steppers and a pointer to the root system.

  */

  class Model
  {

  public:

    Model();
    ~Model();

    /**
       Initialize the whole model.

       This method must be called before running the model, and when
       structure of the model is changed.

       Procedure of the initialization is as follows:

       1. Initialize Systems recursively starting from theRootSystem.
       ( System::initialize() )
       1. Check if all the Systems have a Stepper.
       1. Initialize Steppers. ( Stepper::initialize() )
       1. Construct Stepper interdependency graph 
       ( Stepper::updateDependentStepperVector() )

    */

    void initialize();

    /**
       Conduct a step of the simulation.

       @see Scheduler
    */

    void step()
    {
      theScheduler.step();
    }


    void reschedule( StepperPtr const aStepperPtr )
    {
      theScheduler.reschedule( aStepperPtr );
    }


    /**
       Returns the current time.

       @return time elasped since start of the simulation.
    */

    const Real getCurrentTime() const
    {
      return theScheduler.getCurrentTime();
    }


    /**
       Creates a new Entity object and register it in an appropriate System
       in  the Model.

       @param aClassname
       @param aFullID
       @param aName
    */

    void createEntity( StringCref aClassname, FullIDCref aFullID );


    /**
       This method finds an Entity object pointed by the FullID.

       @param aFullID a FullID of the requested Entity.
       @return A borrowed pointer to an Entity specified by the FullID.
    */

    EntityPtr getEntity( FullIDCref aFullID ) const;

    /**
       This method finds a System object pointed by the SystemPath.  


       @param aSystemPath a SystemPath of the requested System.
       @return A borrowed pointer to a System.
    */


    SystemPtr getSystem( SystemPathCref aSystemPath ) const;


    /**
       Create a stepper with an ID and a classname. 

       @param aClassname  a classname of the Stepper to create.  

       @param anID        a Stepper ID string of the Stepper to create.  

    */

    void createStepper( StringCref aClassname, StringCref anID );


    /**
       Get a stepper by an ID.

       @param anID a Stepper ID string of the Stepper to get.
       @return a borrowed pointer to the Stepper.
    */

    StepperPtr getStepper( StringCref anID ) const;


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

    void flushLogger();


    /**
       Get the RootSystem.

       @return a borrowed pointer to the RootSystem.
    */

    SystemPtr getRootSystem() const
    {
      return theRootSystemPtr;
    }


    /**
       Get the LoggerBroker.

       @return a borrowed pointer to the LoggerBroker.
    */

    LoggerBrokerRef getLoggerBroker() const
    { 
      return theLoggerBroker; 
    }

    /// @internal

    StepperMakerRef     getStepperMaker()     { return theStepperMaker; }

    /// @internal

    ProcessMakerRef     getProcessMaker()     { return theProcessMaker; }

    /// @internal

    VariableMakerRef    getVariableMaker()    { return theVariableMaker; }

    /// @internal

    SystemMakerRef      getSystemMaker()      { return theSystemMaker; }

    /// @internal

    //    AccumulatorMakerRef getAccumulatorMaker() { return theAccumulatorMaker; }


    StringLiteral getClassName() const  { return "Model"; }


  private:

    /**
       This method checks recursively if all systems have Steppers
       connected.

       @param aSystem a root node to start recursive search.
       
       @throw InitializationFailed if the check failed.
    */

    void checkStepper( SystemCptr const aSystem ) const;

    static void initializeSystems( SystemPtr const aSystem );

  private:

    SystemPtr           theRootSystemPtr;

    StepperMap          theStepperMap;

    Scheduler           theScheduler;

    LoggerBrokerRef     theLoggerBroker;


    StepperMakerRef     theStepperMaker;
    SystemMakerRef      theSystemMaker;
    VariableMakerRef   theVariableMaker;
    ProcessMakerRef     theProcessMaker;
    //    AccumulatorMakerRef theAccumulatorMaker;


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

