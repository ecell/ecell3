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

//#include <queue>
#include "DynamicPriorityQueue.hpp"

#include "Stepper.hpp"


namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 


  DECLARE_MAP( const String, StepperPtr, std::less< const String >,
	       StepperMap ); 

  typedef std::pair<Real,StepperPtr> RealStepperPtrPair;
  DECLARE_TYPE( RealStepperPtrPair, Event );

  //typedef std::priority_queue<Event,std::vector<Event>,std::greater<Event> >
  //    EventPriorityQueue_less;
  //  DECLARE_TYPE( EventPriorityQueue_less, ScheduleQueue );

  typedef DynamicPriorityQueue<Event> EventDynamicPriorityQueue;
  DECLARE_TYPE( EventDynamicPriorityQueue, ScheduleQueue );


  /**
     Model class represents a simulation model.

     Model has a list of Steppers and a pointer to the root system.

     This also works as a global scheduler with a heap-tree based
     priority queue.

     In addition to theStepperMap, this class has theScheduleQueue.
     theStepperMap stores all the Steppers in the model.  theScheduleQueue
     is basically a priority queue used for scheduling, of which containee
     is synchronized with the StepperMap by resetScheduleQueue() method.  

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
    */

    void initialize();

    /**
       Conducts a step of the simulation.

       This method picks a Stepper on the top of theScheduleQueue,
       calls sync(), step(), and push() of the Stepper, and
       reschedules it on the queue.

    */

    void step();


    /**
       Returns the current time.

       \return time elasped since start of the simulation.
    */

    const Real getCurrentTime() const
    {
      return theCurrentTime;
    }


    /**
       Creates a new Entity object and register it in an appropriate System
       in  the Model.

       \param aClassname
       \param aFullID
       \param aName
    */

    void createEntity( StringCref aClassname,
		       FullIDCref aFullID,
		       StringCref aName );


    /**
       This method finds an Entity object pointed by the FullID.

       \param aFullID a FullID of the requested Entity.
       \return A borrowed pointer to an Entity specified by the FullID.
    */

    EntityPtr getEntity( FullIDCref aFullID );

    /**
       This method finds a System object pointed by the SystemPath.  


       \param aSystemPath a SystemPath of the requested System.
       \return A borrowed pointer to a System.
    */


    SystemPtr getSystem( SystemPathCref aSystemPath );


    /**
       Create a stepper with an ID and a classname. 

       \param aClassname  a classname of the Stepper to create.  

       \param anID        a Stepper ID string of the Stepper to create.  

       \param aParameters a UVariableVector of parameters to give to
       the created Stepper.
    */

    void createStepper( StringCref aClassname, 
			StringCref anID,
			UVariableVectorCref aParameterList );



    /**
       Get a stepper by an ID.

       \param anID a Stepper ID string of the Stepper to get.
       \return a borrowed pointer to the Stepper.
    */

    StepperPtr getStepper( StringCref anID );


    /**
       Flush the data in all Loggers immediately.

       Usually Loggers record data with logging intervals.  This method
       orders every Logger to write the data immediately ignoring the
       logging interval.

    */

    void flushLogger();


    /**
       Get the RootSystem.

       \return a borrowed pointer to the RootSystem.
    */

    SystemPtr getRootSystem() const
    {
      return theRootSystemPtr;
    }


    /**
       Get the LoggerBroker.

       \return a borrowed pointer to the LoggerBroker.
    */

    LoggerBrokerRef getLoggerBroker()
    { 
      return theLoggerBroker; 
    }


    /// \internal

    StepperMakerRef     getStepperMaker()     { return theStepperMaker; }

    /// \internal

    ReactorMakerRef     getReactorMaker()     { return theReactorMaker; }

    /// \internal

    SubstanceMakerRef   getSubstanceMaker()   { return theSubstanceMaker; }

    /// \internal

    SystemMakerRef      getSystemMaker()      { return theSystemMaker; }

    /// \internal

    AccumulatorMakerRef getAccumulatorMaker() { return theAccumulatorMaker; }


    StringLiteral getClassName() const  { return "Model"; }


  private:

    /**
       This method checks recursively if all systems have Steppers
       connected.

       \param aSystem a root node to start recursive search.
       
       \throw InitializationFailed if the check failed.
    */

    void checkStepper( SystemCptr aSystem );


    /**
       This method clears the ScheduleQueue and reconstruct the queue
       from the StepperMap.
    */

    void resetScheduleQueue();

  private:

    Real                theCurrentTime;

    StepperMap          theStepperMap;
    ScheduleQueue       theScheduleQueue;

    SystemPtr           theRootSystemPtr;


    LoggerBrokerRef     theLoggerBroker;

    StepperMakerRef     theStepperMaker;

    SystemMakerRef      theSystemMaker;
    SubstanceMakerRef   theSubstanceMaker;
    ReactorMakerRef     theReactorMaker;

    AccumulatorMakerRef theAccumulatorMaker;


  };



} // namespace libecs




#endif /* __STEPPERLEADER_HPP */




/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

