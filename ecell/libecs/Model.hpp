//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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


  //FIXME: should be merged with PropertySlot::SlotTypes

  template <class T,typename Ret>
  class VoidObjectMethod
  {
    typedef Ret (T::* Method )( void ) const;

  public:

    VoidObjectMethod( T& anObject, Method aMethod )
      :
      theObject( anObject ),
      theMethod( aMethod )
    {
      ; // do nothing
    }

    const Ret operator()( void ) const 
    {
      return ( theObject.*theMethod )();
    }

  private:

    T&     theObject;
    Method theMethod;

  };

  typedef VoidObjectMethod< Model, const Real > GetCurrentTimeMethodType;


  /**
     

     This also work as a global scheduler with a heap-tree based
     priority queue.

     Model has two containers; theStepperMap and theScheduleQueue.

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
       Get a stepper by ID.
    */

    StepperPtr getStepper( StringCref anID );

    /**
       Create a stepper with an ID and a classname.

    */

    void createStepper( StringCref aClassName, 
			StringCref anID,
			UVariableVectorCref aMessage );

    void initialize();

    const Real getCurrentTime() const
    {
      return theCurrentTime;
    }

    SystemPtr getRootSystem() const
    {
      return theRootSystemPtr;
    }

    void step();


    /**
       This method finds a System object pointed by the @a SystemPath.  

       @return An pointer to a System object that is pointed by @a SystemPath
    */

    SystemPtr getSystem( SystemPathCref aSystemPath );

    /**
       This method finds an Entity object pointed by the @a FullID.

       @return An pointer to an Entity object that is pointed by @a FullID
    */

    EntityPtr getEntity( FullIDCref aFullID );


    void createEntity( StringCref aClassname,
		       FullIDCref aFullID,
		       StringCref aName );


    LoggerBrokerRef getLoggerBroker()
    { 
      return theLoggerBroker; 
    }

    StepperMakerRef     getStepperMaker()     { return theStepperMaker; }

    ReactorMakerRef     getReactorMaker()     { return theReactorMaker; }
    SubstanceMakerRef   getSubstanceMaker()   { return theSubstanceMaker; }
    SystemMakerRef      getSystemMaker()      { return theSystemMaker; }
    AccumulatorMakerRef getAccumulatorMaker() { return theAccumulatorMaker; }


    StringLiteral getClassName() const  { return "Model"; }


  private:

    void checkStepper( SystemCptr aSystem );

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

