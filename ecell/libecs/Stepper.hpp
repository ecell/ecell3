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

#ifndef ___STEPPER_H___
#define ___STEPPER_H___
#include <queue>
#include <vector>
#include <map>
#include <algorithm>
#include <utility>

#include "libecs.hpp"

#include "System.hpp"



namespace libecs
{

  DECLARE_CLASS( Euler1Stepper );
  DECLARE_CLASS( RungeKutta4Stepper );
  DECLARE_CLASS( PropertySlotVectorImplementation );

  DECLARE_VECTOR( SubstancePtr, SubstanceVector );
  DECLARE_VECTOR( ReactorPtr,   ReactorVector );
  DECLARE_VECTOR( SystemPtr,    SystemVector );


  typedef IntegratorPtr ( *IntegratorAllocator_ )( SubstanceRef );
  DECLARE_TYPE( IntegratorAllocator_, IntegratorAllocator );

  typedef StepperPtr (* StepperAllocator_ )();
  DECLARE_TYPE( StepperAllocator_, StepperAllocator );

  DECLARE_VECTOR( StepperPtr, StepperVector );
  DECLARE_VECTOR( SlaveStepperPtr, SlaveStepperVector );
  DECLARE_VECTOR( MasterStepperPtr, MasterStepperVector );

  DECLARE_VECTOR( PropertySlotPtr, PropertySlotVector );

  //  DECLARE_LIST( MasterStepperPtr, MasterStepperList )


  typedef std::pair<Real,MasterStepperPtr> RealMasterStepperPtrPair;
  DECLARE_TYPE( RealMasterStepperPtrPair, Event );
  typedef std::priority_queue<Event,std::vector<Event>,std::greater<Event> >
  EventPriorityQueue_less;
  DECLARE_TYPE( EventPriorityQueue_less, ScheduleQueue );

  class StepperLeader
  {




  public:

    StepperLeader();
    virtual ~StepperLeader() {}

    virtual void initialize();

    RealCref getCurrentTime() const
    {
      return theCurrentTime;
    }

    void setRootSystem( const RootSystemPtr aRootSystem)
    {
      theRootSystem = aRootSystem;
    }

    RootSystemPtr getRootSystem() const
    {
      return theRootSystem;
    }

    void step();
    void push();


    virtual const char* const className() const  { return "StepperLeader"; }

  private:

    void updateMasterStepperVector( SystemPtr aSystem );
    void updateScheduleQueue();

  private:

    RootSystemPtr   theRootSystem;
    Real            theCurrentTime;

    MasterStepperVector   theMasterStepperVector;
    ScheduleQueue   theScheduleQueue;

  };



  class Stepper
  {

  public:

    Stepper(); 
    virtual ~Stepper() {}

    void setOwner( SystemPtr owner ) { theOwner = owner; }
    SystemPtr getOwner() const { return theOwner; }

    virtual void initialize();

    virtual void setStepInterval( RealCref aStepInterval ) = 0;
    virtual RealCref getStepInterval() const = 0;
    virtual RealCref getStepsPerSecond() const = 0;

    void setMasterStepper( MasterStepperPtr aMasterStepper )
    { 
      theMasterStepper = aMasterStepper;
    }

    MasterStepperPtr getMasterStepper() const
    {
      return theMasterStepper;
    }

    virtual void sync() { }
    virtual const Real step() { }
    virtual void push() { }

    virtual void registerPropertySlot( PropertySlotPtr ) = 0;

    virtual const char* const className() const  { return "Stepper"; }

  protected:

    SystemPtr        theOwner;
    MasterStepperPtr theMasterStepper;

  };

  class MasterStepper 
    : 
    public Stepper
  {

  public:

    MasterStepper();
    virtual ~MasterStepper() {}

    virtual void sync();
    virtual const Real step() = 0;
    virtual void push();

    /**

       This may be overridden in dynamically scheduled steppers.

     */

    void setStepInterval( RealCref aStepInterval );

    void calculateStepsPerSecond();

    virtual RealCref getStepInterval() const
    {
      return theStepInterval;
    }

    virtual RealCref getStepsPerSecond() const
    {
      return theStepsPerSecond;
    }

    bool isEntityListChanged() const
    {
      return theEntityListChanged;
    }

    void setEntityListChanged()
    {
      theEntityListChanged = true;
    }

    void clearEntityListChanged()
    {
      theEntityListChanged = false;
    }

    virtual void initialize();

    SlaveStepperVectorCref getSlaveStepperVector() const
    {
      return theSlaveStepperVector;
    }

    void updateSlaveStepperVector( SystemPtr aStartSystemPtr );

    void registerSlaves( SystemPtr );
    void registerPropertySlot( PropertySlotPtr );


    virtual const char* const className() const  { return "MasterStepper"; }

  protected:

    Real                theStepInterval;
    Real                theStepsPerSecond;

    IntegratorAllocator theAllocator;
    SlaveStepperVector  theSlaveStepperVector;
    PropertySlotVector  thePropertySlotVector;

    bool                theEntityListChanged;
  };



  class SlaveStepper 
    : 
    public Stepper
  {

  public:

    SlaveStepper() {}
    virtual ~SlaveStepper() {}

  
    virtual void initialize()
    {
      Stepper::initialize();
    }

    void setMasterStepper( MasterStepperPtr aMasterStepperPtr )
    {
      theMasterStepper = aMasterStepperPtr;
    }

    void setStepInterval( RealCref aStepInterval )
    {
      // Slaves are synchronous to their masters.
      theMasterStepper->setStepInterval( aStepInterval ); 
    }

    virtual RealCref getStepInterval() const
    { 
      // Slaves are synchronous to their masters.
      return theMasterStepper->getStepInterval(); 
    }

    virtual RealCref getStepsPerSecond() const
    { 
      // Slaves are synchronous to their masters.
      return theMasterStepper->getStepsPerSecond();
    }
    
    void registerPropertySlot( PropertySlotPtr aPropertySlotPtr )
    { 
      theMasterStepper->registerPropertySlot( aPropertySlotPtr );
    }

    static StepperPtr instance() { return new SlaveStepper; }

    virtual const char* const className() const  { return "SlaveStepper"; }

  private:

    MasterStepperPtr theMasterStepper;

  };


  class MasterStepperWithEntityCache
    : 
    public MasterStepper
  {

  public:

    MasterStepperWithEntityCache()
    {
      ; // do nothing
    }

    ~MasterStepperWithEntityCache() {}


    virtual void initialize();

    /**
       Update the cache if any of the systems have any newly added 
       or removed Entity.
    */

    void updateCacheWithCheck()
    {
      if( isEntityListChanged() )
	{
	  updateCache();
	  clearEntityListChanged();
	}
    }

    /**
       Update the cache.
    */

    void updateCache();

    void updateCacheWithSort();

  protected:

    SubstanceVector               theSubstanceCache;
    ReactorVector                 theReactorCache;
    SystemVector                  theSystemCache;

  };


  class SRMStepper 
    : 
    public MasterStepperWithEntityCache
  {
  public:

    SRMStepper();
    virtual ~SRMStepper() {}

    virtual const Real step()
    {
      clear();
      differentiate();
      integrate();
      compute();

      return getStepInterval();
    }

    virtual void initialize();

    virtual void clear();
    virtual void differentiate();
    virtual void turn();
    virtual void integrate();
    virtual void compute();

    virtual const char* const className() const  { return "SRMStepper"; }
 

  protected:

    IntegratorAllocator theIntegratorAllocator;

  private:

    virtual void distributeIntegrator( IntegratorAllocator );

  };

  class Euler1SRMStepper
    :
    public SRMStepper
  {

  public:

    Euler1SRMStepper();
    virtual ~Euler1SRMStepper() {}

    static StepperPtr instance() { return new Euler1SRMStepper; }

    virtual const char* const className() const  { return "Euler1SRMStepper"; }
 
  protected:

    static IntegratorPtr newEuler1( SubstanceRef substance );

  };


  class RungeKutta4SRMStepper
    : 
    public SRMStepper
  {

  public:

    RungeKutta4SRMStepper();
    virtual ~RungeKutta4SRMStepper() {}

    static StepperPtr instance() { return new RungeKutta4SRMStepper; }

    virtual void differentiate();

    virtual const char* const className() const 
    { return "RungeKutta4SRMStepper"; }

  private:

    static IntegratorPtr newRungeKutta4( SubstanceRef substance );

  };

} // namespace libecs

#endif /* ___STEPPER_H___ */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
