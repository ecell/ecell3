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

#ifndef ___STEPPER_H___
#define ___STEPPER_H___

#include <vector>
#include <map>
#include <algorithm>
#include <utility>

#include "libecs.hpp"

#include "System.hpp"



namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
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

  DECLARE_VECTOR( PropertySlotPtr, PropertySlotVector );



  class Stepper
  {

  public:

    Stepper(); 
    virtual ~Stepper() {}

    void connectSystem( SystemPtr aSystem );
    void disconnectSystem( SystemPtr aSystem );


    RealCref getCurrentTime() const
    {
      return theCurrentTime;
    }

    void setCurrentTime( RealCref aTime )
    {
      theCurrentTime = aTime;
    }


    /**

    This may be overridden in dynamically scheduled steppers.

    */

    void setStepInterval( RealCref aStepInterval );

    void calculateStepsPerSecond();

    RealCref getStepInterval() const
    {
      return theStepInterval;
    }

    RealCref getStepsPerSecond() const
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

    SystemVectorCref getSystemVector() const
    {
      return theSystemVector;
    }

    void registerSlaves( SystemPtr );

    void registerPropertySlotWithProxy( PropertySlotPtr );

    void registerLoggedPropertySlot( PropertySlotPtr );


    void sync();
    void step() 
    { 
      compute();
      theCurrentTime += getStepInterval();
    }
    void push();

    // each stepper class defines this
    virtual void compute() = 0;


    StringCref getName() const
    {
      return theName;
    }

    void setName( StringCref aName )
    {
      theName = aName;
    }


    bool operator<( StepperCref rhs )
    {
      return getCurrentTime() < rhs.getCurrentTime();
    }


    virtual StringLiteral getClassName() const  { return "Stepper"; }


  protected:

    void searchSlaves( SystemPtr aStartSystemPtr );

  protected:

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theStepsPerSecond;

    SystemVector        theSystemVector;

    PropertySlotVector  thePropertySlotWithProxyVector;
    PropertySlotVector  theLoggedPropertySlotVector;

    String              theName;

    bool                theEntityListChanged;

  };


  class StepperWithEntityCache
    : 
    public Stepper
  {

  public:

    StepperWithEntityCache()
    {
      ; // do nothing
    }

    ~StepperWithEntityCache() {}


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
       Update the caches.

       SubstanceCache and ReactorCache are updated so that they contain
       all Substances and Reactors in the Systems that are belong to this
       Stepper.
    */

    void updateCache();


    /**
       Update the caches with sort.

       SubstanceCache and ReactorCache are updated so that they contain
       all Substances and Reactors in the Systems that are belong to this
       Stepper.

       Not fully implemented.
    */

    void updateCacheWithSort();

  protected:

    SubstanceVector               theSubstanceCache;
    ReactorVector                 theReactorCache;

  };

  class SRMStepper 
    : 
    public StepperWithEntityCache
  {
  public:

    SRMStepper();
    virtual ~SRMStepper() {}

    virtual void compute()
    {
      clear();
      differentiate();
      integrate();
      //???();

    }

    virtual void initialize();

    virtual void clear();
    virtual void differentiate();
    virtual void turn();
    virtual void integrate();
    //    virtual void ???();

    virtual StringLiteral getClassName() const  { return "SRMStepper"; }
 

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

    static StepperPtr createInstance() { return new Euler1SRMStepper; }

    virtual StringLiteral getClassName() const  { return "Euler1SRMStepper"; }
 
  protected:

    static IntegratorPtr newIntegrator( SubstanceRef substance );

  };


  class RungeKutta4SRMStepper
    : 
    public SRMStepper
  {

  public:

    RungeKutta4SRMStepper();
    virtual ~RungeKutta4SRMStepper() {}

    static StepperPtr createInstance() { return new RungeKutta4SRMStepper; }

    virtual void differentiate();

    virtual StringLiteral getClassName() const 
    { return "RungeKutta4SRMStepper"; }

  private:

    static IntegratorPtr newIntegrator( SubstanceRef substance );

  };

  /** @} */ //end of libecs_module 

} // namespace libecs

#endif /* ___STEPPER_H___ */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
