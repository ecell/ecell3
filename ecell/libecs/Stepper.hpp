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

#include "Util.hpp"
#include "Polymorph.hpp"
#include "PropertyInterface.hpp"
#include "System.hpp"

// to be removed
#include "Integrators.hpp"


namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */

  
  DECLARE_CLASS( Euler1Stepper );
  DECLARE_CLASS( RungeKutta4Stepper );
  DECLARE_CLASS( PropertySlotVectorImplementation );

  DECLARE_VECTOR( SubstancePtr, SubstanceVector );
  DECLARE_VECTOR( ReactorPtr,   ReactorVector );
  DECLARE_VECTOR( SystemPtr,    SystemVector );
  
  DECLARE_VECTOR( StepperPtr, StepperVector );

  DECLARE_VECTOR( PropertySlotPtr, PropertySlotVector );


  template< class Base_, class Derived_ = Base_ >
  class EntityCache
    :
    public std::vector<Derived_*>
  {

  public:

    DECLARE_TYPE( Base_, Base );
    DECLARE_TYPE( Derived_, Derived );

    typedef std::vector<DerivedPtr> DerivedVector_;
    DECLARE_TYPE( DerivedVector_, DerivedVector );

    EntityCache()
    {
      ; // do nothing
    }

    ~EntityCache() {}


    /**
       Update the cache.

    */

    void update( SystemVectorCref aSystemVector )
    {
      clear();

      for( SystemVectorConstIterator i( aSystemVector.begin() );
	   i != aSystemVector.end() ; ++i )
	{
	  const SystemCptr aSystem( *i );

	  typedef std::map<const String,BasePtr> BaseMap;
	  
	  for( typename BaseMap::const_iterator 
		 j( aSystem->System::getMap<Base>().begin() );
	       j != aSystem->System::getMap<Base>().end(); ++j )
	    {
	      BasePtr aBasePtr( (*j).second );

	      DerivedPtr aDerivedPtr( dynamic_cast<DerivedPtr>( aBasePtr ) );
	      if( aDerivedPtr != NULLPTR )
		{
		  push_back( aDerivedPtr );
		}
	    }

	}

    }

  };



  /**
     Stepper class defines and governs computation unit in a model.



  */

  class Stepper
    :
    public PropertyInterface
  {

  public:

    //    typedef std::pair<StepperPtr,Real> StepIntervalConstraint;
    //    DECLARE_VECTOR( StepIntervalConstraint, StepIntervalConstraintVector );
    DECLARE_ASSOCVECTOR( StepperPtr, Real, std::less<StepperPtr>,
			 StepIntervalConstraintMap );

    /** 
	A function type that returns a pointer to Stepper.

	Every subclass must have this type of a function which returns
	an instance for the StepperMaker.
    */

    typedef StepperPtr (* AllocatorFuncPtr )();


    Stepper(); 
    virtual ~Stepper() 
    {
      ; // do nothing
    }

    virtual void makeSlots();

    /**

    */

    virtual void initialize();

    /**


    */

    void sync();

    /**
       
    Each subclass of Stepper defines this.

    @note Subclass of Stepper must call this by Stepper::step() from
    their step().
    */

    virtual void step()
    {
      theCurrentTime += getStepInterval();
    }

    /**

    */

    void push();

    
    /**


       @param aSystem
    */

    void registerSystem( SystemPtr aSystem );

    /**


       @param aSystem
    */

    void removeSystem( SystemPtr aSystem );


    /**
       Get the current time of this Stepper.

       The current time is defined as a next scheduled point in time
       of this Stepper.

       @return the current time in Real.
    */

    const Real getCurrentTime() const
    {
      return theCurrentTime;
    }

    /**

    This may be overridden in dynamically scheduled steppers.

    */

    void setStepInterval( RealCref aStepInterval );


    /**
       Get the step interval of this Stepper.

       The step interval is a length of time that this Stepper proceeded
       in the last step.
       
       @return the step interval of this Stepper
    */

    const Real getStepInterval() const
    {
      return theStepInterval;
    }

    /**
       Get the number of steps per a second.

       'StepsPerSecond' is defined as 1.0 / getStepInterval().

       If you need to get a reciprocal of the step interval,
       use of this is more efficient than just getStepInterval(), because
       it is pre-calculated when the setStepInterval() is called.


       @return the number of steps per a second. (== 1.0 / getStepInterval )
    */

    const Real getStepsPerSecond() const
    {
      return theStepsPerSecond;
    }

    //@{

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

    //@}


    void registerPropertySlotWithProxy( PropertySlotPtr );

    void registerLoggedPropertySlot( PropertySlotPtr );


    const String getID() const
    {
      return theID;
    }

    void setID( StringCref anID )
    {
      theID = anID;
    }

    ModelPtr getModel() const
    {
      return theModel;
    }


    /**
       @internal
    */

    void setModel( ModelPtr const aModel )
    {
      theModel = aModel;
    }





    void setUserMinInterval( RealCref aValue )
    {
      theUserMinInterval = aValue;
    }

    const Real getUserMinInterval() const
    {
      return theUserMinInterval;
    }

    void setUserMaxInterval( RealCref aValue )
    {
      theUserMaxInterval = aValue;
    }

    const Real getUserMaxInterval() const
    {
      return theUserMaxInterval;
    }

    const Real getMinInterval() const
    {
      return getUserMinInterval();
    }

    const Real getMaxInterval() const;

    void setStepIntervalConstraint( PolymorphVectorRCPtrCref aValue );

    void setStepIntervalConstraint( StepperPtr aStepperPtr, RealCref aFactor );

    const PolymorphVectorRCPtr getStepIntervalConstraint() const;


    bool operator<( StepperCref rhs )
    {
      return getCurrentTime() < rhs.getCurrentTime();
    }


    SystemVectorCref getSystemVector() const
    {
      return theSystemVector;
    }

    void setCurrentTime( RealCref aTime )
    {
      theCurrentTime = aTime;
    }

    virtual StringLiteral getClassName() const  { return "Stepper"; }


    const PolymorphVectorRCPtr getSystemList() const;


  protected:

    void calculateStepsPerSecond()
    {
      theStepsPerSecond = 1.0 / getStepInterval();
    }

  protected:

    SystemVector        theSystemVector;

    PropertySlotVector  thePropertySlotWithProxyVector;
    PropertySlotVector  theLoggedPropertySlotVector;

    StepIntervalConstraintMap theStepIntervalConstraintMap;

  private:

    ModelPtr            theModel;

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theStepsPerSecond;

    Real                theUserMinInterval;
    Real                theUserMaxInterval;

    String              theID;

    bool                theEntityListChanged;

  };




  DECLARE_CLASS( SRMReactor );

  class SRMStepper 
    : 
    public Stepper
  {
  public:

    typedef EntityCache<Substance> SubstanceCache;
    typedef EntityCache<Substance,SRMSubstance> SRMSubstanceCache;
    typedef EntityCache<Reactor,SRMReactor> ReactorCache;
    //    typedef EntityCache<Reactor,Reactor> ReactorCache;

    SRMStepper();
    virtual ~SRMStepper() {}

    virtual void step();


    void clear();
    void react();
    void integrate();
    void rule();

    virtual void initialize();

    virtual StringLiteral getClassName() const  { return "SRMStepper"; }
 

  protected:

    Integrator::AllocatorFuncPtr theIntegratorAllocator;

    SubstanceCache        theSubstanceCache;
    SRMSubstanceCache     theSRMSubstanceCache;
    ReactorCache          theReactorCache;
    ReactorCache          theRuleReactorCache;

  private:

    virtual void distributeIntegrator( Integrator::AllocatorFuncPtr );

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

    static IntegratorPtr newIntegrator( SRMSubstanceRef substance );

  };


  class RungeKutta4SRMStepper
    : 
    public SRMStepper
  {

  public:

    RungeKutta4SRMStepper();
    virtual ~RungeKutta4SRMStepper() {}

    static StepperPtr createInstance() { return new RungeKutta4SRMStepper; }

    virtual void step();

    virtual StringLiteral getClassName() const 
    { return "RungeKutta4SRMStepper"; }

  private:

    static IntegratorPtr newIntegrator( SRMSubstanceRef substance );

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
