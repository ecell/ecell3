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
#include <valarray>

#include "libecs.hpp"

#include "Util.hpp"
#include "Polymorph.hpp"
#include "PropertyInterface.hpp"
#include "System.hpp"


namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */

  DECLARE_TYPE( std::valarray<Real>, RealValarray );

  DECLARE_CLASS( Euler1Stepper );
  DECLARE_CLASS( RungeKutta4Stepper );
  DECLARE_CLASS( AdaptiveStepsizeEuler1Stepper );

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

    DECLARE_TYPE( SubstanceVector, SubstanceCache );
    DECLARE_TYPE( ReactorVector, ReactorCache );

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

    @note Subclass of Stepper must call this by Stepper::calculate() from
    their step().
    */

    virtual void step() = 0;


    void integrate();

    /**
       Update loggers.

    */

    void log();
    void slave();
    void clear();
    void react();

    void reset();

    void updateVelocityBuffer();
    
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

    void setStepInterval( RealCref aStepInterval )
    {
      theTolerantStepInterval = aStepInterval;

      loadStepInterval( aStepInterval );
    }

    virtual void loadStepInterval( RealCref aStepInterval )
    {
      if( aStepInterval > getUserMaxInterval()
	  || aStepInterval <= getUserMinInterval() )
	{
	  // should use other exception?
	  THROW_EXCEPTION( RangeError, "Stepper StepInterval: out of range." );
	}

      theStepInterval = aStepInterval;
    }

    void setNextStepInterval( RealCref aStepInterval )
    {
      theNextStepInterval = aStepInterval;
    }

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

    const Real getTolerantStepInterval() const
    {
      return theTolerantStepInterval;
    }

    const Real getNextStepInterval() const
    {
      return theNextStepInterval;
    }

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


    /**
       Set slave stepper by a stepper ID string.

       If an empty string is given, this method unsets the slave stepper.
    */

    void setSlaveStepperID( StringCref aStepperID );

    /**
       Get an ID string of the slave stepper.

       @return an ID string of the slave stepper.
    */

    const String getSlaveStepperID() const;

    void setSlaveStepper( StepperPtr aStepperPtr )
    {
      theSlaveStepper = aStepperPtr;
    }

    StepperPtr getSlaveStepper() const
    {
      return theSlaveStepper;
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

    void setStepIntervalConstraint( PolymorphCref aValue );

    void setStepIntervalConstraint( StepperPtr aStepperPtr, RealCref aFactor );

    const Polymorph getStepIntervalConstraint() const;


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

    const UnsignedInt getSubstanceCacheIndex( SubstancePtr aSubstance );

    RealCptr getVelocityBufferElementPtr( UnsignedInt anIndex )
    {
      return &theVelocityBuffer[ anIndex ];
    }

    const Polymorph getSubstanceCache() const;
    const Polymorph getReactorCache() const;


    virtual StringLiteral getClassName() const  { return "Stepper"; }

    const Polymorph getSystemList() const;


  protected:

    SystemVector        theSystemVector;

    PropertySlotVector  theLoggedPropertySlotVector;

    StepIntervalConstraintMap theStepIntervalConstraintMap;

    SubstanceCache        theSubstanceCache;
    ReactorCache          theReactorCache;

    RealValarray theQuantityBuffer;
    RealValarray theVelocityBuffer;


  private:

    ModelPtr            theModel;

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theStepsPerSecond;

    Real                theTolerantStepInterval;
    Real                theNextStepInterval;

    Real                theUserMinInterval;
    Real                theUserMaxInterval;

    String              theID;

    StepperPtr          theSlaveStepper;

  };


  class Euler1Stepper
    :
    public Stepper
  {

  public:

    Euler1Stepper();
    virtual ~Euler1Stepper() {}

    virtual void step();

    static StepperPtr createInstance() { return new Euler1Stepper; }

    virtual StringLiteral getClassName() const  { return "Euler1Stepper"; }
 
  };


  class RungeKutta4Stepper
    : 
    public Stepper
  {

  public:

    RungeKutta4Stepper();
    virtual ~RungeKutta4Stepper() {}

    static StepperPtr createInstance() { return new RungeKutta4Stepper; }

    virtual void step();

    virtual StringLiteral getClassName() const 
    { return "RungeKutta4Stepper"; }


  protected:

  };

  class AdaptiveStepsizeEuler1Stepper
    :
    public Stepper
  {

  public:

    AdaptiveStepsizeEuler1Stepper();
    virtual ~AdaptiveStepsizeEuler1Stepper() {}

    static StepperPtr createInstance() { return new AdaptiveStepsizeEuler1Stepper; }

    virtual void initialize();
    virtual void step();

    virtual StringLiteral getClassName() const { return "AdaptiveStepsizeEuler1Stepper"; }

  protected:

  };

  class AdaptiveStepsizeMidpoint2Stepper
    : 
    public Stepper
  {

  public:

    AdaptiveStepsizeMidpoint2Stepper();
    virtual ~AdaptiveStepsizeMidpoint2Stepper() {}

    static StepperPtr createInstance() { return new AdaptiveStepsizeMidpoint2Stepper; }

    virtual void initialize();
    virtual void step();

    virtual StringLiteral getClassName() const 
    { return "AdaptiveStepsizeMidpoint2Stepper"; }

  protected:

    RealValarray theK1;
    RealValarray theErrorEstimate;
  };

  class CashKarp4Stepper
    : 
    public Stepper
  {

  public:

    CashKarp4Stepper();
    virtual ~CashKarp4Stepper() {}

    static StepperPtr createInstance() { return new CashKarp4Stepper; }

    virtual void initialize();
    virtual void step();

    virtual StringLiteral getClassName() const 
    { return "CashKarp4Stepper"; }

  protected:

    RealValarray theK1;
    RealValarray theK2;
    RealValarray theK3;
    RealValarray theK4;
    RealValarray theK5;
    RealValarray theK6;

    RealValarray theErrorEstimate;
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
