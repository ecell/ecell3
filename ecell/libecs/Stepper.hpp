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

#ifndef __STEPPER_HPP
#define __STEPPER_HPP

#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>

#include "libecs.hpp"

#include "Util.hpp"
#include "Polymorph.hpp"
#include "VariableProxy.hpp"
#include "PropertyInterface.hpp"
//#include "System.hpp"



namespace libecs
{

  /** @addtogroup stepper
   *@{
   */

  /** @file */


  //  DECLARE_TYPE( std::valarray<Real>, RealValarray );

  DECLARE_VECTOR( Real, RealVector );


  /**
     Stepper class defines and governs a computation unit in a model.

     The computation unit is defined as a set of Process objects.

  */

  class Stepper
    :
    public PropertyInterface
  {

  public:

    //    typedef std::pair<StepperPtr,Real> StepIntervalConstraint;
    //    DECLARE_VECTOR( StepIntervalConstraint, StepIntervalConstraintVector );

    //    DECLARE_ASSOCVECTOR( StepperPtr, Real, std::less<StepperPtr>,
    //			 StepIntervalConstraintMap );

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

    void makeSlots();

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
    void clear();
    void process();

    virtual void reset();
    
    /**
       Register a System to this Stepper.

       @param aSystemPtr a pointer to a System object to register
    */

    void registerSystem( SystemPtr aSystemPtr );

    /**
       Remove a System from this Stepper.

       @note This method is not currently supported.  Calling this method
       causes undefined behavior.

       @param aSystemPtr a pointer to a System object
    */

    void removeSystem( SystemPtr aSystemPtr );

    /**
       Register a Process to this Stepper.

       @param aProcessPtr a pointer to a Process object to register
    */

    void registerProcess( ProcessPtr aProcessPtr );

    /**
       Remove a Process from this Stepper.

       @note This method is not currently supported.  Calling this method
       causes undefined behavior.

       @param aProcessPtr a pointer to a Process object
    */

    void removeProcess( ProcessPtr aProcessPtr );

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

    void setCurrentTime( RealCref aTime )
    {
      theCurrentTime = aTime;
    }

    /**
       This may be overridden in dynamically scheduled steppers.

    */

    virtual void setStepInterval( RealCref aStepInterval )
    {
      if( aStepInterval > getMaxInterval()
	  || aStepInterval < getMinInterval() )
	{
	  // should use other exception?
	  THROW_EXCEPTION( RangeError, "Stepper StepInterval: out of range. ("
			   + toString( aStepInterval ) + String( ")\n" ) );
	}

      loadStepInterval( aStepInterval );
    }

    void loadStepInterval( RealCref aStepInterval )
    {
      theStepInterval = aStepInterval;
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

    virtual const Real getTimeScale() const
    {
      return getStepInterval();
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

    void setSchedulerIndex( IntCref anIndex )
    {
      theSchedulerIndex = anIndex;
    }

    const Int getSchedulerIndex() const
    {
      return theSchedulerIndex;
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

    //    void setStepIntervalConstraint( PolymorphCref aValue );

    //    void setStepIntervalConstraint( StepperPtr aStepperPtr, RealCref aFactor );

    //    const Polymorph getStepIntervalConstraint() const;


    SystemVectorCref getSystemVector() const
    {
      return theSystemVector;
    }

    ProcessVectorCref getProcessVector() const
    {
      return theProcessVector;
    }

    VariableProxyVectorCref getVariableProxyVector() const
    {
      return theVariableProxyVector;
    }

    VariableVectorCref getReadVariableVector() const
    {
      return theReadVariableVector;
    }

    StepperVectorCref getDependentStepperVector() const
    {
      return theDependentStepperVector;
    }

    RealVectorCref getValueBuffer() const
    {
      return theValueBuffer;
    }


    const UnsignedInt getVariableProxyIndex( VariableCptr const aVariable );

    /**
       Update theProcessVector and theFirstNormalProcess iterator.

    */

    void updateProcessVector();


    /**
       Update theReadVariableVector and theVariableProxyVector.

    */

    void updateVariableVectors();

    void updateLoggedPropertySlotVector();

    /**

	Definition of the Stepper dependency:
	Stepper A depends on Stepper B 
	if:
	- A and B share at least one Variable, AND
	- A reads AND B writes (changes) the Variable.

	See VariableReference class about the definitions of
	Variable 'read' and 'write'.


	@see Process, VariableReference
    */

    void updateDependentStepperVector();


    virtual void dispatchInterruptions();
    virtual void interrupt( StepperPtr const aCaller );


    const Polymorph getWriteVariableList()    const;
    const Polymorph getReadVariableList()     const;
    const Polymorph getProcessList()          const;
    const Polymorph getSystemList()           const;
    const Polymorph getDependentStepperList() const;


    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariablePtr )
    {
      return new VariableProxy( aVariablePtr );
    }


    bool operator<( StepperCref rhs )
    {
      return getCurrentTime() < rhs.getCurrentTime();
    }

    virtual StringLiteral getClassName() const  { return "Stepper"; }

  protected:

    SystemVector        theSystemVector;

    PropertySlotVector  theLoggedPropertySlotVector;

    //    StepIntervalConstraintMap theStepIntervalConstraintMap;

    VariableVector        theReadVariableVector;

    VariableProxyVector   theVariableProxyVector;

    ProcessVector         theProcessVector;

    StepperVector         theDependentStepperVector;

    RealVector theValueBuffer;

  private:

    ModelPtr            theModel;
    
    // the index on the scheduler
    Int                 theSchedulerIndex;

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theStepsPerSecond;

    Real                theUserMinInterval;
    Real                theUserMaxInterval;

    String              theID;

  };


  /**
     DIFFERENTIAL EQUATION SOLVER


  */

  DECLARE_CLASS( DifferentialStepper );

  class DifferentialStepper
    :
    public Stepper
  {


  public:

    class VariableProxy
      :
      public libecs::VariableProxy
    {
    public:
      VariableProxy( DifferentialStepperRef aStepper, 
		    VariablePtr const aVariablePtr )
	:
	libecs::VariableProxy( aVariablePtr ),
	theStepper( aStepper ),
	theIndex( theStepper.getVariableProxyIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getVelocity( RealCref aTime )
      {
	return theStepper.getVelocityBuffer()[ theIndex ]
	  * ( aTime - theStepper.getCurrentTime() );
      }
      

    protected:

      DifferentialStepperRef theStepper;
      UnsignedInt            theIndex;

    };


  public:

    DifferentialStepper();
    virtual ~DifferentialStepper() {}

    /**
       Adaptive stepsize control.

       These methods are for handling the standerd error control objects.
    */

    void setTolerance( RealCref aValue )
    {
      theTolerance = aValue;
    }

    const Real getTolerance() const
    {
      return theTolerance;
    }

    const Real getAbsoluteToleranceFactor() const
    {
      return theAbsoluteToleranceFactor;
    }

    void setAbsoluteToleranceFactor( RealCref aValue )
    {
      theAbsoluteToleranceFactor = aValue;
    }

    void setStateToleranceFactor( RealCref aValue )
    {
      theStateToleranceFactor = aValue;
    }

    const Real getStateToleranceFactor() const
    {
      return theStateToleranceFactor;
    }

    void setDerivativeToleranceFactor( RealCref aValue )
    {
      theDerivativeToleranceFactor = aValue;
    }

    const Real getDerivativeToleranceFactor() const
    {
      return theDerivativeToleranceFactor;
    }

    void setMaxErrorRatio( RealCref aValue )
    {
      theMaxErrorRatio = aValue;
    }

    const Real getMaxErrorRatio() const
    {
      return theMaxErrorRatio;
    }

    /**
       Override setStepInterval() for theTolerantStepInterval.

    */

    virtual void setStepInterval( RealCref aStepInterval )
    {
      theTolerantStepInterval = aStepInterval;

      loadStepInterval( aStepInterval );
    }

    void setNextStepInterval( RealCref aStepInterval )
    {
      theNextStepInterval = aStepInterval;
    }

    const Real getTolerantStepInterval() const
    {
      return theTolerantStepInterval;
    }

    const Real getNextStepInterval() const
    {
      return theNextStepInterval;
    }

    void initializeStepInterval( RealCref aStepInterval )
    {
      setStepInterval( aStepInterval );
      setNextStepInterval( aStepInterval );
    }

    void makeSlots();

    virtual void initialize();

    virtual void reset();

    virtual void interrupt( StepperPtr const aCaller );


    RealVectorCref getVelocityBuffer() const
    {
      return theVelocityBuffer;
    }


    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariable )
    {
      return new VariableProxy( *this, aVariable );
    }

    virtual StringLiteral getClassName() const 
    { 
      return "DifferentialStepper";
    }

  protected:

    const bool isExternalErrorTolerable() const;

  protected:

    Real safety;

    RealVector theVelocityBuffer;

  private:

    Real theTolerance;
    Real theAbsoluteToleranceFactor;
    Real theStateToleranceFactor;
    Real theDerivativeToleranceFactor;

    Real theTolerantStepInterval;
    Real theNextStepInterval;

    Real theMaxErrorRatio;
  };


  /**

  */

  class DiscreteEventStepper
    :
    public Stepper
  {

  public:

    DiscreteEventStepper();
    virtual ~DiscreteEventStepper() {}

    // virtual void step();

    //    virtual void interrupt( StepperPtr const aCaller )
    //    {
    //      ; // do nothing -- ignore interruption
    //    }

    virtual void dispatchInterruptions();

    //    static StepperPtr createInstance() { return new DiscreteEventStepper; }

    virtual StringLiteral getClassName() const 
    { 
      return "DiscreteEventStepper";
    }

  };


  /**
     DiscreteTimeStepper has a fixed step interval.
     
     This stepper ignores incoming interruptions, but dispatches 
     interruptions always when it steps.

     Process objects in this Stepper isn't allowed to use 
     Variable::addVelocity() method, but Variable::setValue() method only.

  */

  class DiscreteTimeStepper
    :
    public Stepper
  {

  public:

    DiscreteTimeStepper();
    virtual ~DiscreteTimeStepper() {}


    virtual void step();

    virtual void interrupt( StepperPtr const aCaller )
    {
      ; // do nothing -- ignore interruption
    }

    static StepperPtr createInstance() { return new DiscreteTimeStepper; }

    virtual StringLiteral getClassName() const 
    { 
      return "DiscreteTimeStepper";
    }


  };


  /**
     SlaveStepper steps only when triggered by incoming interruptions from
     other Steppers.

     This Stepper never dispatch interruptions to other Steppers.

     The step interval of this Stepper is fixed to infinity -- which 
     means that this doesn't step spontaneously.

  */

  class SlaveStepper
    :
    public Stepper
  {
  public:

    SlaveStepper();
    ~SlaveStepper() {}
    
    virtual void initialize();

    virtual void step()
    {
      process();
    }

    virtual void interrupt( StepperPtr const aCaller )
    {
      setCurrentTime( aCaller->getCurrentTime() );

      step();
    }

    virtual void setStepInterval( RealCref aStepInterval )
    {
      // skip range check
      loadStepInterval( aStepInterval );
    }

    virtual void dispatchInterruptions()
    {
      ; // do nothing
    }

    static StepperPtr createInstance() { return new SlaveStepper; }

    virtual StringLiteral getClassName() const 
    { 
      return "SlaveStepper";
    }


  };


} // namespace libecs

#endif /* __STEPPER_HPP */



/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
