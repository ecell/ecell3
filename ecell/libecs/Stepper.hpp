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
      Real aNewStepInterval( aStepInterval );

      if( aNewStepInterval > getMaxInterval() )
	{
	  aNewStepInterval = getMaxInterval();
	}
      else if ( aNewStepInterval < getMinInterval() )
	{
	  aNewStepInterval = getMinInterval();
	}

      loadStepInterval( aNewStepInterval );
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


    void setMinInterval( RealCref aValue )
    {
      theMinInterval = aValue;
    }

    const Real getMinInterval() const
    {
      return theMinInterval;
    }

    void setMaxInterval( RealCref aValue )
    {
      theMaxInterval = aValue;
    }

    const Real getMaxInterval() const
    {
      Real aMaxInterval( theMaxInterval );

      /*
      for( StepIntervalConstraintMapConstIterator 
	     i( theStepIntervalConstraintMap.begin() ); 
	   i != theStepIntervalConstraintMap.end() ; ++i )
	{
	  const StepperPtr aStepperPtr( (*i).first );
	  Real aConstraint( aStepperPtr->getStepInterval() * (*i).second );

	  if( aMaxInterval > aConstraint )
	    {
	      aMaxInterval = aConstraint;
	    }
	}
      */

      return aMaxInterval;
    }
  
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

    VariableVectorCref getVariableVector() const
    {
      return theVariableVector;
    }

    const VariableVector::size_type getReadWriteVariableOffset()
    {
      return theReadWriteVariableOffset;
    }

    const VariableVector::size_type getReadOnlyVariableOffset()
    {
      return theReadOnlyVariableOffset;
    }

    StepperVectorCref getDependentStepperVector() const
    {
      return theDependentStepperVector;
    }

    RealVectorCref getValueBuffer() const
    {
      return theValueBuffer;
    }


    const UnsignedInt getVariableIndex( VariableCptr const aVariable );


    virtual void dispatchInterruptions();
    virtual void interrupt( StepperPtr const aCaller );


    const Polymorph getWriteVariableList()    const;
    const Polymorph getReadVariableList()     const;
    const Polymorph getProcessList()          const;
    const Polymorph getSystemList()           const;
    const Polymorph getDependentStepperList() const;

    /**

	Definition of the Stepper dependency:
	Stepper A depends on Stepper B 
	if:
	- A and B share at least one Variable, AND
	- A reads AND B writes on (changes) the same Variable.

	See VariableReference class about the definitions of
	Variable 'read' and 'write'.


	@see Process, VariableReference
    */

    void updateDependentStepperVector();


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

    /**
       Update theProcessVector.

    */

    void updateProcessVector();


    /**
       Update theVariableVector and theVariableProxyVector.

       This method makes the following data structure:
    
       In theVariableVector, Variables are sorted in this order:
     
       | Write-Only | Read-Write.. | Read-Only.. |
    
    */

    void updateVariableVector();

    /**
       Create VariableProxy objects and distribute the objects to 
       write Variables.

       Ownership of the VariableProxy objects are given away to the Variables.

       @see Variable::registerVariableProxy()
    */

    void createVariableProxies();

    /**
       Scan all the relevant Entity objects to this Stepper and construct
       the list of logged PropertySlots.

       The list, theLoggedPropertySlotVector, is used when in log() method.

    */

    void updateLoggedPropertySlotVector();




  protected:

    SystemVector        theSystemVector;

    PropertySlotVector  theLoggedPropertySlotVector;

    //    StepIntervalConstraintMap theStepIntervalConstraintMap;

    VariableVector         theVariableVector;
    VariableVector::size_type theReadWriteVariableOffset;
    VariableVector::size_type theReadOnlyVariableOffset;

    ProcessVector         theProcessVector;

    StepperVector         theDependentStepperVector;

    RealVector theValueBuffer;



  private:

    ModelPtr            theModel;
    
    // the index on the scheduler
    Int                 theSchedulerIndex;

    Real                theCurrentTime;

    Real                theStepInterval;

    Real                theMinInterval;
    Real                theMaxInterval;

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
	theIndex( theStepper.getVariableIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getDifference( RealCref aTime, RealCref anInterval )
      {
	// First order interpolation.  This should be overridden in
	// higher order DifferentialSteppers.
	return theStepper.getVelocityBuffer()[ theIndex ] * anInterval;
      }
      

    protected:

      DifferentialStepperRef theStepper;
      UnsignedInt            theIndex;

    };


  public:

    DifferentialStepper();
    virtual ~DifferentialStepper() {}

    /**
       Override setStepInterval() for theTolerantStepInterval.

    */

    virtual void setStepInterval( RealCref aStepInterval )
    {
      theTolerantStepInterval = aStepInterval;

      Stepper::setStepInterval( aStepInterval );
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
      return new DifferentialStepper::VariableProxy( *this, aVariable );
    }

    virtual StringLiteral getClassName() const 
    { 
      return "DifferentialStepper";
    }

  protected:

    const bool isExternalErrorTolerable() const;

  protected:

    RealVector theVelocityBuffer;

  private:

    Real theTolerantStepInterval;
    Real theNextStepInterval;

  };


  /**
     ADAPTIVE STEPSIZE DIFFERENTIAL EQUATION SOLVER


  */

  DECLARE_CLASS( AdaptiveDifferentialStepper );

  class AdaptiveDifferentialStepper
    :
    public DifferentialStepper
  {
  public:

    class VariableProxy
      :
      public libecs::VariableProxy
    {
    public:

      VariableProxy( AdaptiveDifferentialStepperRef aStepper, 
		     VariablePtr const aVariablePtr )
	:
	libecs::VariableProxy( aVariablePtr ),
	theStepper( aStepper ),
	theIndex( theStepper.getVariableIndex( aVariablePtr ) )
      {
	; // do nothing
      }

      virtual const Real getDifference( RealCref aTime, RealCref anInterval )
      {
	const Real aTimeInterval( aTime - theStepper.getCurrentTime() );

	const Real theta( ( aTimeInterval + aTimeInterval - anInterval )
			   / theStepper.getStepInterval() );

	const Real k1 = theStepper.getK1()[ theIndex ];
	const Real k2 = theStepper.getVelocityBuffer()[ theIndex ];

	return ( ( k1 + ( k2 - k1 ) * theta ) * anInterval );
      }

    protected:

      AdaptiveDifferentialStepperRef theStepper;
      UnsignedInt                    theIndex;
    };

  public:

    AdaptiveDifferentialStepper();
    virtual ~AdaptiveDifferentialStepper() {}

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

    virtual const Int getOrder() const { return 1; }

    void makeSlots();

    virtual void initialize();
    void step();
    virtual bool calculate() = 0;

    virtual StringLiteral getClassName() const 
    { 
      return "AdaptiveDifferentialStepper";
    }

    RealVectorCref getK1() const
    {
      return theK1;
    }

  protected:

    Real safety;
    RealVector theK1;

  private:

    Real theTolerance;
    Real theAbsoluteToleranceFactor;
    Real theStateToleranceFactor;
    Real theDerivativeToleranceFactor;

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
