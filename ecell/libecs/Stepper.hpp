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
//#include <valarray>

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

  //  DECLARE_TYPE( std::valarray<Real>, RealValarray );

  DECLARE_VECTOR( Real, RealVector );


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

    //    DECLARE_TYPE( VariableVector, VariableCache );
    //    DECLARE_TYPE( ProcessVector,  ProcessCache );

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
    void process();
    void processNegative();
    void processNormal();

    void reset();
    
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

    void setCurrentTime( RealCref aTime )
    {
      theCurrentTime = aTime;
    }

    /**
       This may be overridden in dynamically scheduled steppers.

    */

    virtual void setStepInterval( RealCref aStepInterval )
    {
      loadStepInterval( aStepInterval );
    }

    void loadStepInterval( RealCref aStepInterval )
    {
      if( aStepInterval > getUserMaxInterval()
	  || aStepInterval <= getUserMinInterval() )
	{
	  // should use other exception?
	  THROW_EXCEPTION( RangeError, "Stepper StepInterval: out of range. ("
			   + toString( aStepInterval ) + String( ")\n" ) );
	}

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

    const UnsignedInt findInVariableCache( VariablePtr aVariable );

    RealCptr getVelocityBufferElementPtr( UnsignedInt anIndex )
    {
      return &theVelocityBuffer[ anIndex ];
    }

    const Polymorph getVariableCache() const;
    const Polymorph getProcessCache() const;

    virtual StringLiteral getClassName() const  { return "Stepper"; }

    const Polymorph getSystemList() const;

  protected:

    SystemVector        theSystemVector;

    PropertySlotVector  theLoggedPropertySlotVector;

    StepIntervalConstraintMap theStepIntervalConstraintMap;

    VariableVector        theVariableCache;
    ProcessVector         theProcessCache;

    ProcessVectorIterator theFirstNormalProcess;

    RealVector theValueBuffer;
    RealVector theVelocityBuffer;


  private:

    ModelPtr            theModel;

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theStepsPerSecond;

    Real                theUserMinInterval;
    Real                theUserMaxInterval;

    String              theID;

    StepperPtr          theSlaveStepper;
  };


  /**
     DIFFERENTIAL EQUATION SOLVER


  */

  class DifferentialStepper
    :
    public Stepper
  {
  public:

    DifferentialStepper();
    virtual ~DifferentialStepper() {}

    /**
       Adaptive stepsize control.

       These methods are for handling the standerd error control objects.
    */

    const Real getRelativeTorelance() const
    {
      return theRelativeTorelance;
    }

    void setRelativeTorelance( RealCref aValue )
    {
      theRelativeTorelance = aValue;
    }

    void setAbsoluteTorelance( RealCref aValue )
    {
      theAbsoluteTorelance = aValue;
    }

    const Real getAbsoluteTorelance() const
    {
      return theAbsoluteTorelance;
    }

    void setStateScalingFactor( RealCref aValue )
    {
      theStateScalingFactor = aValue;
    }

    const Real getStateScalingFactor() const
    {
      return theStateScalingFactor;
    }

    void setDerivativeScalingFactor( RealCref aValue )
    {
      theDerivativeScalingFactor = aValue;
    }

    const Real getDerivativeScalingFactor() const
    {
      return theDerivativeScalingFactor;
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

    virtual void makeSlots();

    virtual void initialize();

    virtual StringLiteral getClassName() const 
    { 
      return "DifferentialStepper";
    }


  protected:

    Real safety;


  private:

    Real theRelativeTorelance;
    Real theAbsoluteTorelance;
    Real theStateScalingFactor;
    Real theDerivativeScalingFactor;

    Real theTolerantStepInterval;
    Real theNextStepInterval;
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
