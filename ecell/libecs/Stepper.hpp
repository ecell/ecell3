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

//#include <vector>
//#include <algorithm>
//#include <utility>
//#include <iostream>

#include "libecs.hpp"

#include "Util.hpp"
#include "Polymorph.hpp"
#include "VariableProxy.hpp"
#include "PropertyInterface.hpp"



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

    DM_BASECLASS( Stepper );

    class PriorityCompare
    {
    public:
      bool operator()( StepperPtr aLhs, StepperPtr aRhs ) const
      {
	return compare( aLhs->getPriority(), aRhs->getPriority() );
      }

      bool operator()( StepperPtr aLhs, const Int aRhs ) const
      {
	return compare( aLhs->getPriority(), aRhs );
      }

      bool operator()( const Int aLhs, StepperPtr aRhs ) const
      {
	return compare( aLhs, aRhs->getPriority() );
      }

    private:

      // if statement can be faster than returning an expression directly
      inline static bool compare( const Int aLhs, const Int aRhs )
      {
	if( aLhs < aRhs )
	  {
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }


    };



    //    typedef std::pair<StepperPtr,Real> StepIntervalConstraint;
    //    DECLARE_VECTOR( StepIntervalConstraint, StepIntervalConstraintVector );

    //    DECLARE_ASSOCVECTOR( StepperPtr, Real, std::less<StepperPtr>,
    //			 StepIntervalConstraintMap );

    Stepper(); 
    virtual ~Stepper() 
    {
      ; // do nothing
    }


    /**
       Get the current time of this Stepper.

       The current time is defined as a next scheduled point in time
       of this Stepper.

       @return the current time in Real.
    */

    GET_METHOD( Real, CurrentTime )
    {
      return theCurrentTime;
    }

    SET_METHOD( Real, CurrentTime )
    {
      theCurrentTime = value;
    }

    /**
       This may be overridden in dynamically scheduled steppers.

    */

    virtual SET_METHOD( Real, StepInterval )
    {
      Real aNewStepInterval( value );

      if( aNewStepInterval > getMaxStepInterval() )
	{
	  aNewStepInterval = getMaxStepInterval();
	}
      else if ( aNewStepInterval < getMinStepInterval() )
	{
	  aNewStepInterval = getMinStepInterval();
	}

      loadStepInterval( aNewStepInterval );
    }


    /**
       Get the step interval of this Stepper.

       The step interval is a length of time that this Stepper proceeded
       in the last step.
       
       @return the step interval of this Stepper
    */


    GET_METHOD( Real, StepInterval )
    {
      return theStepInterval;
    }

    virtual GET_METHOD( Real, TimeScale )
    {
      return getStepInterval();
    }

    /**
       theOriginalStepInterval for getDifference(),
       must need to be independent of interruption, theStepInterval.
    */

    SET_METHOD( Real, OriginalStepInterval )
    {
      theOriginalStepInterval = value;
    }

    GET_METHOD( Real, OriginalStepInterval )
    {
      return theOriginalStepInterval;
    }

    SET_METHOD( String, ID )
    {
      theID = value;
    }

    GET_METHOD( String, ID )
    {
      return theID;
    }


    SET_METHOD( Real, MinStepInterval )
    {
      theMinStepInterval = value;
    }

    GET_METHOD( Real, MinStepInterval )
    {
      return theMinStepInterval;
    }

    SET_METHOD( Real, MaxStepInterval )
    {
      theMaxStepInterval = value;
    }

    GET_METHOD( Real, MaxStepInterval )
    {
      Real aMaxStepInterval( theMaxStepInterval );

      /*
      for( StepIntervalConstraintMapConstIterator 
	     i( theStepIntervalConstraintMap.begin() ); 
	   i != theStepIntervalConstraintMap.end() ; ++i )
	{
	  const StepperPtr aStepperPtr( (*i).first );
	  Real aConstraint( aStepperPtr->getStepInterval() * (*i).second );

	  if( aMaxStepInterval > aConstraint )
	    {
	      aMaxStepInterval = aConstraint;
	    }
	}
      */

      return aMaxStepInterval;
    }


    GET_METHOD( Polymorph, WriteVariableList );
    GET_METHOD( Polymorph, ReadVariableList );
    GET_METHOD( Polymorph, ProcessList );
    GET_METHOD( Polymorph, SystemList );
    GET_METHOD( Polymorph, DependentStepperList );


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

    void integrate( const Real aTime );

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


    void loadStepInterval( RealCref aStepInterval )
    {
      theStepInterval = aStepInterval;
    }


    void registerLoggedPropertySlot( PropertySlotPtr );



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


    /**
       Set a priority value of this Stepper.

       The priority is an Int value which is used to determine the
       order of step when more than one Stepper is scheduled at the
       same point in time (such as starting up: t=0).

       @param aValue the priority value as an Int.
       @see Scheduler
    */

    SET_METHOD( Int, Priority )
    {
      thePriority = value;
    }

    /**
       @see setPriority()
    */

    GET_METHOD( Int, Priority )
    {
      return thePriority;
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

    /**

	Definition of the Stepper dependency:
	Stepper A depends on Stepper B 
	if:
	- A and B share at least one Variable, AND
	- A reads AND B writes on (changes) the same Variable.
m
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

    //    virtual StringLiteral getClassName() const  { return "Stepper"; }


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
    Int                 thePriority;

    Real                theCurrentTime;

    Real                theStepInterval;
    Real                theOriginalStepInterval;

    Real                theMinStepInterval;
    Real                theMaxStepInterval;

    String              theID;

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
