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

#include <gsl/gsl_rng.h>

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


  LIBECS_DM_CLASS( Stepper, PropertiedClass )
  {

  public:

    LIBECS_DM_BASECLASS( Stepper );

    LIBECS_DM_OBJECT_ABSTRACT( Stepper )
      {
	INHERIT_PROPERTIES( PropertiedClass );
	
	PROPERTYSLOT_SET_GET( Int,       Priority );
	PROPERTYSLOT_SET_GET( Real,      StepInterval );
	PROPERTYSLOT_SET_GET( Real,      MaxStepInterval );
	PROPERTYSLOT_SET_GET( Real,      MinStepInterval );
	PROPERTYSLOT_SET    ( String,    RngSeed );


	// these properties are not loaded/saved.
	PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real,  OriginalStepInterval );
	PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Real,      CurrentTime );
	PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, ProcessList );
	PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, SystemList );
	PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, DependentStepperList );
	PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, ReadVariableList );
	PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, WriteVariableList );


	// setting rng type:  not yet supported
	//PROPERTYSLOT_SET_GET( Polymorph, Rng,              Stepper );
      }


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
	if( aLhs > aRhs )
	  {
	    return true;
	  }
	else
	  {
	    return false;
	  }
      }


    };


    Stepper(); 
    virtual ~Stepper();


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
      return theMaxStepInterval;
    }


    GET_METHOD( Polymorph, WriteVariableList );
    GET_METHOD( Polymorph, ReadVariableList );
    GET_METHOD( Polymorph, ProcessList );
    GET_METHOD( Polymorph, SystemList );
    GET_METHOD( Polymorph, DependentStepperList );

    SET_METHOD( String, RngSeed );

    GET_METHOD( String, RngType );


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


    void registerLogger( LoggerPtr );



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

  
    SystemVectorCref getSystemVector() const
    {
      return theSystemVector;
    }

    /**
       Get the reference to the ProcessVector of this Stepper.

       The ProcessVector holds a set of pointers to this Stepper's Processes.

       The ProcessVector is partitioned in this way:

       |  Continuous Processes  |  Discrete Processes |

       getDiscreteProcessOffset() method returns the offset (index number)
       of the first discrete Process in this Stepper.

       Each part of the ProcessVector is sorted by Priority properties
       of Processes.

    */

    ProcessVectorCref getProcessVector() const
    {
      return theProcessVector;
    }

    /**

    @see getProcessVector()

    */

    const ProcessVector::size_type getDiscreteProcessOffset() const
    {
      return theDiscreteProcessOffset;
    }

    /**
       Get the reference to the VariableVector of this Stepper.

       In the VariableVector, Variables are classified and partitioned
       into the following three groups:

       | Write-Only Variables | Read-Write Variables | Read-Only Variables |

       Use getReadWriteVariableOffset() method to get the index of the first 
       Read-Write Variable in the VariableVector.  

       Use getReadOnlyVariableOffset() method to get the index of the first
       Read-Only Variable in the VariableVector.
       

    */

    VariableVectorCref getVariableVector() const
    {
      return theVariableVector;
    }

    /**
       @see getVariableVector()
    */

    const VariableVector::size_type getReadWriteVariableOffset()
    {
      return theReadWriteVariableOffset;
    }

    /**
       @see getVariableVector()
    */

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

	See VariableReference class about the definitions of
	Variable 'read' and 'write'.


	@see Process, VariableReference
    */

    void updateDependentStepperVector();

    /** 
	This method updates theIntegratedVariableVector.

	theIntegratedVariableVector holds the Variables those
	isIntegrationNeeded() method return true.   
    
	This method must be called after initialize().

	@internal
     */

    void updateIntegratedVariableVector();


    virtual VariableProxyPtr createVariableProxy( VariablePtr aVariablePtr )
    {
      return new VariableProxy( aVariablePtr );
    }

    const gsl_rng* getRng() const
    {
      return theRng;
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
       Update theVariableVector.

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
       the list of loggers.

       The list, theLoggerVector, is used in log() method.

    */

    void updateLoggerVector();




  protected:

    SystemVector        theSystemVector;

    LoggerVector  theLoggerVector;

    VariableVector            theVariableVector;
    VariableVector::size_type theReadWriteVariableOffset;
    VariableVector::size_type theReadOnlyVariableOffset;

    VariableVector            theIntegratedVariableVector;

    ProcessVector             theProcessVector;
    ProcessVector::size_type  theDiscreteProcessOffset;

    StepperVector             theDependentStepperVector;

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

    gsl_rng*   theRng;

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
