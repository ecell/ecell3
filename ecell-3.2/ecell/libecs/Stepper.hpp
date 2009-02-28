//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __STEPPER_HPP
#define __STEPPER_HPP

#ifdef DLL_EXPORT
#undef DLL_EXPORT
#include <gsl/gsl_rng.h>
#define DLL_EXPORT
#else
#include <gsl/gsl_rng.h>
#endif

#include "libecs/Defs.hpp"

#include "libecs/Util.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/Interpolant.hpp"
#include "libecs/PropertyInterface.hpp"

namespace libecs
{

class Model;

/**
   Stepper class defines and governs a computation unit in a model.

   The computation unit is defined as a set of Process objects.
*/
LIBECS_DM_CLASS( Stepper, EcsObject )
{
public:
    DECLARE_VECTOR( Real, RealVector );
    DECLARE_VECTOR( Variable*, VariableVector );
    DECLARE_VECTOR( Process*, ProcessVector );
    DECLARE_VECTOR( System*, SystemVector );

    typedef VariableVector::size_type VariableIndex;

    LIBECS_DM_BASECLASS( Stepper );

    LIBECS_DM_OBJECT_ABSTRACT( Stepper )
    {
        INHERIT_PROPERTIES( EcsObject );
        
        PROPERTYSLOT_SET_GET( Integer, Priority );
        PROPERTYSLOT_SET_GET( Real,    StepInterval );
        PROPERTYSLOT_SET_GET( Real,    MaxStepInterval );
        PROPERTYSLOT_SET_GET( Real,    MinStepInterval );
        PROPERTYSLOT_SET( String, RngSeed );

        // these properties are not loaded/saved.
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real,      CurrentTime );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Polymorph, ProcessList );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Polymorph, SystemList );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Polymorph, ReadVariableList );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Polymorph, WriteVariableList );
    }


    class PriorityCompare
    {
    public:
        bool operator()( StepperPtr aLhs, StepperPtr aRhs ) const
        {
            return compare( aLhs->getPriority(), aRhs->getPriority() );
        }

        bool operator()( StepperPtr aLhs, IntegerParam aRhs ) const
        {
            return compare( aLhs->getPriority(), aRhs );
        }

        bool operator()( IntegerParam aLhs, StepperPtr aRhs ) const
        {
            return compare( aLhs, aRhs->getPriority() );
        }

    private:
        // if statement can be faster than returning an expression directly
        static bool compare( IntegerParam aLhs, IntegerParam aRhs )
        {
            return aLhs > aRhs;
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

    SET_METHOD( String, RngSeed );

    GET_METHOD( String, RngType );

    virtual void initialize();

    /**             
       Each subclass of Stepper defines this.

       @note Subclass of Stepper must call this by Stepper::calculate() from
       their step().
    */
    virtual void step() = 0;

    virtual void integrate( RealParam aTime );

    /**
       Let the Loggers log data.

       The default behavior is to call all the Loggers attached to
       any Entities related to this Stepper.
    */
    virtual void log();
    
    /**
       Register a System to this Stepper.

       @param aSystemPtr a pointer to a System object to register
    */
    void registerSystem( SystemPtr aSystemPtr );

    /**
       Remove a System from this Stepper.

       @param aSystemPtr a pointer to a System object
    */
    void unregisterSystem( SystemPtr aSystemPtr );

    /**
       Remove all the associated System from this Stepper.

       @param aSystemPtr a pointer to a System object
    */
    void unregisterAllSystem();

    /**
         Register a Process to this Stepper.

         @param aProcessPtr a pointer to a Process object to register
    */
    void registerProcess( ProcessPtr aProcessPtr );

    /**
       Remove a Process from this Stepper.

       @note This method is not currently supported.

       @param aProcessPtr a pointer to a Process object
    */
    void unregisterProcess( ProcessPtr aProcessPtr );


    void loadStepInterval( RealParam aStepInterval )
    {
        theStepInterval = aStepInterval;
    }

    void registerLogger( LoggerPtr );

    void setSchedulerIndex( const int anIndex )
    {
        theSchedulerIndex = anIndex;
    }

    const int getSchedulerIndex() const
    {
        return theSchedulerIndex;
    }

    /**
       Set a priority value of this Stepper.

       The priority is an Int value which is used to determine the
       order of step when more than one Stepper is scheduled at the
       same point in time (such as starting up: t=0).     

       Larger value means higher priority, and called first.

       @param value the priority value as an Int.
       @see Scheduler
    */
    SET_METHOD( Integer, Priority )
    {
        thePriority = value;
    }

    /**
       @see setPriority()
    */
    GET_METHOD( Integer, Priority )
    {
        return thePriority;
    }

    SystemVector const& getSystemVector() const
    {
        return theSystemVector;
    }

    /**
       Get the reference to the ProcessVector of this Stepper.

       The ProcessVector holds a set of pointers to this Stepper's Processes.

       The ProcessVector is partitioned in this way:

       |    Continuous Processes    |    Discrete Processes |

       getDiscreteProcessOffset() method returns the offset (index number)
       of the first discrete Process in this Stepper.

       Each part of the ProcessVector is sorted by Priority properties
       of Processes.

    */
    ProcessVector const& getProcessVector() const
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
    VariableVector const& getVariableVector() const
    {
        return theVariableVector;
    }

    /**
       @see getVariableVector()
    */
    const VariableVector::size_type getReadWriteVariableOffset() const
    {
        return theReadWriteVariableOffset;
    }

    /**
       @see getVariableVector()
    */
    const VariableVector::size_type getReadOnlyVariableOffset() const
    {
        return theReadOnlyVariableOffset;
    }


    RealVector const& getValueBuffer() const
    {
        return theValueBuffer;
    }


    const VariableIndex getVariableIndex( Variable const* aVariable );

    virtual void interrupt( TimeParam aTime ) = 0;

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
    const bool isDependentOn( Stepper const* aStepper );

    /** 
       This method updates theIntegratedVariableVector.

       theIntegratedVariableVector holds the Variables those
       isIntegrationNeeded() method return true.     
    
       This method must be called after initialize().

       @internal
     */
    void updateIntegratedVariableVector();


    virtual InterpolantPtr createInterpolant( Variable* aVariable )
    {
        return new Interpolant( aVariable );
    }

    const gsl_rng* getRng() const
    {
        return theRng;
    }

    bool operator<( Stepper const& rhs )
    {
        return getCurrentTime() < rhs.getCurrentTime();
    }

protected:
    void clearVariables();

    void fireProcesses();

    virtual void reset();


    /**
       Update theProcessVector.
    */
    void updateProcessVector();


    /**
       Update theVariableVector.
    */
    void updateVariableVector();


    /**
       Create Interpolant objects and distribute the objects to 
       write Variables.

       Ownership of the Interpolant objects are given away to the Variables.

       @see Variable::registerInterpolant()
    */
    void createInterpolants();


    /**
       Get a string representation of this Entity as String.

       @return a description string of this Entity.
    */
    virtual String asString() const;

protected:
    SystemVector              theSystemVector;

    VariableVector            theVariableVector;
    VariableVector::size_type theReadWriteVariableOffset;
    VariableVector::size_type theReadOnlyVariableOffset;

    VariableVector            theIntegratedVariableVector;

    ProcessVector             theProcessVector;
    ProcessVector::size_type  theDiscreteProcessOffset;

    RealVector                theValueBuffer;

private:
    /** the index on the scheduler */
    int                       theSchedulerIndex;

    Integer                   thePriority;

    Real                      theCurrentTime;

    Real                      theStepInterval;

    Real                      theMinStepInterval;
    Real                      theMaxStepInterval;

    String                    theID;

    gsl_rng*                  theRng;
};

} // namespace libecs

#endif /* __STEPPER_HPP */
