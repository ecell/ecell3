//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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

#include <boost/range/iterator_range.hpp>

#include "libecs.hpp"

#include "Polymorph.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"
#include "PropertyInterface.hpp"
#include "PartitionedList.hpp"
#include "PropertiedClass.hpp"

/**
   @addtogroup stepper
   @{
 */

/** @file */

namespace libecs
{

class Model;
class SimulationContext;
class LoggerManager;

/**
   Stepper class defines and governs a computation unit in a model.
   The computation unit is defined as a set of Process objects.
*/
LIBECS_DM_CLASS( Stepper, PropertiedClass )
{
    friend class Model;

public:
    typedef std::vector<Variable*> VariableVector;
    typedef PartitionedList< 3, VariableVector > Variables;
    typedef boost::iterator_range<Variables::iterator> VariableVectorRange;
    typedef boost::iterator_range<Variables::const_iterator> VariableVectorCRange;
    typedef std::vector<Process*> ProcessVector;
    typedef PartitionedList< 2, ProcessVector> Processes;
    typedef boost::iterator_range<Processes::iterator> ProcessVectorRange;
    typedef boost::iterator_range<Processes::const_iterator> ProcessVectorCRange;
    typedef std::vector<System*> SystemSet;
    typedef boost::iterator_range<SystemSet::iterator> SystemSetRange;
    typedef boost::iterator_range<SystemSet::const_iterator> SystemSetCRange;

    typedef std::vector<Real> RealVector;

public:
    LIBECS_DM_OBJECT_ABSTRACT( Stepper )
    {
        INHERIT_PROPERTIES( PropertiedClass );

        PROPERTYSLOT_SET_GET( Integer,   Priority );
        PROPERTYSLOT_SET_GET( Real,      StepInterval );
        PROPERTYSLOT_SET_GET( Real,      MaxStepInterval );
        PROPERTYSLOT_SET_GET( Real,      MinStepInterval );


        // these properties are not loaded/saved.
        PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Real,      CurrentTime );
        PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, ProcessList );
        PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, SystemList );
        PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, ReadVariableList );
        PROPERTYSLOT_GET_NO_LOAD_SAVE    ( Polymorph, WriteVariableList );
    }


    class PriorityComparator
    {
    public:
        bool operator()( StepperPtr lhs, StepperPtr rhs ) const
        {
            return compare( lhs->getPriority(), rhs->getPriority() );
        }

        bool operator()( StepperPtr lhs, IntegerParam rhs ) const
        {
            return compare( lhs->getPriority(), rhs );
        }

        bool operator()( IntegerParam lhs, StepperPtr rhs ) const
        {
            return compare( lhs, rhs->getPriority() );
        }

    private:

        // if statement can be faster than returning an expression directly
        inline static bool compare( IntegerParam lhs, IntegerParam rhs )
        {
            return lhs > rhs;
        }
    };

    virtual ~Stepper();


    /**
       Get the current time of this Stepper.

       The current time is defined as a next scheduled point in time
       of this Stepper.

       @return the current time in Real.
    */

    GET_METHOD( Real, CurrentTime )
    {
        return currentTime_;
    }

    SET_METHOD( Real, CurrentTime )
    {
        currentTime_ = value;
    }

    /**
       This may be overridden in dynamically scheduled steppers.

    */

    virtual SET_METHOD( Real, StepInterval )
    {
        Real aNewStepInterval( value );

        if ( aNewStepInterval > getMaxStepInterval() )
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
        return stepInterval_;
    }

    virtual GET_METHOD( Real, TimeScale )
    {
        return getStepInterval();
    }

    SET_METHOD( String, ID )
    {
        id_ = value;
    }

    GET_METHOD( String, ID )
    {
        return id_;
    }

    SET_METHOD( Real, MinStepInterval )
    {
        minStepInterval_ = value;
    }

    GET_METHOD( Real, MinStepInterval )
    {
        return minStepInterval_;
    }

    SET_METHOD( Real, MaxStepInterval )
    {
        maxStepInterval_ = value;
    }

    GET_METHOD( Real, MaxStepInterval )
    {
        return maxStepInterval_;
    }


    GET_METHOD( Polymorph, WriteVariableList );
    GET_METHOD( Polymorph, ReadVariableList );
    GET_METHOD( Polymorph, ProcessList );
    GET_METHOD( Polymorph, SystemList );

    virtual void startup();

    virtual void initialize();

    void initializeProcesses();

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

    void registerSystem( System* aSystemPtr );

    /**
       Remove a System from this Stepper.
       @note This method is not currently supported.  Calling this method
       causes undefined behavior.
       @param aSystemPtr a pointer to a System object
    */

    void removeSystem( System* aSystemPtr );

    /**
       Register a Process to this Stepper.
       @param aProcessPtr a pointer to a Process object to register
    */

    virtual void registerProcess( Process* aProcessPtr );

    /**
       Remove a Process from this Stepper.
       @note This method is not currently supported.
       @param aProcessPtr a pointer to a Process object
    */

    void removeProcess( Process* aProcessPtr );


    void loadStepInterval( RealParam aStepInterval )
    {
        stepInterval_ = aStepInterval;
    }

    LoggerManager* getLoggerManager() const
    {
        return loggerManager_;
    } 

    void setLoggerManager( LoggerManager* manager )
    {
        loggerManager_ = manager;
    }

    void setSchedulerIndex( const int anIndex )
    {
        schedulerIndex_ = anIndex;
    }

    const int getSchedulerIndex() const
    {
        return schedulerIndex_;
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
        priority_ = value;
    }

    /**
       @see setPriority()
    */

    GET_METHOD( Integer, Priority )
    {
        return priority_;
    }

    const SystemSetCRange getSystems() const
    {
        return SystemSetCRange( systems_.begin(), systems_.end() );
    }

    SystemSetRange getSystems()
    {
        return SystemSetRange( systems_.begin(), systems_.end() );
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
    ProcessVectorCRange getProcesses() const
    {
        return ProcessVectorCRange( processes_.begin(), processes_.end() );
    }

    ProcessVectorRange getProcesses()
    {
        return ProcessVectorRange( processes_.begin(), processes_.end() );
    }

    ProcessVectorCRange getContinuousProcesses() const
    {
        return processes_.partition_range( 0 );
    }

    ProcessVectorRange getContinuousProcesses()
    {
        return processes_.partition_range( 0 );
    }

    ProcessVectorCRange getDiscreteProcesses() const
    {
        return processes_.partition_range( 1 );
    }

    ProcessVectorRange getDiscreteProcesses()
    {
        return processes_.partition_range( 1 );
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
    VariableVectorCRange getInvolvedVariables() const
    {
        return VariableVectorCRange( variables_.begin(), variables_.end() );
    }

    VariableVectorRange getInvolvedVariables()
    {
        return VariableVectorRange( variables_.begin(), variables_.end() );
    }

    /**
       @see getVariables()
    */
    VariableVectorCRange getAffectedVariables() const
    {
        return VariableVectorCRange( variables_.begin( 0 ), variables_.end( 1 ) );
    }

    VariableVectorRange getAffectedVariables()
    {
        return VariableVectorRange( variables_.begin( 0 ), variables_.end( 1 ) );
    }

    /**
       @see getVariables()
    */
    VariableVectorCRange getReadVariables() const
    {
        return VariableVectorCRange( variables_.begin( 1 ), variables_.end( 2 ) );
    }

    VariableVectorRange getReadVariables()
    {
        return VariableVectorRange( variables_.begin( 1 ), variables_.end( 2 ) );
    }

    /**
       @see getVariables()
    */
    VariableVectorCRange getReadWriteVariables() const
    {
        return VariableVectorCRange( variables_.begin( 1 ), variables_.end( 1 ) );
    }

    VariableVectorRange getReadWriteVariables()
    {
        return VariableVectorRange( variables_.begin( 1 ), variables_.end( 1 ) );
    }


    /**
       @see getVariables()
    */
    VariableVectorCRange getReadOnlyVariables() const
    {
        return VariableVectorCRange( variables_.begin( 2 ), variables_.end( 2 ) );
    }

    VariableVectorRange getReadOnlyVariables()
    {
        return VariableVectorRange( variables_.begin( 2 ), variables_.end( 2 ) );
    }

    const RealVector& getValueBuffer() const
    {
        return valueBuffer_;
    }


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
    bool isDependentOn( const StepperCptr aStepper ) const;

    virtual Interpolant* createInterpolant();

    bool operator<( const Stepper& rhs )
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
      This method updates theIntegratedVariableVector.
      theIntegratedVariableVector holds the Variables those
      isIntegrationNeeded() method return true.
      @internal
     */
    void updateIntegratedVariableVector();

    /**
       Create Interpolant objects and distribute the objects to
       write Variables.

       Ownership of the Interpolant objects are given away to the Variables.

       @see Variable::registerInterpolant()
    */
    void createInterpolants();

    void loadVariablesToBuffer();

    void saveBufferToVariables( bool onlyAffected = true );

    const Variables::size_type getVariableIndex( const Variable* const var ) const
    { 
        VariableVectorCRange range( getInvolvedVariables() );
        Variables::const_iterator pos(
                std::find( range.begin(), range.end(), var ) );
        if ( pos == range.end() )
        {
            THROW_EXCEPTION( NotFound,
                    String( "no such variable involved: " ) + var->asString() );
        }

        return pos - range.begin();
    }

protected:
    void __setID( const String& id )
    {
        id_ = id;
    }

protected:
    SystemSet                    systems_;
    Variables                    variables_;
    VariableVector               variablesToIntegrate_;
    Processes                    processes_;
    RealVector                   valueBuffer_;
    LoggerManager*               loggerManager_;

    /** the index on the scheduler */
    int                 schedulerIndex_;
    Integer             priority_;
    Real                currentTime_;
    Real                stepInterval_;
    Real                minStepInterval_;
    Real                maxStepInterval_;
    String              id_;
};

} // namespace libecs

/** @} */

#endif /* __STEPPER_HPP */
/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
