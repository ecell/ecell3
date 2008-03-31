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

#ifndef __PROCESS_HPP
#define __PROCESS_HPP

#include <boost/mem_fn.hpp>
#include <boost/functional.hpp>

#include "AssocVector.h"

#include "libecs.hpp"
#include "Entity.hpp"
#include "VariableReference.hpp"
#include "PartitionedList.hpp"

/**
   @addtogroup entities
 */

/** @file */

/** @{ */

namespace libecs {

/**
   Process class is used to represent chemical and other phenonema which
   may or may not result in change in value of one or more Variables.
 */
LIBECS_DM_CLASS( Process, Entity )
{
public:
    typedef ::std::vector< VariableReference > VarRefVector;
    typedef PartitionedList< 3, VarRefVector > VarRefs;
    typedef ::boost::iterator_range< VarRefVector::iterator >
            VarRefVectorRange;
    typedef ::boost::iterator_range< VarRefVector::const_iterator >
            VarRefVectorCRange;
public:
    LIBECS_DM_BASECLASS( Process );

    LIBECS_DM_OBJECT_ABSTRACT( Process )
    {
        INHERIT_PROPERTIES( Entity );

        PROPERTYSLOT_LOAD_SAVE( Polymorph, VariableReferenceList,
                                &Process::loadVariableReferenceList,
                                &Process::saveVariableReferenceList,
                                &Process::loadVariableReferenceList,
                                &Process::saveVariableReferenceList );

        PROPERTYSLOT_SET_GET( Integer,       Priority );
        PROPERTYSLOT_SET_GET( String,        StepperID );

        PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real,    Activity );
        PROPERTYSLOT_GET_NO_LOAD_SAVE(     Real,    MolarActivity );

        PROPERTYSLOT_GET_NO_LOAD_SAVE(     Integer, IsContinuous );
    }

    /**
       Sort Processes in reversed order of 'Priority' values.
       (Largest one first, smallest one last)
    */
    class PriorityCompare
    {
    public:
        bool operator()( ProcessPtr aLhs, ProcessPtr aRhs ) const
        {
            return compare( aLhs->getPriority(), aRhs->getPriority() );
        }

        bool operator()( ProcessPtr aLhs, IntegerParam aRhs ) const
        {
            return compare( aLhs->getPriority(), aRhs );
        }

        bool operator()( IntegerParam aLhs, ProcessPtr aRhs ) const
        {
            return compare( aLhs, aRhs->getPriority() );
        }

    private:
        // if statement can be faster than returning an expression directly
        inline static bool compare( IntegerParam aLhs, IntegerParam aRhs )
        {
            return aLhs > aRhs;
        }
    };


public:
    virtual ~Process();

    virtual void startup();

    virtual void fire() = 0;

    virtual GET_METHOD( Real, StepInterval )
    {
        return INF;
    }

    /**
       This method returns true if this Process is compatible with
       continuous Steppers.
    */

    virtual const bool isContinuous() const
    {
        return false;
    }

    GET_METHOD( Integer, IsContinuous )
    {
        return isContinuous();
    }

    /**
       Set activity value.

       Semantics of this property can be defined in each subclass of
       Process.  Usually it is a turnover number if the Process represents a
       chemical reaction.

       If the value has time in its dimension, the unit should be [per
       second], not [per step].

       @param anActivity An activity value to be set.
       @see getActivity()
     */
    SET_METHOD( Real, Activity )
    {
        activity_ = value;
    }

    /**
       Get activity value.

       @see setActivity()
       @return the activity value of this Process.
     */
    GET_METHOD( Real, Activity )
    {
        return activity_;
    }

    LOAD_METHOD( VariableReferenceList );
    SAVE_METHOD( VariableReferenceList );

    GET_METHOD( Real, MolarActivity )
    {
        return activity_ / ( getEnclosingSystem()->getSize() * N_A );
    }

    /**
       Set a priority value of this Process.

       The priority is an Integer value which is used to determine the
       order of call to Process::fire() method in Stepper.

       @param aValue the priority value as an Integer.
       @see Stepper
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

    /**
       Register the Stepper of this Process by an ID.

       @param anID Stepper ID.
     */
    SET_METHOD( String, StepperID );

    /**
       Get an ID of the Stepper of this Process.

       @return StepperID as a String.
     */
    GET_METHOD( String, StepperID );

    void removeVariableReference( const String& aName );


    /**
       Register a new VariableReference to varRefs_.

       VarRefs are sorted by coefficients, preserving the relative
       order by the names.

       @param aName name of the VariableReference.
       @param aVariable a Pointer to the Variable.
       @param aCoefficient an Integer value of the coefficient.
    */

    void registerVariableReference( const String& aName,
                                    VariablePtr aVariable,
                                    IntegerParam aCoefficient,
                                    const bool isAccessor = true );

    /**
       Get VariableReference by a tag name.

       @param aVariableReferenceName
       @return a VariableReference
       @see VariableReference
    */

    const VariableReference& getVariableReference(
        const String& aVariableReferenceName ) const;

    /**
       @return a const reference to the VarRefVector
    */
    VarRefVectorCRange getVariableReferences() const
    {
        return VarRefVectorCRange( varRefs_.begin(), varRefs_.end() );
    }

    VarRefVectorRange
    getNegativeVariableReferences()
    {
        return varRefs_.partition_range( 0 );
    }

    VarRefVectorCRange
    getNegativeVariableReferences() const
    {
        return varRefs_.partition_range( 0 );
    }

    VarRefVectorRange getZeroVariableReferences()
    {
        return varRefs_.partition_range( 1 );
    }

    VarRefVectorCRange getZeroVariableReferences() const
    {
        return varRefs_.partition_range( 1 );
    }

    VarRefVectorRange
    getPositiveVariableReferences()
    {
        return varRefs_.partition_range( 2 );
    }

    VarRefVectorCRange
    getPositiveVariableReferences() const
    {
        return varRefs_.partition_range( 2 );
    }

    void setStepper( Stepper* const aStepper );

    /**
       Returns a pointer to a Stepper object that this Process belongs.
       @return A pointer to a Stepper object that this Process, or
       NULLPTR if it is not set yet.
     */
    Stepper* getStepper() const
    {
        return stepper_;
    }

    /**
       Add a value to each of VarRefs.

       For each VariableReference, the new value is:
       old_value + ( aValue * theCoeffiencnt ).

       VarRefs with zero coefficients are skipped for optimization.

       This is a convenient method for use in subclasses.

       @param aValue aReal value to be added.
     */
    void addValue( RealParam aValue )
    {
        setActivity( aValue );

        // Increase or decrease variables, skipping zero coefficients.
        VarRefVectorCRange zeroVarRefs( getZeroVariableReferences() );
        VarRefVectorCRange positiveVarRefs( getPositiveVariableReferences() );
        std::for_each(
                zeroVarRefs.begin(), zeroVarRefs.end(),
                boost::bind2nd( boost::mem_fun_ref(
                    &VariableReference::addValue ), aValue ) );

        std::for_each(
                positiveVarRefs.begin(), positiveVarRefs.end(),
               boost::bind2nd( boost::mem_fun_ref(
                    &VariableReference::addValue ), aValue ) );
    }

    /**
       Set velocity of each VariableReference according to stoichiometry.
       VarRefs with zero coefficients are skipped for optimization.
       This is a convenience method for use in subclasses.
       @param aVelocity a base velocity to be added.
     */
    void setFlux( RealParam aVelocity )
    {
        setActivity( aVelocity );
    }

    /**
       Unset all the product species' isAccessor() bit.
       Product species here means VarRefs those have positive
       non-zero coefficients.
       As a result these becomes write-only VarRefs.
       This method is typically called in initialize() of subclasses.
       This method should be called before getVariableReference().

       This is a convenient method.
     */
    void declareUnidirectional();

    /**
       Check if this Process can affect on a given Process.
     */
    const bool isDependentOn( const Process* proc ) const;

protected:
    VarRefVector::iterator findVariableReference( const String& aName );
    VarRefVector::const_iterator findVariableReference( const String& aName ) const;

    void updateVarRefVector();

protected:
    VarRefs     varRefs_;
    Stepper*    stepper_;
    Real        activity_;
    Integer     priority_;
};

} // namespace libecs

/** @}*/

#endif /* __PROCESS_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
