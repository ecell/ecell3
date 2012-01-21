//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

#include <vector>
#include <boost/mem_fn.hpp>
#include <boost/functional.hpp>

#include "libecs/AssocVector.h"
#include "libecs/Defs.hpp"
#include "libecs/AssocVector.h"
#include "libecs/Entity.hpp"
#include "libecs/VariableReference.hpp"

namespace libecs
{

/**
   Process class is used to represent chemical and other phenonema which 
   may or may not result in change in value of one or more Variables.
*/

LIBECS_DM_CLASS( Process, Entity )
{
public:
    typedef Loki::AssocVector< String, VariableReference,
                               std::less< String > > VariableReferenceMap;
    typedef std::vector< VariableReference > VariableReferenceVector;

public:

    LIBECS_DM_BASECLASS( Process );

    LIBECS_DM_OBJECT_ABSTRACT( Process )
    {
        INHERIT_PROPERTIES( Entity );

        PROPERTYSLOT_LOAD_SAVE( Polymorph, VariableReferenceList,
                                &Process::setVariableReferenceList,
                                &Process::getVariableReferenceList,
                                &Process::setVariableReferenceList,
                                &Process::saveVariableReferenceList );

        PROPERTYSLOT_SET_GET( Integer, Priority );
        PROPERTYSLOT_SET_GET( String,  StepperID );

        PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real, Activity );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, MolarActivity );

        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer, IsContinuous );
    }

    /** 
       Sort Processes in reversed order of 'Priority' values.
       (Largest one first, smallest one last)
    */
    class PriorityCompare
    {
    public:
        bool operator()( Process const* aLhs, Process const* aRhs ) const
        {
            return compare( aLhs->getPriority(), aRhs->getPriority() );
        }

        bool operator()( Process const* aLhs, Integer aRhs ) const
        {
            return compare( aLhs->getPriority(), aRhs );
        }

        bool operator()( Integer aLhs, Process const* aRhs ) const
        {
            return compare( aLhs, aRhs->getPriority() );
        }

    private:

        // if statement can be faster than returning an expression directly
        inline static bool compare( Integer aLhs, Integer aRhs )
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

public:
    Process();

    virtual ~Process();

    virtual EntityType getEntityType() const
    {
        return EntityType( EntityType::PROCESS );
    }

    virtual void preinitialize();

    virtual void initialize();
    
    virtual void fire() = 0;
    
    virtual GET_METHOD( Real, StepInterval )
    {
        return INF;
    }


    /**
       This method returns true if this Process is compatible with 
       continuous Steppers.
    */
    virtual bool isContinuous() const
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
       Process.    Usually it is a turnover number if the Process represents a
       chemical reaction.

       If the value has time in its dimension, the unit should be [per
       second], not [per step].

       @param anActivity An activity value to be set.
       @see getActivity()
    */
    SET_METHOD( Real, Activity )
    { 
        theActivity = value; 
    }


    /**
       Get activity value.

       @see setActivity()
       @return the activity value of this Process.
    */
    GET_METHOD( Real, Activity )
    {
        return theActivity;
    }

    SET_METHOD( Polymorph, VariableReferenceList );
    GET_METHOD( Polymorph, VariableReferenceList );
    SAVE_METHOD( Polymorph, VariableReferenceList );


    GET_METHOD( Real, MolarActivity )
    {
        return theActivity / ( getSuperSystem()->getSize() * N_A );
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
        thePriority = value;
    }


    /**
       @see setPriority()
    */
    GET_METHOD( Integer, Priority )
    {
        return thePriority;
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


    /**
       Register a new VariableReference to theVariableReferenceVector.

       VariableReferences are sorted by coefficients, preserving the relative
       order by the names.

       @param aName name of the VariableReference. 
       @param aFullID the FullID of the target variable
       @param aCoefficient an Integer value of the coefficient.
       @param isAccessor true if the specified variable affects the Process's
              behavior.
       @return a serial number that refers to the registered variable reference.
    */
    Integer registerVariableReference( String const& aName, 
                                       FullID const& aFullID, 
                                       Integer aCoefficient, 
                                       bool isAccessor = true );

    /**
       Register a new anonymous VariableReference to theVariableReferenceVector.

       VariableReferences are sorted by coefficients, preserving the relative
       order by the names.

       @param aFullID the FullID of the target variable.
       @param aCoefficient an Integer value of the coefficient.
       @param isAccessor true if the specified variable affects the Process's
              behavior.
       @return a serial number that refers to the registered variable reference.
    */
    Integer registerVariableReference( FullID const& aFullID, 
                                       Integer aCoefficient, 
                                       bool isAccessor = true );

    /**
       Register a new VariableReference to theVariableReferenceVector.

       VariableReferences are sorted by coefficients, preserving the relative
       order by the names.

       @param aName name of the VariableReference. 
       @param aVariable target Variable
       @param aCoefficient an Integer value of the coefficient.
       @param isAccessor true if the specified variable affects the Process's
              behavior.
       @return a serial number that refers to the registered variable reference.
    */
    Integer registerVariableReference( String const& aName, 
                                       Variable* aVariable, 
                                       Integer aCoefficient, 
                                       bool isAccessor = true );

    /**
       Register a new anonymous VariableReference to theVariableReferenceVector.

       VariableReferences are sorted by coefficients, preserving the relative
       order by the names.

       @param aVariable target Variable
       @param aCoefficient an Integer value of the coefficient.
       @param isAccessor true if the specified variable affects the Process's
              behavior.
       @return a serial number that refers to the registered variable reference.
    */
    Integer registerVariableReference( Variable* aVariable, 
                                       Integer aCoefficient, 
                                       bool isAccessor = true );


    bool removeVariableReference( String const& aName, bool raiseException = true );

    bool removeVariableReference( Integer anID, bool raiseException = true );

    bool removeVariableReference( Variable const* aVariable, bool raiseException = true );


    /**
       Get VariableReference by the tag name.

       @param aVariableReferenceName
       @return a VariableReference
       @see VariableReference
    */
    VariableReference const& getVariableReference( String const& aVariableReferenceName ) const;

    /**
       Get VariableReference by the serial number.

       @param aVariableReferenceName
       @return a VariableReference
       @see VariableReference
    */
    VariableReference const& getVariableReference( Integer anID ) const;

    /**
       Get VariableReference by the variable

       @param aVariableReferenceName
       @return a VariableReference
       @see VariableReference
    */
    VariableReference const& getVariableReference( Variable const* anID ) const;

    /**
       @return a const reference to the VariableReferenceVector
    */
    VariableReferenceVector const& getVariableReferenceVector() const
    {
        return theVariableReferenceVector;
    }

    VariableReferenceVector::size_type getZeroVariableReferenceOffset() const
    {
        return theZeroVariableReferenceIterator - getVariableReferenceVector().begin();
    }

    VariableReferenceVector::size_type 
    getPositiveVariableReferenceOffset() const
    {
        return thePositiveVariableReferenceIterator - getVariableReferenceVector().begin();
    }

    void setStepper( Stepper* aStepper );

    /**
       Returns a pointer to a Stepper object that this Process belongs.

       @return A pointer to a Stepper object that this Process, or
       NULLPTR if it is not set yet.
    */
    Stepper* getStepper() const
    {
        return theStepper;
    }

    /**
       Add a value to each of VariableReferences.

       For each VariableReference, the new value is: 
       old_value + ( aValue * theCoeffiencnt ).

       VariableReferences with zero coefficients are skipped for optimization.

       This is a convenient method for use in subclasses.

       @param aValue aReal value to be added.
    */
    void addValue( libecs::Param<Real>::type aValue );


    /**
       Set velocity of each VariableReference according to stoichiometry.

       VariableReferences with zero coefficients are skipped for optimization.

       This is a convenient method for use in subclasses.

       @param aVelocity a base velocity to be added.
    */
    void setFlux( libecs::Param<Real>::type aVelocity )
    {
        setActivity( aVelocity );
    }


    /**
       Unset all the product species' isAccessor() bit.

       Product species here means VariableReferences those have positive 
       non-zero coefficients.

       As a result these becomes write-only VariableReferences.

       This method is typically called in initialize() of subclasses.
       This method should be called before getVariableReference().

       This is a convenient method.
    */
    void declareUnidirectional();


    /**
       Check if this Process can affect on a given Process.
    */
    bool isDependentOn( Process const* aProcessPtr ) const;

    virtual void detach();

protected:
    VariableReferenceVector::iterator findVariableReference( String const& aName );

    VariableReferenceVector::const_iterator findVariableReference( String const& aName ) const;

    VariableReferenceVector::iterator findVariableReference( Integer anID );

    VariableReferenceVector::const_iterator findVariableReference( Integer anID ) const; 

    static void addValue( VariableReference const& aVarRef, Real value );

    void updateVariableReferenceVector();

    void resolveVariableReferences();

protected:
    VariableReferenceVector theVariableReferenceVector;
    VariableReferenceVector::iterator theZeroVariableReferenceIterator;
    VariableReferenceVector::iterator thePositiveVariableReferenceIterator;

private:
    Stepper*      theStepper;
    Real          theActivity;
    Integer       thePriority;
    Integer       theNextSerial;
};

} // namespace libecs

#endif /* __PROCESS_HPP */
