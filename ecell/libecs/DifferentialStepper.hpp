//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
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

#ifndef __DIFFERENTIALSTEPPER_HPP
#define __DIFFERENTIALSTEPPER_HPP

#include "libecs/Defs.hpp"
#include "libecs/Stepper.hpp"

#include <boost/multi_array.hpp>
#include <iostream>

namespace libecs
{

LIBECS_DM_CLASS( DifferentialStepper, Stepper )
{
public:
    typedef boost::multi_array<Real, 2> RealMatrix;

public:
    typedef VariableVector::size_type VariableIndex;
    typedef std::pair< VariableIndex, Integer > ExprComponent;
    typedef std::vector< ExprComponent > VariableReferenceList;
    typedef std::vector< VariableReferenceList > VariableReferenceListVector;

public:

    LIBECS_DM_OBJECT_ABSTRACT( DifferentialStepper )
    {
        INHERIT_PROPERTIES( Stepper );

        // FIXME: load/save ??
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, NextStepInterval );
        PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real,    TolerableStepInterval );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,    Stage );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,    Order );
    }

    class LIBECS_API Interpolant
        : public libecs::Interpolant
    {

    public:
        Interpolant( Variable const* aVariablePtr,
                     Stepper const* aStepper )
            : libecs::Interpolant( aVariablePtr, aStepper ),
              theIndex( theStepper->getVariableIndex( aVariablePtr ) )
        {
            ; // do nothing
        }


        virtual const Real getDifference( Real aTime,
                                          Real anInterval ) const;

        virtual const Real getVelocity( Real aTime ) const;

    protected:

        VariableVector::size_type theIndex;
    };

public:

    DifferentialStepper();

    virtual ~DifferentialStepper();

    SET_METHOD( Real, NextStepInterval )
    {
        theNextStepInterval = std::min( value, theMaxStepInterval );
    }

    GET_METHOD( Real, NextStepInterval )
    {
        return theNextStepInterval;
    }

    SET_METHOD( Real, TolerableStepInterval )
    {
        theTolerableStepInterval = value;
    }

    GET_METHOD( Real, TolerableStepInterval )
    {
        return theTolerableStepInterval;
    }

    virtual SET_METHOD( Real, StepInterval );

    void resetAll();

    void interIntegrate();

    void initializeVariableReferenceList();

    void setVariableVelocity( boost::detail::multi_array::sub_array<Real, 1> aVelocityBuffer );

    virtual void initialize();

    virtual void reset();

    virtual void step();

    virtual void updateInternalState( Real aStepInterval );

    virtual void interrupt( Time aTime );

    virtual libecs::Interpolant* createInterpolant( Variable const* aVariable ) const;

    virtual GET_METHOD( Integer, Stage )
    { 
        return 1; 
    }

    virtual GET_METHOD( Integer, Order )
    { 
        return getStage(); 
    }

    RealMatrix const& getTaylorSeries() const
    {
        return theTaylorSeries;
    }

protected:

    const bool isExternalErrorTolerable() const;

    RealMatrix theTaylorSeries;

    VariableReferenceListVector theVariableReferenceListVector;

protected:

    bool theStateFlag;
    bool isInterrupted;
    Real theNextStepInterval;
    Real theTolerableStepInterval;
};

} // namespace libecs

#endif /* __DIFFERENTIALSTEPPER_HPP */
