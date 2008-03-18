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

#ifndef __DIFFERENTIALSTEPPER_HPP
#define __DIFFERENTIALSTEPPER_HPP

#include <boost/multi_array.hpp>

#include "libecs.hpp"
#include "Stepper.hpp"

/**
   @addtogroup stepper
   @{
 */
/** @file */

namespace libecs {

LIBECS_DM_CLASS( DifferentialStepper, Stepper )
{
public:
    typedef VariableVector::size_type VariableIndex;
    typedef boost::multi_array<Real, 2> RealMatrix;
    typedef std::pair< VariableIndex, Integer > ExprComponent;
    typedef std::vector< ExprComponent > VarRefs;
    typedef std::vector< VarRefs > VarRefsOfProcesses;

public:

    LIBECS_DM_OBJECT_ABSTRACT( DifferentialStepper )
    {
        INHERIT_PROPERTIES( Stepper );

        // FIXME: load/save ??
        PROPERTYSLOT( Real, StepInterval,
                      &DifferentialStepper::initializeStepInterval,
                      &DifferentialStepper::getStepInterval );

        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, NextStepInterval );
        PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real,  TolerableStepInterval );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Stage );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Order );
    }

    class Interpolant: public libecs::Interpolant
    {
    public:
        Interpolant( DifferentialStepperRef aStepper )
            : libecs::Interpolant(), theStepper( aStepper ),
              theIndex( 0 )
        {
            ; // do nothing
        }

        virtual void setVariable( Variable* var );

        virtual const Real getDifference( RealParam aTime,
                                          RealParam anInterval ) const;

        virtual const Real getVelocity( RealParam aTime ) const;

    protected:

        DifferentialStepperRef    theStepper;
        Variables::size_type      theIndex;
    };

public:

    DifferentialStepper();

    virtual ~DifferentialStepper();

    SET_METHOD( Real, NextStepInterval )
    {
        theNextStepInterval = value;
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

    void initializeStepInterval( RealParam aStepInterval )
    {
        setStepInterval( aStepInterval );
        setTolerableStepInterval( aStepInterval );
        setNextStepInterval( aStepInterval );
    }

    void interIntegrate();

    void initializeVariableReferenceList();

    void setVariableVelocity( boost::detail::multi_array::sub_array<Real, 1> aVelocityBuffer );

    virtual void initialize();

    virtual void reset();

    virtual void interrupt( TimeParam aTime );

    virtual libecs::Interpolant* createInterpolant();

    virtual GET_METHOD( Integer, Stage )
    {
        return 1;
    }

    virtual GET_METHOD( Integer, Order )
    {
        return getStage();
    }

    const RealMatrix& getTaylorSeries() const
    {
        return theTaylorSeries;
    }

    virtual void registerProcess( Process* aProcessPtr );

protected:
    ExprComponent toExprComponent( const VariableReference varRef ) const;

    static void interIntegrateVariable( Variable* var, TimeParam aCurrentTime );

protected:

    const bool isExternalErrorTolerable() const;
    RealMatrix theTaylorSeries;
    VarRefsOfProcesses varRefsOfProcesses_;

    bool theStateFlag;

private:

    Real theNextStepInterval;
    Real theTolerableStepInterval;
};

} // namespace libecs

#endif /* __DIFFERENTIALSTEPPER_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
