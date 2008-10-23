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

#include "libecs/libecs.hpp"
#include "libecs/Stepper.hpp"

#include <boost/multi_array.hpp>

/**
   @addtogroup stepper
   @{
*/
namespace libecs
{

typedef boost::multi_array<Real, 2> RealMatrix_;
DECLARE_TYPE( RealMatrix_, RealMatrix );

DECLARE_CLASS( DifferentialStepper );

LIBECS_DM_CLASS( DifferentialStepper, Stepper )
{
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
        PROPERTYSLOT( Real, StepInterval,
                      &DifferentialStepper::initializeStepInterval,
                      &DifferentialStepper::getStepInterval );
        
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, NextStepInterval );
        PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( Real,    TolerableStepInterval );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,    Stage );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,    Order );
    }

    class Interpolant
        : public libecs::Interpolant
    {

    public:
        Interpolant( DifferentialStepperRef aStepper, 
                     VariablePtr const aVariablePtr )
            : libecs::Interpolant( aVariablePtr ),
              theStepper( aStepper ),
              theIndex( theStepper.getVariableIndex( aVariablePtr ) )
        {
            ; // do nothing
        }


        virtual const Real getDifference( RealParam aTime,
                                          RealParam anInterval ) const;

        virtual const Real getVelocity( RealParam aTime ) const;

    protected:

        DifferentialStepperRef        theStepper;
        VariableVector::size_type theIndex;

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

    void resetAll();

    void interIntegrate();

    void initializeVariableReferenceList();

    void setVariableVelocity( boost::detail::multi_array::sub_array<Real, 1> aVelocityBuffer );

    virtual void initialize();

    virtual void reset();

    virtual void interrupt( TimeParam aTime );

    virtual InterpolantPtr createInterpolant( VariablePtr aVariable )
    {
        return new DifferentialStepper::Interpolant( *this, aVariable );
    }

    virtual GET_METHOD( Integer, Stage )
    { 
        return 1; 
    }

    virtual GET_METHOD( Integer, Order )
    { 
        return getStage(); 
    }

    RealMatrixCref getTaylorSeries() const
    {
        return theTaylorSeries;
    }

protected:

    const bool isExternalErrorTolerable() const;

    RealMatrix theTaylorSeries;

    VariableReferenceListVector theVariableReferenceListVector;

    bool theStateFlag;

private:

    Real theNextStepInterval;
    Real theTolerableStepInterval;
};


/**
     ADAPTIVE STEPSIZE DIFFERENTIAL EQUATION SOLVER


*/

DECLARE_CLASS( AdaptiveDifferentialStepper );

LIBECS_DM_CLASS( AdaptiveDifferentialStepper, DifferentialStepper )
{

public:

    LIBECS_DM_OBJECT_ABSTRACT( AdaptiveDifferentialStepper )
    {
        INHERIT_PROPERTIES( DifferentialStepper );

        PROPERTYSLOT_SET_GET( Real, Tolerance );
        PROPERTYSLOT_SET_GET( Real, AbsoluteToleranceFactor );
        PROPERTYSLOT_SET_GET( Real, StateToleranceFactor );
        PROPERTYSLOT_SET_GET( Real, DerivativeToleranceFactor );

        PROPERTYSLOT( Integer, IsEpsilonChecked,
                      &AdaptiveDifferentialStepper::setEpsilonChecked,
                      &AdaptiveDifferentialStepper::isEpsilonChecked );
        PROPERTYSLOT_SET_GET( Real, AbsoluteEpsilon );
        PROPERTYSLOT_SET_GET( Real, RelativeEpsilon );

        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, MaxErrorRatio );
    }

public:

    AdaptiveDifferentialStepper();

    virtual ~AdaptiveDifferentialStepper();

    /**
       Adaptive stepsize control.

       These methods are for handling the standerd error control objects.
    */

    SET_METHOD( Real, Tolerance )
    {
        theTolerance = value;
    }

    GET_METHOD( Real, Tolerance )
    {
        return theTolerance;
    }

    SET_METHOD( Real, AbsoluteToleranceFactor )
    {
        theAbsoluteToleranceFactor = value;
    }

    GET_METHOD( Real, AbsoluteToleranceFactor )
    {
        return theAbsoluteToleranceFactor;
    }

    SET_METHOD( Real, StateToleranceFactor )
    {
        theStateToleranceFactor = value;
    }

    GET_METHOD( Real, StateToleranceFactor )
    {
        return theStateToleranceFactor;
    }

    SET_METHOD( Real, DerivativeToleranceFactor )
    {
        theDerivativeToleranceFactor = value;
    }

    GET_METHOD( Real, DerivativeToleranceFactor )
    {
        return theDerivativeToleranceFactor;
    }

    SET_METHOD( Real, MaxErrorRatio )
    {
        theMaxErrorRatio = value;
    }

    GET_METHOD( Real, MaxErrorRatio )
    {
        return theMaxErrorRatio;
    }

    /**
       check difference in one step
    */

    SET_METHOD( Integer, EpsilonChecked )
    {
        if ( value > 0 ) {
            theEpsilonChecked = true;
        }
        else {
            theEpsilonChecked = false;
        }
    }

    const Integer isEpsilonChecked() const
    {
        return theEpsilonChecked;
    }

    SET_METHOD( Real, AbsoluteEpsilon )
    {
        theAbsoluteEpsilon = value;
    }

    GET_METHOD( Real, AbsoluteEpsilon )
    {
        return theAbsoluteEpsilon;
    }

    SET_METHOD( Real, RelativeEpsilon )
    {
        theRelativeEpsilon = value;
    }

    GET_METHOD( Real, RelativeEpsilon )
    {
        return theRelativeEpsilon;
    }

    virtual void initialize();

    virtual void step();

    virtual bool calculate() = 0;

    virtual GET_METHOD( Integer, Stage )
    { 
        return 2;
    }

private:

    Real safety;
    Real theTolerance;
    Real theAbsoluteToleranceFactor;
    Real theStateToleranceFactor;
    Real theDerivativeToleranceFactor;

    bool theEpsilonChecked;
    Real theAbsoluteEpsilon;
    Real theRelativeEpsilon;

    Real theMaxErrorRatio;
};

} // namespace libecs

/** @} */

#endif /* __DIFFERENTIALSTEPPER_HPP */
