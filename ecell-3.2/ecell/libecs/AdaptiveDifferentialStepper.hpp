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

#ifndef __ADAPTIVEDIFFERENTIALSTEPPER_HPP
#define __ADAPTIVEDIFFERENTIALSTEPPER_HPP

#include "libecs/DifferentialStepper.hpp"

/**
   ADAPTIVE STEPSIZE DIFFERENTIAL EQUATION SOLVER
*/
namespace libecs
{
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

#endif /* __ADAPTIVEDIFFERENTIALSTEPPER_HPP */
