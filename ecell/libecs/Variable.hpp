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

#ifndef __VARIABLE_HPP
#define __VARIABLE_HPP

#include <utility>

#include "libecs.hpp"
#include "Entity.hpp"
#include "System.hpp"


/**
   @addtogroup entities
 */
/** @{ */
/** @file */

namespace libecs {

class VariableValueIntegrator;

/**
   Variable class represents state variables in the simulation model, such as
   amounts of molecular species in a compartment.

*/
LIBECS_DM_CLASS( Variable, Entity )
{
public:
    LIBECS_DM_BASECLASS( Variable );

    LIBECS_DM_OBJECT( Variable, Variable )
    {
        INHERIT_PROPERTIES( Entity );

        PROPERTYSLOT_LOAD_SAVE( Real, Value,
                                &Variable::setValue,
                                &Variable::getValue,
                                &Variable::loadValue,
                                &Variable::saveValue );

        PROPERTYSLOT( Integer, Fixed, &Variable::setFixed, &Variable::isFixed );

        PROPERTYSLOT_NO_LOAD_SAVE( Real, Velocity,
                                   NOMETHOD,
                                   &Variable::getVelocity );

        PROPERTYSLOT_LOAD_SAVE( Real, MolarConc,
                                &Variable::setMolarConc,
                                &Variable::getMolarConc,
                                &Variable::loadMolarConc,
                                NOMETHOD );

        PROPERTYSLOT_LOAD_SAVE( Real, NumberConc,
                                &Variable::setNumberConc,
                                &Variable::getNumberConc,
                                &Variable::loadNumberConc,
                                NOMETHOD );
    }


    Variable();
    virtual ~Variable();

    /**
       Initializes this variable right before the simulation starts.
    */
    virtual void initialize();

    /**
       This simply sets the value of this Variable if getFixed() is false.
       @see getFixed()
    */
    virtual SET_METHOD( Real, Value )
    {
        if ( fixed_ )
        {
            THROW_EXCEPTION( IllegalOperation,
                    "cannot modify constant variable" );
        }
        value_ = value;
    }

    GET_METHOD( Real, Value )
    {
        return value_;
    }

    void addValue( RealParam aValue )
    {
        setValue( getValue() + aValue );
    }

    LOAD_METHOD( Value )
    {
        value_ = value;
    }

    SAVE_METHOD( Value )
    {
        return Polymorph( value_ );
    }

    /**
       @return current velocity value in (number of molecules)/(step)
    */
    GET_METHOD( Real, Velocity );

    /**

    A wrapper to set Fixed property by a bool value.

    */

    void setFixed( const bool aValue )
    {
        fixed_ = aValue;
    }

    // wrappers to expose is/setFixed as PropertySlots
    SET_METHOD( Integer, Fixed )
    {
        fixed_ = value != 0;
    }

    Integer isFixed()
    {
        return fixed_;
    }

    /**
       Returns the molar concentration of this Variable.
       @return Concentration in M [mol/L].
    */

    GET_METHOD( Real, MolarConc )
    {
        return getNumberConc() * N_A_R;
    }

    /**
       Set the molar concentration of this Variable.
       @param value Concentration in M [mol/L].
    */

    SET_METHOD( Real, MolarConc )
    {
        setNumberConc( value * N_A );
    }

    /**
       Load the molar concentration of this Variable.
       This method uses loadNumberConc() instead of setNumberConc().
       @see setNumberConc()
    */
    LOAD_METHOD( MolarConc )
    {
        loadNumberConc( value.as<Real>() * N_A );
    }

    /**
       Returns the number concentration of this Variable.
       Unlike getMolarConc, this method just returns value / size.
       @return Concentration in [number/L].
    */

    GET_METHOD( Real, NumberConc )
    {
        return getValue() / getSizeOfSuperSystem();
    }

    /**
       Set the number concentration of this Variable.
       @param value concentration in [number/L].
    */
    SET_METHOD( Real, NumberConc )
    {
        setValue( value * getSizeOfSuperSystem() );
    }

    SAVE_METHOD( NumberConc )
    {
        return Polymorph( getNumberConc() );
    }

    LOAD_METHOD( NumberConc )
    {
        setNumberConc( value.as<Real>() );
    }

    void setVariableValueIntegrator( VariableValueIntegrator* integrator )
    {
        integrator_ = integrator_;
    }

    VariableValueIntegrator* getVariableValueIntegrator()
    {
        return integrator_;
    }

    const VariableValueIntegrator* getVariableValueIntegrator() const
    {
        return integrator_;
    }

protected:
    const Real getSizeOfSuperSystem() const
    {
        return getEnclosingSystem()->getSizeVariable()->getValue();
    }

protected:
    Real value_;
    VariableValueIntegrator* integrator_;
    bool fixed_;
};


} // namespace libecs

/** @} */

#endif /* __VARIABLE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
