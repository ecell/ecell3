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
// written by Tomoya Kitayama <tomo@e-cell.org>, 
// E-Cell Project.
//

#include <gsl/gsl_randist.h>

#include <libecs/libecs.hpp>
#include <libecs/ContinuousProcess.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>

#include "GillespieProcessInterface.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_EXTRA_1( TauLeapProcess, ContinuousProcess,
                         GillespieProcessInterface )
{
    typedef Real (TauLeapProcess::* getPropensityMethodPtr)() const;
    typedef Real (TauLeapProcess::* getPDMethodPtr)( Variable const* ) const;

public:

    LIBECS_DM_OBJECT( TauLeapProcess, Process )
    {
        INHERIT_PROPERTIES( ContinuousProcess );
        PROPERTYSLOT_SET_GET( Real, k );

        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Propensity );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,    Order );
    }
    
    TauLeapProcess() 
        : theOrder( 0 ), k( 0.0 ),
          theGetPropensityMethodPtr( &TauLeapProcess::getZero ),
          theGetPDMethodPtr( &TauLeapProcess::getZero )
    {
        ; // do nothing
    }

    virtual ~TauLeapProcess()
    {
        ; // do nothing
    }

    SIMPLE_SET_GET_METHOD( Real, k );
    
    GET_METHOD( Integer, Order )
    {
        return theOrder;
    }
 
    virtual GET_METHOD( Real, Propensity )
    {
        return ( this->*theGetPropensityMethodPtr )();
    }

    virtual Real getPD( Variable const* value )const
    {
        return ( this->*theGetPDMethodPtr )( value );
    }

    virtual void initialize()
    {
        ContinuousProcess::initialize();

        calculateOrder();
        
        if( ! ( getOrder() == 1 || getOrder() == 2 ) )
        {
            THROW_EXCEPTION_INSIDE( ValueError, 
                                   asString() +
                                   ": either first or second order scheme is "
                                   "allowed" );
        }
    }    

    virtual void fire()
    {
        setFlux( gsl_ran_poisson( getStepper()->getRng(), getPropensity() ) );
    }
    
protected:

    void calculateOrder()
    {
        theOrder = 0;

        for( VariableReferenceVector::const_iterator i(
                theVariableReferenceVector.begin() );
             i != theVariableReferenceVector.end() ; ++i )
        {
            VariableReference const& aVariableReference( *i );
            const Integer aCoefficient( aVariableReference.getCoefficient() );

            // here assume aCoefficient != 0
            if( aCoefficient == 0 )
            {
                THROW_EXCEPTION_INSIDE( InitializationFailed,
                                       asString() + ": zero stoichiometry is "
                                       "not allowed" );
            }

            if( aCoefficient < 0 )
            {
                // sum the coefficient to get the order of this reaction.
                theOrder -= aCoefficient;
            }
        }

        // set theGetPropensityMethodPtr and theGetMinValueMethodPtr

        if( getOrder() == 0 )     // no substrate
        {
            theGetPropensityMethodPtr = &TauLeapProcess::getZero;
            theGetPDMethodPtr = &TauLeapProcess::getZero;
        }
        else if( getOrder() == 1 )     // one substrate, first order.
        {
            theGetPropensityMethodPtr = &TauLeapProcess::getPropensity_FirstOrder;
            theGetPDMethodPtr = &TauLeapProcess::getPD_FirstOrder;
        }
        else if( getOrder() == 2 )
        {
            if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
            {
                theGetPropensityMethodPtr =
                    &TauLeapProcess::getPropensity_SecondOrder_TwoSubstrates;
                theGetPDMethodPtr =
                    &TauLeapProcess::getPD_SecondOrder_TwoSubstrates;
            }
            else // one substrate, second order (coeff == -2)
            {
                theGetPropensityMethodPtr =
                    &TauLeapProcess::getPropensity_SecondOrder_OneSubstrate;
                theGetPDMethodPtr =
                    &TauLeapProcess::getPD_SecondOrder_OneSubstrate;
            }
        }
        else
        {
            //FIXME: generic functions should come here.
            theGetPropensityMethodPtr = &TauLeapProcess::getZero;
            theGetPDMethodPtr = &TauLeapProcess::getZero;
        }
    }
    
    Real getZero( Variable const * value ) const
    {
        return 0.0;
    }

    Real getZero( ) const
    {
        return 0.0;
    }
        
    Real getPropensity_FirstOrder() const
    {
        const Real 
            aMultiplicity( theVariableReferenceVector[0].getVariable()->getValue() );
        
        if( aMultiplicity > 0.0 )
        {
            return k * aMultiplicity;
        }
        else
        {
            return 0.0;
        }
    }

    Real getPD_FirstOrder( Variable const* value ) const
    {
        if( theVariableReferenceVector[0].getVariable() == value )
        {
            return k;
        }
        else
        {
            return 0.0;
        }
    }

    Real getPropensity_SecondOrder_TwoSubstrates() const
    {
        const Real aMultiplicity( theVariableReferenceVector[0].getVariable()->getValue() *
                                   theVariableReferenceVector[1].getVariable()->getValue() );
        
        if( aMultiplicity > 0.0 )
        {
            return ( k * aMultiplicity ) /
                ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
        }
        else
        {
            return 0;
        }
    }

    Real getPD_SecondOrder_TwoSubstrates( Variable const* value ) const
    {
        if( theVariableReferenceVector[0].getVariable() == value )
        {
            return ( k * theVariableReferenceVector[1].getVariable()->getValue() ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
        }
        else if( theVariableReferenceVector[1].getVariable() == value )
        {
            return ( k * theVariableReferenceVector[0].getVariable()->getValue() ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
        }
        else
        {
            return 0;
        }
    }
    
    Real getPropensity_SecondOrder_OneSubstrate() const
    {
        Variable const* aVariable( theVariableReferenceVector[ 0 ].getVariable() );
        const Real aValue( aVariable->getValue() );
        
        if( aValue > 1.0 ) // there must be two or more molecules
        {
            return ( k * aValue * ( aValue - 1.0 ) ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
                
        }
        else if( aValue < 0.0 )
        {
            THROW_EXCEPTION_INSIDE( SimulationError,
                                    asString() + ": the value of Variable ["
                                    + aVariable->asString()
                                    + "] went under 0" );
        }
        return 0.0;
    }

    Real getPD_SecondOrder_OneSubstrate( Variable const* value ) const
    {
        if( theVariableReferenceVector[0].getVariable() == value )
        {
            Variable const* aVariable( theVariableReferenceVector[ 0 ].getVariable() );
            const Real aValue( aVariable->getValue() );
            if( aValue > 1.0 ) // there must be two or more molecules
            {
                return ( ( 2 * k * aValue - k ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A ) );
            }
            else if( aValue < 0.0 )
            {
                THROW_EXCEPTION_INSIDE( SimulationError,
                                        asString() + ": the value of Variable ["
                                        + aVariable->asString()
                                        + "] went under 0" );
            }
        }
        return 0.0;
    }
    
protected:
    
    Integer theOrder;
    Real k;
    
    getPropensityMethodPtr theGetPropensityMethodPtr;
    getPDMethodPtr theGetPDMethodPtr;
    
};

LIBECS_DM_INIT( TauLeapProcess, Process );
