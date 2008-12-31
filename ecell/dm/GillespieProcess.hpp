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
#ifndef __GILLESPIEPROCESS_HPP
#define __GILLESPIEPROCESS_HPP

#include <limits>

#include <gsl/gsl_rng.h>

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>
#include <libecs/MethodProxy.hpp>

LIBECS_DM_CLASS( GillespieProcess, libecs::Process )
{
    typedef libecs::MethodProxy< GillespieProcess, libecs::Real > RealMethodProxy;
    typedef const libecs::Real( GillespieProcess::* PDMethodPtr )( libecs::Variable* ) const; 
    
public:
    
    LIBECS_DM_OBJECT( GillespieProcess, Process );
    
    GillespieProcess() ;

    virtual ~GillespieProcess();

    // c means stochastic reaction constant
    SIMPLE_SET_GET_METHOD( libecs::Real, k );
    SIMPLE_SET_GET_METHOD( libecs::Real, c );
    
    GET_METHOD( libecs::Real, Propensity );

    GET_METHOD( libecs::Real, Propensity_R );

    const libecs::Real getPD( libecs::Variable* aVariable ) const;

    virtual const bool isContinuous() const;

    // The order of the reaction, i.e. 1 for a unimolecular reaction.

    GET_METHOD( libecs::Integer, Order );

    //    virtual void updateStepInterval()
    virtual GET_METHOD( libecs::Real, StepInterval );

    void calculateOrder();

    virtual void initialize();

    virtual void fire();

protected:

    const libecs::Real getZero() const;

    const libecs::Real getPD_Zero( libecs::Variable* aVariable ) const;

    /**
         FirstOrder_OneSubstrate
     */
    const libecs::Real getPropensity_FirstOrder() const;

    const libecs::Real getMinValue_FirstOrder() const;

    const libecs::Real getPD_FirstOrder( libecs::Variable* aVariable ) const;

    /**
         SecondOrder_TwoSubstrates
     */
    const libecs::Real getPropensity_SecondOrder_TwoSubstrates() const;

    const libecs::Real getMinValue_SecondOrder_TwoSubstrates() const;

    const libecs::Real getPD_SecondOrder_TwoSubstrates( libecs::Variable* aVariable ) const;

    /**
         SecondOrder_OneSubstrate
     */
    const libecs::Real getPropensity_SecondOrder_OneSubstrate() const;

    const libecs::Real getMinValue_SecondOrder_OneSubstrate() const;

    const libecs::Real getPD_SecondOrder_OneSubstrate( libecs::Variable* aVariable ) const;

protected:

    libecs::Real k;
    libecs::Real c;

    libecs::Integer theOrder;

    RealMethodProxy theGetPropensityMethodPtr;    
    RealMethodProxy theGetMinValueMethodPtr;
    PDMethodPtr     theGetPDMethodPtr; // this should be MethodProxy

};


inline void GillespieProcess::calculateOrder()
{
    theOrder = 0;
        
    for( libecs::VariableReferenceVectorConstIterator i(
            theVariableReferenceVector.begin() );
         i != theVariableReferenceVector.end() ; ++i )
    {
        libecs::VariableReference const& aVariableReference( *i );
        const libecs::Integer aCoefficient( aVariableReference.getCoefficient() );
            
        // here assume aCoefficient != 0
        if( aCoefficient == 0 )
        {
            THROW_EXCEPTION( libecs::InitializationFailed,
                             asString() + ": Zero stoichiometry is not allowed." );
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
        theGetPropensityMethodPtr =
                    RealMethodProxy::create<&GillespieProcess::getZero>();
        theGetMinValueMethodPtr   =
                    RealMethodProxy::create<&GillespieProcess::getZero>();
        theGetPDMethodPtr         = &GillespieProcess::getPD_Zero;
    }
    else if( getOrder() == 1 )     // one substrate, first order.
    {
        theGetPropensityMethodPtr =
                    RealMethodProxy::create<&GillespieProcess::getPropensity_FirstOrder>();
        theGetMinValueMethodPtr   =
                    RealMethodProxy::create<&GillespieProcess::getMinValue_FirstOrder>();
        theGetPDMethodPtr         = &GillespieProcess::getPD_FirstOrder;
    }
    else if( getOrder() == 2 )
    {
        if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
        {    
            theGetPropensityMethodPtr   = RealMethodProxy::create<
                    &GillespieProcess::getPropensity_SecondOrder_TwoSubstrates > ();
            theGetMinValueMethodPtr     = RealMethodProxy::create<
                    &GillespieProcess::getMinValue_SecondOrder_TwoSubstrates >();
            theGetPDMethodPtr           =
                    &GillespieProcess::getPD_SecondOrder_TwoSubstrates;
        }
        else // one substrate, second order (coeff == -2)
        {
            theGetPropensityMethodPtr = RealMethodProxy::create<
                    &GillespieProcess::getPropensity_SecondOrder_OneSubstrate>();
            theGetMinValueMethodPtr   = RealMethodProxy::create<
                    &GillespieProcess::getMinValue_SecondOrder_OneSubstrate>();
            theGetPDMethodPtr         =
                    &GillespieProcess::getPD_SecondOrder_OneSubstrate;
        }
    }
    else
    {
        //FIXME: generic functions should come here.
        theGetPropensityMethodPtr = RealMethodProxy::create<
                &GillespieProcess::getZero>();
        theGetPropensityMethodPtr = RealMethodProxy::create<
                &GillespieProcess::getZero>();
        theGetPDMethodPtr         = &GillespieProcess::getPD_Zero;
    }



    //
    if ( theOrder == 1 ) 
    {
        c = k;
    }
    else if ( theOrder == 2 && getZeroVariableReferenceOffset() == 1 )
    {
        c = k * 2.0 / ( libecs::N_A * getSuperSystem()->getSize() );
    }
    else if ( theOrder == 2 && getZeroVariableReferenceOffset() == 2 )
    {
        c = k / ( libecs::N_A * getSuperSystem()->getSize() );
    }
    else
    {
        NEVER_GET_HERE;
    } 
}

#endif /* __GILLESPIEPROCESS_HPP */
