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

struct GillespieProcessInterface
{
    virtual GET_METHOD( libecs::Real, Propensity ) = 0;

    virtual const libecs::Real getPD( libecs::Variable* aVariable ) const = 0;
};

LIBECS_DM_CLASS_EXTRA_1( GillespieProcess, libecs::Process,
                         GillespieProcessInterface )
{
    typedef libecs::MethodProxy< GillespieProcess, libecs::Real > RealMethodProxy;
    typedef const libecs::Real( GillespieProcess::* PDMethodPtr )( libecs::Variable* ) const; 
    
public:
    LIBECS_DM_OBJECT( GillespieProcess, Process );
    
    GillespieProcess();

    virtual ~GillespieProcess();

    // c means stochastic reaction constant
    SET_METHOD( libecs::Real, k );
    GET_METHOD( libecs::Real, k );
    SET_METHOD( libecs::Real, c );
    GET_METHOD( libecs::Real, c );
    
    virtual GET_METHOD( libecs::Real, Propensity );

    GET_METHOD( libecs::Real, Propensity_R );

    virtual const libecs::Real getPD( libecs::Variable* aVariable ) const;

    virtual const bool isContinuous() const;

    // The order of the reaction, i.e. 1 for a unimolecular reaction.

    GET_METHOD( libecs::Integer, Order );

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

#endif /* __GILLESPIEPROCESS_HPP */
