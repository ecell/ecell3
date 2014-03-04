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
// Warning:
//
// the number of Substrate of DecayReactor must be one.
//

#include <libecs/libecs.hpp>
#include <libecs/Util.hpp>

#include <libecs/ContinuousProcess.hpp>


USE_LIBECS;

LIBECS_DM_CLASS( DecayFluxProcess, ContinuousProcess )
{
public:
    LIBECS_DM_OBJECT( DecayFluxProcess, Process )
    {
        CLASS_DESCRIPTION(
            "DecayFluxProcess is a FluxProcess which calculates "
            "mass-action decay process with a half-time T.\n\n"
            "This is a mass-action reaction with a single reactant"
            "VariableReference S0, which must be specified "
            "in the model:\n"
            "S0 --> (..0, 1, or more products..)\n"
            "The half-time T is converted to the rate constant k as:\n"
            "k = log( 2 ) / T\n\n"
            "Flux rate of this Process is calculated by the following "
            "equation:\n"
            "flux rate = k * pow( S0.Value, S0.Coefficient )\n"
            "When the coefficient of S0 is 1, then it is simplified as:"
            "flux rate = k * S0.Value\n\n"
            "Although only S0 is used for calculating the flux rate,"
            "velocities are set to all VariableReferences with non-zero"
            "coefficients, as defined in the FluxProcess base class.\n"
            "Zero or negative half time is not allowed.\n" );

        INHERIT_PROPERTIES( ContinuousProcess );

        PROPERTYSLOT_SET_GET( Real, T );
    }    

    DecayFluxProcess()
        : T( 1.0 ), k( 0.0 )
    {
        ; // do nothing
    }
    
    SIMPLE_SET_GET_METHOD( Real, T );
        
    virtual void initialize()
    {
        Process::initialize();

        if( T <= 0.0 )
        {
            THROW_EXCEPTION_INSIDE( InitializationFailed, 
                                    asString() + ": zero or negative half time" );
        }

        k = log( 2.0 ) / T;
        S0 = getVariableReference( "S0" );
    }

    virtual void fire()
    {
        Real velocity( k );

        velocity *= pow( S0.getVariable()->getValue(), - S0.getCoefficient() );
        setFlux( velocity );
    }

protected:
    
    Real T;
    Real k;
    VariableReference S0;
    
};

LIBECS_DM_INIT( DecayFluxProcess, Process );
