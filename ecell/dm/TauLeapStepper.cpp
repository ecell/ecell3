//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2015 Keio University
//       Copyright (C) 2008-2015 RIKEN
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

#include <utility>
#include <gsl/gsl_randist.h>

#include <libecs/DifferentialStepper.hpp>
#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>

#include "GillespieProcessInterface.hpp"

USE_LIBECS;

template< typename Tnew_, typename Tgiven_ >
struct Caster: public std::unary_function< Tgiven_, std::pair< Tnew_, Tgiven_ > >
{
    Caster(): dc_() {}

    std::pair< Tnew_, Tgiven_ > operator()( Tgiven_ const& ptr )
    {
        return std::make_pair( dc_( ptr ), ptr );
    }

private:
    DynamicCaster< Tnew_, Tgiven_ > dc_;
};

typedef Caster< GillespieProcessInterface*, Process* > GPCaster;

typedef std::vector< GPCaster::result_type > GillespieProcessVector;

LIBECS_DM_CLASS( TauLeapStepper, DifferentialStepper )
{    
public:
    LIBECS_DM_OBJECT( TauLeapStepper, Stepper )
    {
        INHERIT_PROPERTIES( DifferentialStepper );

        PROPERTYSLOT_SET_GET( Real, Epsilon );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Tau );
    }

    TauLeapStepper( void )
        : epsilon( 0.03 ),
          tau( libecs::INF )
    {
        ; // do nothing
    }

    virtual ~TauLeapStepper( void )
    {
        ; // do nothing
    }    

    virtual void initialize()
    {
        DifferentialStepper::initialize();

        theGillespieProcessVector.clear();

        try
        {
            std::transform( theProcessVector.begin(), theProcessVector.end(),
                            std::back_inserter( theGillespieProcessVector ),
                            GPCaster() );
        }
        catch( const libecs::TypeError& )
        {
            THROW_EXCEPTION_INSIDE( InitializationFailed,
                                    asString() + ": "
                                    "only GillespieProcesses can be associated "
                                    "with this Stepper" );
        }
    }

    virtual void updateInternalState( Real aStepInterval )
    {
        clearVariables();

        calculateTau();

        DifferentialStepper::setStepInterval( getTau() );

        FOR_ALL( GillespieProcessVector, theGillespieProcessVector )
        {
            (*i).second->setActivity( gsl_ran_poisson( getRng(),
                                      (*i).first->getPropensity() ) );
        }

        setVariableVelocity( theTaylorSeries[ 0 ] );
    }

    GET_METHOD( Real, Epsilon )
    {
        return epsilon;
    }

    SET_METHOD( Real, Epsilon )
    {
        epsilon = value;
    }

    GET_METHOD( Real, Tau )
    {
        return tau;
    }

protected:

    const Real getTotalPropensity()
    {
        Real totalPropensity( 0.0 );
        FOR_ALL( GillespieProcessVector, theGillespieProcessVector )
        {
            totalPropensity += (*i).first->getPropensity();
        }

        return totalPropensity;
    }

    void calculateTau()
    {
        tau = libecs::INF;

        const Real totalPropensity( getTotalPropensity() );
        
        const GillespieProcessVector::size_type 
            aSize( theGillespieProcessVector.size() );    
        for( GillespieProcessVector::size_type i( 0 ); i < aSize; ++i )
        {
            Real aMean( 0.0 );
            Real aVariance( 0.0 );
            
            for( GillespieProcessVector::size_type j( 0 ); j < aSize; ++j )
            {
                const Real aPropensity(
                    theGillespieProcessVector[ j ].first->getPropensity() );
                Process::VariableReferenceVector const& aVariableReferenceVector(
                    theGillespieProcessVector[ j ].second->getVariableReferenceVector() );
                
                // future works : theDependentProcessVector
                Real expectedChange( 0.0 );
                for( Process::VariableReferenceVector::const_iterator
                             k( aVariableReferenceVector.begin() ); 
                     k != aVariableReferenceVector.end(); ++k )
                {
                    expectedChange += theGillespieProcessVector[ i ].first->getPD( (*k).getVariable() ) * (*k).getCoefficient();
                }
                
                aMean += expectedChange * aPropensity;
                aVariance += expectedChange * expectedChange * aPropensity;
            }
            
            const Real aTolerance( epsilon * totalPropensity );
            const Real expectedTau( std::min( aTolerance / std::abs( aMean ), 
                                    aTolerance * aTolerance / aVariance ) );
            if ( expectedTau < tau )
            {
                tau = expectedTau;
            }
        }
    }

protected:
    
    Real epsilon;
    Real tau;
    GillespieProcessVector theGillespieProcessVector;

};

LIBECS_DM_INIT( TauLeapStepper, Stepper );
