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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif /* HAVE_CONFIG_H */

#include <time.h>

#include "SimulationContext.hpp"
#include "Model.hpp"

namespace libecs {

SimulationContext::SimulationContext()
    : lastEvent_( 0, 0 )
{
    rng_ = gsl_rng_alloc( gsl_rng_default );
}

SimulationContext::~SimulationContext()
{
    gsl_rng_free( rng_ );
}

void SimulationContext::setModel( Model* model )
{
    model_ = model;
    model_->setSimulationContext( this );
}

void SimulationContext::setRngSeed( const String& value )
{
    unsigned int seed( 0 );
    if ( value == "TIME" )
    {
        // Using just time() still gives the same seeds to Steppers
        // in multi-stepper model.  Stepper index is added to prevent this.
        seed = static_cast<UnsignedInteger>(
                time( NULLPTR ) + scheduler_.getSize() );
    }
    else if ( value == "DEFAULT" )
    {
        seed = gsl_rng_default_seed;
    }
    else
    {
        seed = stringCast<unsigned int>( value );
    }

    gsl_rng_set( rng_, seed );
}

void SimulationContext::startup()
{
    // initialize systemStepper_
    systemStepper_.setModel( model_ );
    systemStepper_.setID( "___SYSTEM" );
    world_.setStepper( &systemStepper_ );
}

void SimulationContext::ensureExistenceOfSizeVariable()
{
    FullID fullID( FullID::parse( "Variable:/:SIZE" ) );

    if ( !model_->getEntity( fullID, false ) )
    {
        Variable* var = model_->createEntity<Variable>( "Variable" );
        model_->addEntity( fullID, var );
        var->setValue( 1.0 );
    }
}

/**
   Initialize steppers
   initialization of Stepper needs four stages:
   (1) update current times of all the steppers, and integrate Variables.
   (2) call user-initialization methods of Processes.
   (3) call user-defined initialize() methods.
   (4) post-initialize() procedures:
       - construct stepper dependency graph and
       - fill theIntegratedVariableVector.
   @internal
 */
void SimulationContext::initializeSteppers()
{
    systemStepper_.initialize();
    Model::SteppersCRange steppers( model_->getSteppers() );
    for ( Model::SteppersCRange::iterator i( steppers.begin() );
            i != steppers.end(); ++i ) {
        (*i)->initialize();
    }
}

void SimulationContext::postInitializeSteppers()
{
    systemStepper_.postInitialize();
    scheduler_.addEvent(
        StepperEvent(
            lastEvent_.getTime() + systemStepper_.getStepInterval(),
            &systemStepper_ ) );

    Model::SteppersCRange steppers( model_->getSteppers() );
    for ( Model::SteppersCRange::iterator i( steppers.begin() );
            i != steppers.end(); ++i ) {
        (*i)->postInitialize();
        for ( Model::SteppersCRange::iterator i( steppers.begin() );
                i != steppers.end(); ++i ) {
            scheduler_.addEvent(
                    StepperEvent(
                        lastEvent_.getTime() + (*i)->getStepInterval(),
                        (*i) ) );
        }
    }

    scheduler_.updateEventDependency();
}

void SimulationContext::initialize()
{
    world_.initialize();
    ensureExistenceOfSizeVariable();
    initializeSteppers();
    world_.postInitialize();
    postInitializeSteppers();
}

} // namespace libecs
