//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include <functional>

#include "Substance.hpp"
#include "Integrators.hpp"
#include "RootSystem.hpp"
#include "Util.hpp"
#include "FullID.hpp"

#include "Stepper.hpp"


namespace libecs
{


  ////////////////////////// Stepper

  Stepper::Stepper() 
    :
    theStepInterval( 0.001 )
  {
    calculateStepsPerSecond();
  }

  void Stepper::initialize()
  {
    for( SystemVectorConstIterator i( theSystemVector.begin() ); 
	 i != theSystemVector.end() ; ++i )
      {
	//FIXME: workaround, should be eliminated
	if( typeid( **i ) != typeid( RootSystem ) )
	  {
	    (*i)->initialize();
	  }
      }
  }


  void Stepper::connectSystem( SystemPtr aSystem )
  { 
    theSystemVector.push_back( aSystem );

  }

  void Stepper::disconnectSystem( SystemPtr aSystem )
  { 
    SystemVectorIterator i( find( theSystemVector.begin(), 
				  theSystemVector.end(),
				  aSystem ) );

    if( i == theSystemVector.end() )
      {
	throw NotFound( __PRETTY_FUNCTION__, getClassName() + String( ": " ) 
			+ getName() + ": " + aSystem->getFullID().getString() +
			" not found in this stepper." );
      }

    theSystemVector.erase( i );
  }

  void Stepper::registerPropertySlot( PropertySlotPtr propertyslot )
  {
    thePropertySlotVector.push_back( propertyslot );
  }

  void Stepper::setStepInterval( RealCref aStepInterval )
  {
    theStepInterval = aStepInterval;
    calculateStepsPerSecond();
  }

  void Stepper::calculateStepsPerSecond() 
  {
    theStepsPerSecond = 1 / getStepInterval();
  }

  void Stepper::sync()
  {
    FOR_ALL( PropertySlotVector, thePropertySlotVector, sync );
  }

  void Stepper::push()
  {
    FOR_ALL( PropertySlotVector, thePropertySlotVector, push );
  }


  ////////////////////////// StepperWithEntityCache

  void StepperWithEntityCache::initialize()
  {
    Stepper::initialize();
    updateCache();
  }

  
  void StepperWithEntityCache::updateCache()
  {
    theSubstanceCache.clear();
    theReactorCache.clear();

    for( SystemVectorConstIterator i( getSystemVector().begin() );
	 i != getSystemVector().end() ; ++i )
      {
	const SystemCptr aSystem( *i );

	for( SubstanceMapConstIterator j( aSystem->getSubstanceMap().begin() );
	     j != aSystem->getSubstanceMap().end(); ++j )
	  {
	    theSubstanceCache.push_back( (*j).second );
	  }

	for( ReactorMapConstIterator j( aSystem->getReactorMap().begin() );
	     j != aSystem->getReactorMap().end(); ++j )
	  {
	    theReactorCache.push_back( (*j).second );
	  }

      }

  }


  //FIXME: incomplete
  void StepperWithEntityCache::updateCacheWithSort()
  {
    updateCache();
  }


  ////////////////////////// SRMStepper

  SRMStepper::SRMStepper()
    :
    theIntegratorAllocator( NULLPTR )
  {

  }

  void SRMStepper::clear()
  {
    //
    // Substance::clear()
    //
    FOR_ALL( SubstanceVector, theSubstanceCache, clear );
  }



  void SRMStepper::differentiate()
  {
    //
    // Reactor::differentiate()
    //
    FOR_ALL( ReactorVector, theReactorCache, differentiate );
  }

  void SRMStepper::turn()
  {
    //
    // Substance::turn()
    //
    FOR_ALL( SubstanceVector, theSubstanceCache, turn );
  }

  void SRMStepper::integrate()
  {
    //
    // Reactor::integrate()
    //
    //    FOR_ALL( ReactorVector, theReactorCache, integrate );


    //
    // Substance::integrate()
    //
    FOR_ALL( SubstanceVector, theSubstanceCache, integrate );

  }


  void SRMStepper::compute()
  {
    //
    // Reactor::compute()
    //
    FOR_ALL( ReactorVector, theReactorCache, compute );

    //
    // Reactor::integrate()
    //
    // update activity of reactors by buffered values 
    //    FOR_ALL( ReactorVector, theReactorCache, integrate );

  }

  void SRMStepper::initialize()
  {
    StepperWithEntityCache::initialize();

    //FIXME: memory leak!!
    distributeIntegrator( IntegratorAllocator( theIntegratorAllocator ) );
  }

  void SRMStepper::distributeIntegrator( IntegratorAllocator allocator )
  {
    for( SubstanceVectorConstIterator s( theSubstanceCache.begin() );
	 s != theSubstanceCache.end() ; ++s )
      {
	(*allocator)(**s);
      }
  }



  ////////////////////////// Euler1SRMStepper

  Euler1SRMStepper::Euler1SRMStepper()
  {
    theIntegratorAllocator =
      IntegratorAllocator( &Euler1SRMStepper::newIntegrator );
  }

  IntegratorPtr Euler1SRMStepper::newIntegrator( SubstanceRef substance )
  {
    return new Euler1Integrator( substance );
  }

  ////////////////////////// RungeKutta4SRMStepper

  RungeKutta4SRMStepper::RungeKutta4SRMStepper()
  {
    theIntegratorAllocator = 
      IntegratorAllocator( &RungeKutta4SRMStepper::newIntegrator ); 
  }

  IntegratorPtr RungeKutta4SRMStepper::newIntegrator( SubstanceRef substance )
  {
    return new RungeKutta4Integrator( substance );
  }

  void RungeKutta4SRMStepper::differentiate()
  {
    SRMStepper::differentiate();
    SRMStepper::differentiate();
    SRMStepper::differentiate();
    SRMStepper::differentiate();
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
