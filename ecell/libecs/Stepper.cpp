//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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
#include <algorithm>
#include <limits>

#include "Substance.hpp"
#include "Integrators.hpp"
#include "Model.hpp"
#include "Util.hpp"
#include "FullID.hpp"
#include "PropertySlot.hpp"

#include "Stepper.hpp"


namespace libecs
{


  ////////////////////////// Stepper

  Stepper::Stepper() 
    :
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 ),
    theMinInterval( 0.0 ),
    theMaxInterval( std::numeric_limits<Real>::max() ),
    theEntityListChanged( true )
  {
    calculateStepsPerSecond();
  }

  void Stepper::initialize()
  {
    FOR_ALL( SystemVector, theSystemVector, initialize );
  }


  void Stepper::registerSystem( SystemPtr aSystem )
  { 
    theSystemVector.push_back( aSystem );
  }

  void Stepper::removeSystem( SystemPtr aSystem )
  { 
    SystemVectorIterator i( find( theSystemVector.begin(), 
				  theSystemVector.end(),
				  aSystem ) );

    if( i == theSystemVector.end() )
      {
	THROW_EXCEPTION( NotFound,
			 getClassName() + String( ": " ) 
			 + getName() + ": " + aSystem->getFullID().getString() 
			 + " not found in this stepper." );
      }

    theSystemVector.erase( i );
  }

  void Stepper::registerPropertySlotWithProxy( PropertySlotPtr aSlotPtr )
  {
    if( std::find( thePropertySlotWithProxyVector.begin(),
		   thePropertySlotWithProxyVector.end(), aSlotPtr ) == 
	thePropertySlotWithProxyVector.end() )
      {
	thePropertySlotWithProxyVector.push_back( aSlotPtr );
      }
  }

  void Stepper::registerLoggedPropertySlot( PropertySlotPtr aPropertySlotPtr )
  {
    theLoggedPropertySlotVector.push_back( aPropertySlotPtr );
  }

  void Stepper::setStepInterval( RealCref aStepInterval )
  {
    theStepInterval = aStepInterval;
    calculateStepsPerSecond();
  }

  void Stepper::calculateStepsPerSecond() 
  {
    theStepsPerSecond = 1.0 / getStepInterval();
  }

  void Stepper::sync()
  {
    FOR_ALL( PropertySlotVector, thePropertySlotWithProxyVector, sync );
  }

  void Stepper::push()
  {
    FOR_ALL( PropertySlotVector, thePropertySlotWithProxyVector, push );

    // update loggers
    FOR_ALL( PropertySlotVector, theLoggedPropertySlotVector, updateLogger );
  }


  void Stepper::setParameterList( UVariableVectorCref aParameterList )
  {
    const UVariableVector::size_type aSize( aParameterList.size() );

    if( aSize >= 1 )
      {
	setStepInterval( aParameterList[0].asReal() );
    
	if( aSize >= 2 )
	  {
	    setMinInterval( aParameterList[1].asReal() );
	    
	    if( aSize >= 3 )
	      {
		setMaxInterval( aParameterList[2].asReal() );
	      }
	  }
      }
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
    FOR_ALL( SRMSubstanceCache, theSubstanceCache, clear );
  }



  void SRMStepper::differentiate()
  {
    //
    // Reactor::differentiate()
    //
    FOR_ALL( ReactorVector, theReactorCache, differentiate );

    //
    // Substance::turn()
    //
    FOR_ALL( SRMSubstanceCache, theSubstanceCache, turn );    
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
    FOR_ALL( SRMSubstanceCache, theSubstanceCache, integrate );

  }


#if 0
  void SRMStepper::???()
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

#endif /* 0 */

  void SRMStepper::initialize()
  {
    Stepper::initialize();

    if( isEntityListChanged() )
      {
	theSubstanceCache.update( theSystemVector );
	theReactorCache.update( theSystemVector );

	clearEntityListChanged();
      }

    distributeIntegrator( theIntegratorAllocator );
  }

  void SRMStepper::distributeIntegrator( IntegratorAllocator allocator )
  {
    for( SRMSubstanceCache::const_iterator s( theSubstanceCache.begin() );
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

  IntegratorPtr Euler1SRMStepper::newIntegrator( SRMSubstanceRef substance )
  {
    return new Euler1Integrator( substance );
  }

  ////////////////////////// RungeKutta4SRMStepper

  RungeKutta4SRMStepper::RungeKutta4SRMStepper()
  {
    theIntegratorAllocator = 
      IntegratorAllocator( &RungeKutta4SRMStepper::newIntegrator ); 
  }

  IntegratorPtr RungeKutta4SRMStepper::newIntegrator( SRMSubstanceRef substance )
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
