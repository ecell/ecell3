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

#include "Integrators.hpp"
#include "RootSystem.hpp"
#include "Util.hpp"

#include "Stepper.hpp"


namespace libecs
{


  ////////////////////////// StepperLeader

  StepperLeader::StepperLeader() 
    : 
    theCurrentTime( 0.0 ),
    theRootSystem( NULL )
  {
    ; // do nothing
  }


  void StepperLeader::updateMasterStepperVector( SystemPtr aSystemPtr )
  {

    MasterStepperPtr aMasterStepper( dynamic_cast<MasterStepperPtr>
				     ( aSystemPtr->getStepper() ) );

    if( aMasterStepper != NULL )
      {
	theMasterStepperVector.push_back( aMasterStepper );
      }

    //FIXME: breadth-first search?
    for( SystemMapConstIterator i( aSystemPtr->getSystemMap().begin() ) ;
	 i != aSystemPtr->getSystemMap().end() ; ++i )
      {
	updateMasterStepperVector( i->second );
      }


  }

  void StepperLeader::updateScheduleQueue()
  {
    //FIXME: slow! :  no theScheduleQueue.clear() ?
    while( ! theScheduleQueue.empty() )
      {
	theScheduleQueue.pop();
      }




    for( MasterStepperVectorConstIterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end() ; i++)
      {
	theScheduleQueue.push( Event( theCurrentTime, (*i) ) );
      }
  }

  void StepperLeader::initialize()
  {
    assert( theRootSystem != NULL );

    theMasterStepperVector.clear();

    updateMasterStepperVector( theRootSystem );
    FOR_ALL( MasterStepperVector, theMasterStepperVector, 
	     initialize );

    updateScheduleQueue();

    theCurrentTime = ( theScheduleQueue.top() ).first;
  }


  void StepperLeader::step()
  {
    EventCref aTopEvent( theScheduleQueue.top() );

    MasterStepperPtr aMasterStepper( aTopEvent.second );

    // three-phase progression of the step
    // 1. sync:  synchronize with proxies of the PropertySlots
    aMasterStepper->sync();
    // 2. step:  do the computation, returning a length of the time progression
    const Real aStepSize( aMasterStepper->step() );
    // 3. push:  re-sync with the proxies, and push new values to Loggers
    aMasterStepper->push();


    // the time must be memorized before the Event is deleted by the pop
    const Real aTopTime( aTopEvent.first );

    //FIXME: change_top() is better than pop 'n' push.
    // If the ScheduleQueue holds pointers of Event, not instances,
    // it would be more efficient in the current implementation because
    // the instantiation below can be eliminated, but
    // if there is the change_top(), benefits would be lesser...
    theScheduleQueue.pop();
    theScheduleQueue.push( Event( aTopTime + aStepSize, aMasterStepper ) );

    // update theCurrentTime, which is scheduled time of the Event on the top
    theCurrentTime = ( theScheduleQueue.top() ).first;
  }



  ////////////////////////// Stepper

  Stepper::Stepper() 
    : 
    theOwner( NULLPTR )
  {

  }

  void Stepper::initialize()
  {
    // FIXME: use exception?
    assert( theOwner );
  }

  ////////////////////////// MasterStepper

  MasterStepper::MasterStepper() 
    :
    theStepInterval( 0.001 )
  {
    setMasterStepper( this );
    calculateStepsPerSecond();
  }

  void MasterStepper::initialize()
  {
    // FIXME: is this multiple-time-initialization-proof? 
    Stepper::initialize();

    updateSlaveStepperVector();

    for( SlaveStepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->setMasterStepper( this );
	(*i)->initialize();
      }
  }


  void MasterStepper::updateSlaveStepperVector()
  {
    theSlaveStepperVector.clear();

    searchSlaves( theOwner );
  }

  void MasterStepper::searchSlaves( SystemPtr aStartSystemPtr )
  {
    for( SystemMapConstIterator s( aStartSystemPtr->getSystemMap().begin() );
	 s != aStartSystemPtr->getSystemMap().end() ; ++s )
      {
	SystemPtr aSlaveSystemPtr( s->second );

	//FIXME: handle bad_cast
	SlaveStepperPtr aSlaveStepperPtr( dynamic_cast< SlaveStepperPtr >
					  ( aSlaveSystemPtr->getStepper() ) );

	if( aSlaveStepperPtr != NULLPTR )
	  {
	    theSlaveStepperVector.push_back( aSlaveStepperPtr );
	    aSlaveStepperPtr->setMasterStepper( this );
	    searchSlaves( aSlaveSystemPtr );
	  }
      }
  }


  void MasterStepper::registerPropertySlot( PropertySlotPtr propertyslot )
  {
    thePropertySlotVector.push_back( propertyslot );
  }

  void MasterStepper::setStepInterval( RealCref aStepInterval )
  {
    theStepInterval = aStepInterval;
    calculateStepsPerSecond();
  }

  void MasterStepper::calculateStepsPerSecond() 
  {
    theStepsPerSecond = 1 / getStepInterval();
  }

  void MasterStepper::sync()
  {
    FOR_ALL( PropertySlotVector, thePropertySlotVector, sync );
  }

  void MasterStepper::push()
  {
    FOR_ALL( PropertySlotVector, thePropertySlotVector, push );
  }


  ////////////////////////// MasterStepperWithEntityCache

  void MasterStepperWithEntityCache::initialize()
  {
    MasterStepper::initialize();
    updateCache();
  }

  
  void MasterStepperWithEntityCache::updateCache()
  {
    // clear the caches
    theSystemCache.clear();
    theSubstanceCache.clear();
    theReactorCache.clear();

    theSystemCache.reserve( getSlaveStepperVector().size() + 1 );
    theSystemCache.push_back( getOwner() );

    for( SlaveStepperVectorConstIterator i( getSlaveStepperVector().begin() );
	 i != getSlaveStepperVector().end() ; ++i )
      {
	SystemPtr aSystem( (*i)->getOwner() );
	theSystemCache.push_back( aSystem );

	for( SubstanceMapConstIterator i( aSystem->getSubstanceMap().begin() );
	     i != aSystem->getSubstanceMap().end(); ++i )
	  {
	    theSubstanceCache.push_back( (*i).second );
	  }

	for( ReactorMapConstIterator i( aSystem->getReactorMap().begin() );
	     i != aSystem->getReactorMap().end(); ++i )
	  {
	    theReactorCache.push_back( (*i).second );
	  }

      }

  }


  //FIXME: incomplete
  void MasterStepperWithEntityCache::updateCacheWithSort()
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
    FOR_ALL( ReactorVector, theReactorCache, integrate );


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
    FOR_ALL( ReactorVector, theReactorCache, integrate );

  }

  void SRMStepper::initialize()
  {
    MasterStepperWithEntityCache::initialize();
    distributeIntegrator( IntegratorAllocator( theIntegratorAllocator ) );
  }

  void StepperLeader::push()
  {
    FOR_ALL( MasterStepperVector, theMasterStepperVector, push );
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
