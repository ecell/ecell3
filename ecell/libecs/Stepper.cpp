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

    // the time must be memorized before the Event is deleted by push()
    const Real aTopTime( aTopEvent.first );

    aMasterStepper->sync();

    const Real aStepSize( aMasterStepper->step() );

    aMasterStepper->push();

    //FIXME: change_top() is better than pop 'n' push
    theScheduleQueue.pop();
    theScheduleQueue.push( Event( aTopTime + aStepSize,
				  aMasterStepper ) );

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

    updateSlaveStepperVector( theOwner );

    for( SlaveStepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->setMasterStepper( this );
	(*i)->initialize();
      }
  }


  void MasterStepper::updateSlaveStepperVector( SystemPtr aStartSystemPtr )
  {
    theSlaveStepperVector.clear();

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
	    updateSlaveStepperVector( aSlaveSystemPtr );
	  }
      }
  }


  void MasterStepper::registerPropertySlot( PropertySlotPtr propertyslot )
  {
    thePropertySlotVector.push_back( propertyslot );
  }

  void MasterStepper::setStepInterval( RealCref stepsize )
  {
    theStepInterval = stepsize;
    calculateStepsPerSecond();
  }

  void MasterStepper::calculateStepsPerSecond() 
  {
    theStepsPerSecond = 1 / getStepInterval();
  }

  void MasterStepper::sync()
  {

  }

  void MasterStepper::push()
  {

  }

  ////////////////////////// MasterStepperWithEntityCache

  void MasterStepperWithEntityCache::initialize()
  {
    MasterStepper::initialize();
    updateCacheWithCheck();
  }

  
  void MasterStepperWithEntityCache::updateCache()
  {
    SystemPtr aMasterSystem( getOwner() );
    theSystemCache.resize( getSlaveStepperVector().size() + 1 );
    SystemVectorIterator aSystemCacheIterator( theSystemCache.begin() );
    (*aSystemCacheIterator) = aMasterSystem;

    SubstanceVector::size_type 
      aSubstanceCacheSize( aMasterSystem->getSubstanceMap().size() );
    ReactorVector::size_type   
      aReactorCacheSize( aMasterSystem->getReactorMap().size() );

    SlaveStepperVectorCref 
      aSlaveStepperVector( getSlaveStepperVector() );
    for( SlaveStepperVectorConstIterator i( aSlaveStepperVector.begin() ); 
	 i != aSlaveStepperVector.end() ; ++i )
      {
	SystemPtr aSystem( (*i)->getOwner() );
	aSubstanceCacheSize += aSystem->getSubstanceMap().size();
	aReactorCacheSize += aSystem->getReactorMap().size();
	*(++aSystemCacheIterator) = aSystem;
      }

    theSubstanceCache.resize( aSubstanceCacheSize );
    theReactorCache.resize( aReactorCacheSize );

    SubstanceVectorIterator 
      aSubstanceVectorIterator( theSubstanceCache.begin() );
    ReactorVectorIterator aReactorVectorIterator( theReactorCache.begin() );

    for( SystemVectorConstIterator i( theSystemCache.begin() );
	 i != theSystemCache.end() ; ++i )
      {
	aSubstanceVectorIterator =
	  std::transform( aMasterSystem->getSubstanceMap().begin(), 
			  aMasterSystem->getSubstanceMap().end(), 
			  aSubstanceVectorIterator,
			  std::select2nd<SubstanceMap::value_type>() );    

	aReactorVectorIterator =
	  std::transform( aMasterSystem->getReactorMap().begin(), 
			  aMasterSystem->getReactorMap().end(), 
			  aReactorVectorIterator,
			  std::select2nd<ReactorMap::value_type>() );    
      }

  }


  //FIXME: incomplete
  void MasterStepperWithEntityCache::updateCacheWithSort()
  {
    SystemPtr aMasterSystem( getOwner() );
    theSystemCache.resize( getSlaveStepperVector().size() + 1 );
    SystemVectorIterator aSystemCacheIterator( theSystemCache.begin() );
    (*aSystemCacheIterator) = aMasterSystem;

    SubstanceVector::size_type 
      aSubstanceCacheSize( aMasterSystem->getSubstanceMap().size() );
    ReactorVector::size_type   
      aReactorCacheSize( aMasterSystem->getReactorMap().size() );

    SlaveStepperVectorCref 
      aSlaveStepperVector( getSlaveStepperVector() );
    for( SlaveStepperVectorConstIterator i( aSlaveStepperVector.begin() ); 
	 i != aSlaveStepperVector.end() ; ++i )
      {
	SystemPtr aSystem( (*i)->getOwner() );
	aSubstanceCacheSize += aSystem->getSubstanceMap().size();
	aReactorCacheSize += aSystem->getReactorMap().size();
	*(++aSystemCacheIterator) = aSystem;
      }

    theSubstanceCache.resize( aSubstanceCacheSize );
    theReactorCache.resize( aReactorCacheSize );

    SubstanceVectorIterator aSubstanceVectorIterator
      ( theSubstanceCache.begin() );

    ReactorVectorIterator aReactorVectorIterator
      ( theReactorCache.begin() );

    for( SystemVectorConstIterator i( theSystemCache.begin() );
	 i != theSystemCache.end() ; ++i )
      {
	  
	aSubstanceVectorIterator =
	  std::transform( aMasterSystem->getSubstanceMap().begin(), 
			  aMasterSystem->getSubstanceMap().end(), 
			  aSubstanceVectorIterator,
			  std::select2nd<SubstanceMap::value_type>() );    

	aReactorVectorIterator =
	  std::transform( aMasterSystem->getReactorMap().begin(), 
			  aMasterSystem->getReactorMap().end(), 
			  aReactorVectorIterator,
			  std::select2nd<ReactorMap::value_type>() );    
      }

    //      theMaster->push();
    ;
  }



  ////////////////////////// SRMStepper

  SRMStepper::SRMStepper()
    :
    theIntegratorAllocator( NULLPTR )
  {
    for( PropertySlotVectorIterator i( thePropertySlotVector.begin() );
	 i != thePropertySlotVector.end(); ++i )
      {
	(*i)->sync();
      }
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


    //FIXME: should be removed
    push();
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
    for( MasterStepperVector::iterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end(); ++i )
      {
	(*i)->push();
      }
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
      IntegratorAllocator( &Euler1SRMStepper::newEuler1 );
  }

  IntegratorPtr Euler1SRMStepper::newEuler1( SubstanceRef substance )
  {
    return new Euler1Integrator( substance );
  }

  ////////////////////////// RungeKutta4SRMStepper

  RungeKutta4SRMStepper::RungeKutta4SRMStepper()
  {
    theIntegratorAllocator = 
      IntegratorAllocator( &RungeKutta4SRMStepper::newRungeKutta4 ); 
  }

  IntegratorPtr RungeKutta4SRMStepper::newRungeKutta4( SubstanceRef substance )
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
