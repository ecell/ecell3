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

#include "Integrators.hpp"
#include "RootSystem.hpp"
#include "Util.hpp"

#include "Stepper.hpp"

namespace libecs
{

  ////////////////////////// Stepper

  Stepper::Stepper() 
    : 
    theOwner( NULLPTR )
  {

  }

  void Stepper::distributeIntegrator( IntegratorAllocator* allocator )
  {
    assert( theOwner );
  
    for( SubstanceMapIterator s = theOwner->getFirstSubstanceIterator();
	 s != theOwner->getLastSubstanceIterator() ; ++s)
      {
	(*allocator)(*(s->second));
      }
  }

  void Stepper::initialize()
  {
    // FIXME: use exception?
    assert( theOwner );
  }

  ////////////////////////// MasterStepper

  MasterStepper::MasterStepper() 
    :
    theStepInterval( 0.001 ),
    theStepsPerSecond( 1000 ),
    theAllocator( NULLPTR )
  {
    ; // do nothing
  }

  void MasterStepper::initialize()
  {
    // FIXME: is this multiple-time-initialization-proof? 
    Stepper::initialize();

    registerSlaves( theOwner );

    distributeIntegrator( IntegratorAllocator( theAllocator ) );

    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->initialize();
      }
  }

  void MasterStepper::distributeIntegrator( IntegratorAllocator allocator )
  {
    Stepper::distributeIntegrator( &allocator );
    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->distributeIntegrator( &allocator );
      }
  }


  void MasterStepper::registerSlaves( SystemPtr system )
  {
    for( SystemMapIterator s( theOwner->getFirstSystemIterator() );
	 s != theOwner->getLastSystemIterator() ; ++s )
      {
	SystemPtr aSystemPtr( s->second );

	//FIXME: handle bad_cast
	SlaveStepperPtr aSlaveStepperPtr( dynamic_cast< SlaveStepperPtr >
					  ( aSystemPtr->getStepper() ) );

	if( aSlaveStepperPtr != NULLPTR )
	  {
	    theSlaveStepperVector.push_back( aSlaveStepperPtr );
	    aSlaveStepperPtr->setMaster( this );
	    registerSlaves( aSystemPtr );
	  }
      }
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

  RealCref MasterStepper::getStepsPerSecond() const
  {
    return getOwner()->getRootSystem()->getStepperLeader().getStepsPerSecond();
  }

  RealCref MasterStepper::getStepInterval() const
  {
    return getOwner()->getRootSystem()->getStepperLeader().getStepInterval();
  }

  void MasterStepper::clear()
  {
    theOwner->clear();
    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->clear();
      }
  }

  void MasterStepper::react()
  {  
    theOwner->react();
    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->react();
      }
    theOwner->turn();
    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->turn();
      }
  }

  void MasterStepper::transit()
  {
    theOwner->transit();
    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->transit();
      }
  }

  void MasterStepper::postern()
  {
    theOwner->postern();
    for( StepperVectorIterator i( theSlaveStepperVector.begin() ); 
	 i != theSlaveStepperVector.end() ; ++i )
      {
	(*i)->postern();
      }
  }

  ////////////////////////// StepperLeader

  int StepperLeader::DEFAULT_UPDATE_DEPTH(1);

  StepperLeader::StepperLeader() 
    : 
    theUpdateDepth( DEFAULT_UPDATE_DEPTH ),
    theCurrentTime( 0.0 ),
    theStepInterval( 0.001 )
  {
    calculateStepsPerSecond();
  }

  void StepperLeader::registerMasterStepper( MasterStepperPtr newone )
  {
    theMasterStepperVector.push_back( newone );
  }

  void StepperLeader::initialize()
  {
    for( StepperVector::iterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end() ; i++)
      {
	(*i)->initialize();
      }
  }

  void StepperLeader::step()
  {
    clear();
    react();
    transit();
    postern();

    theCurrentTime += theStepInterval;
  }

  void StepperLeader::clear()
  {
    for( StepperVector::iterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end() ; ++i )
      {
	(*i)->clear();
      }
  }

  void StepperLeader::react()
  {
    for( StepperVector::iterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end(); i++ )
      {
	(*i)->react();
      }
  }

  void StepperLeader::transit()
  {
    for( StepperVector::iterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end(); ++i )
      {
	(*i)->transit();
      }
  }

  void StepperLeader::postern()
  {
    for( StepperVector::iterator i( theMasterStepperVector.begin() );
	 i != theMasterStepperVector.end(); ++i )
      {
	(*i)->postern();
      }
  }

  void StepperLeader::update()
  {
    for( int i( theUpdateDepth ) ; i > 0 ; --i )
      {
	for ( StepperVector::iterator i( theMasterStepperVector.begin() );
	     i != theMasterStepperVector.end(); ++i )
	  {
	    (*i)->postern();
	  }
      }
  }

  ////////////////////////// Euler1Stepper

  Euler1Stepper::Euler1Stepper()
  {
    theAllocator = IntegratorAllocator( &Euler1Stepper::newEuler1 );
  }

  IntegratorPtr Euler1Stepper::newEuler1( SubstanceRef substance )
  {
    return new Euler1Integrator( substance );
  }

  ////////////////////////// RungeKutta4Stepper

  RungeKutta4Stepper::RungeKutta4Stepper()
  {
    theAllocator = IntegratorAllocator( &RungeKutta4Stepper::newRungeKutta4 ); 
  }

  IntegratorPtr RungeKutta4Stepper::newRungeKutta4( SubstanceRef substance )
  {
    return new RungeKutta4Integrator( substance );
  }

  void RungeKutta4Stepper::react()
  {
    MasterStepper::react();
    MasterStepper::react();
    MasterStepper::react();
    MasterStepper::react();
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
