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




#include <exception>
#include "Stepper.h"
#include "SystemMaker.h"
//FIXME: #include "ecell/TimeManager.h"
#include "util/Util.h"
#include "Integrators.h"
#include "RootSystem.h"



void StepperMaker::makeClassList()
{
  NewStepperModule(SlaveStepper);
  NewStepperModule(Eular1Stepper);
  NewStepperModule(RungeKutta4Stepper);
}

////////////////////////// Stepper

Stepper::Stepper() : theOwner(NULL)
{

}

void Stepper::distributeIntegrator( IntegratorAllocator* allocator )
{
  assert( theOwner );
  
  for( SubstanceListIterator s = theOwner->getFirstSubstanceIterator();
       s != theOwner->getLastSubstanceIterator() ; ++s)
    {
      (*allocator)(*(s->second));
    }
}

void Stepper::initialize()
{
  assert(theOwner);
}

////////////////////////// MasterStepper

MasterStepper::MasterStepper() 
  :
  thePace(1),
  theAllocator(NULL)
{
  theRootSystem->getStepperLeader().registerMasterStepper( this );
}

void MasterStepper::initialize()
{
  Stepper::initialize();
  registerSlaves( theOwner );

  distributeIntegrator( IntegratorAllocator( theAllocator ) );

  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    (*i)->initialize();
}

void MasterStepper::distributeIntegrator(IntegratorAllocator allocator)
{
  Stepper::distributeIntegrator( &allocator );
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    (*i)->distributeIntegrator( &allocator );
}


void MasterStepper::registerSlaves(System* system)
{
  for( SystemListIterator s = theOwner->getFirstSystemIterator() ;
       s != theOwner->getLastSystemIterator() ; ++s )
    {
      SystemPtr aSystemPtr = s->second;
      SlaveStepperPtr aSlaveStepperPtr;

      //FIXME: handle bad_cast
      if( ( aSlaveStepperPtr = 
	    dynamic_cast< SlaveStepperPtr >( aSystemPtr->getStepper() ) ) )
	{
	  theSlavesList.insert( theSlavesList.end(), aSlaveStepperPtr );
	  aSlaveStepperPtr->setMaster( this );
	  registerSlaves( aSystemPtr );
#ifdef DEBUG_STEPPER
	  cerr << "MasterStepper(on " << owner()->fqen() << "): registered " << it->fqen()  << endl;
#endif /* DEBUG_STEPPER */
	}
    }
}

Float MasterStepper::getDeltaT()
{
  return theRootSystem->getStepperLeader().getDeltaT();
}


////////////////////////// StepperLeader

int StepperLeader::DEFAULT_UPDATE_DEPTH(1);

StepperLeader::StepperLeader() 
  : 
  theUpdateDepth( DEFAULT_UPDATE_DEPTH ),
  theBaseClock( 1 )
{
  ; // do nothing
}

void StepperLeader::setBaseClock( int clock )
{
  theBaseClock = clock;
}

Float StepperLeader::getDeltaT()
{
  //FIXME: 
  // return theTimeManager->stepInterval();
}


void StepperLeader::registerMasterStepper( MasterStepperPtr newone )
{
  theStepperList.insert( pair< int, MasterStepperPtr >( newone->getPace(),
							newone ) );
  setBaseClock( lcm( newone->getPace(), getBaseClock() ) );

#ifdef DEBUG_STEPPER
  cerr << "registered new master stepper (pace: " << newone->pace() << ")." << endl;
  cerr << "base clock: " << getBaseClock() << endl;
#endif  
}

void StepperLeader::initialize()
{
  for( MasterStepperMap::iterator i = theStepperList.begin();
       i != theStepperList.end() ; i++)
    {
      (*i).second->initialize();
    }
}

void StepperLeader::step()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: step()" << endl;
#endif /* DEBUG_STEPPER */

  clear();
  react();
  transit();
  postern();
}

void StepperLeader::clear()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: clear()" << endl;
#endif /* DEBUG_STEPPER */

  for( MasterStepperMap::iterator i = theStepperList.begin();
       i != theStepperList.end() ; ++i )
    i->second->clear();
}

void StepperLeader::react()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: react()" << endl;
#endif /* DEBUG_STEPPER */

  for( MasterStepperMap::iterator i = theStepperList.begin();
       i != theStepperList.end(); i++ )
    {
      (*i).second->react();
    }
}

void StepperLeader::transit()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: transit()" << endl;
#endif /* DEBUG_STEPPER */

  for( MasterStepperMap::iterator i = theStepperList.begin();
       i != theStepperList.end(); ++i )
    {
      i->second->transit();
    }

}

void StepperLeader::postern()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: transit()" << endl;
#endif /* DEBUG_STEPPER */

  for( MasterStepperMap::iterator i = theStepperList.begin();
       i != theStepperList.end(); ++i )
    {
      i->second->postern();
    }
}

void StepperLeader::update()
{
  for( int i = theUpdateDepth ; i > 0 ; --i )
    {
      for (MasterStepperMap::iterator i = theStepperList.begin();
	   i != theStepperList.end(); ++i )
	{
	  i->second->postern();
	}
    }
}

////////////////////////// SlaveStepper

SlaveStepper::SlaveStepper()
{

}

void SlaveStepper::initialize()
{
  Stepper::initialize();
}

void SlaveStepper::clear()
{
  theOwner->clear();
}

void SlaveStepper::react()
{
  theOwner->react();
}

void SlaveStepper::turn()
{
  theOwner->turn();
}

void SlaveStepper::transit()
{
  theOwner->transit();
}

void SlaveStepper::postern()
{
  theOwner->postern();
}


////////////////////////// Eular1Stepper

Eular1Stepper::Eular1Stepper()
{
  theAllocator = IntegratorAllocator( &Eular1Stepper::newEular1 );
}

IntegratorPtr Eular1Stepper::newEular1( SubstanceRef substance )
{
  return new Eular1Integrator( substance );
}

void Eular1Stepper::initialize()
{
  MasterStepper::initialize();
}

void Eular1Stepper::clear()
{
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: clear()" << endl;
#endif /* DEBUG_STEPPER */
  theOwner->clear();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->clear();
    }
}

void Eular1Stepper::react()
{  
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: react()" << endl;
#endif /* DEBUG_STEPPER */

  theOwner->react();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
#ifdef DEBUG_STEPPER
      cerr << "react slaves: owner: "<< (*i)->owner()->entryname() << endl;
#endif /* DEBUG_STEPPER */
      (*i)->react();
    }
  theOwner->turn();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->turn();
    }
}

void Eular1Stepper::transit()
{
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: transit()" << endl;
#endif /* DEBUG_STEPPER */
  theOwner->transit();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->transit();
    }
}

void Eular1Stepper::postern()
{
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: transit()" << endl;
#endif /* DEBUG_STEPPER */
  theOwner->postern();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->postern();
    }
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

void RungeKutta4Stepper::initialize()
{
  MasterStepper::initialize();
}

void RungeKutta4Stepper::clear()
{
  theOwner->clear();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->clear();
    }
}

void RungeKutta4Stepper::react()
{
  // 1
  theOwner->react();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->react();
    }
  theOwner->turn();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->turn();
    }

  // 2
  theOwner->react();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->react();
    }
  theOwner->turn();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->turn();
    }

  // 3
  theOwner->react();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->react();
    }
  theOwner->turn();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->turn();
    }

  // 4
  theOwner->react();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->react();
    }
  theOwner->turn();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->turn();
    }
}

void RungeKutta4Stepper::transit()
{
  theOwner->transit();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->transit();
    }
}

void RungeKutta4Stepper::postern()
{
  theOwner->postern();
  for( SlaveStepperListIterator i = theSlavesList.begin() ; 
       i != theSlavesList.end() ; ++i )
    {
      (*i)->postern();
    }
}




/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
