
char const Stepper_C_rcsid[] = "$Id$";
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER




#include <exception>
#include <typeinfo>
#include "ecscore/Stepper.h"
#include "ecscore/SystemMaker.h"
#include "ecell/TimeManager.h"
#include "util/Util.h"
#include "ecscore/Integrators.h"
#include "ecscore/RootSystem.h"



void StepperMaker::makeClassList()
{
  NewStepperModule(SlaveStepper);
  NewStepperModule(Eular1Stepper);
  NewStepperModule(RungeKutta4Stepper);
}

////////////////////////// Stepper

Stepper::Stepper()
{
  _owner = NULL;

}

void Stepper::distributeIntegrator(IntegratorAllocator* allocator)
{
  assert(_owner);
  
  SSystem* ssystem = dynamic_cast<SSystem*>(_owner);
  if(ssystem)
    for(SubstanceListIterator s = ssystem->firstSubstance();
	s != ssystem->lastSubstance() ; ++s)
      {
	(*allocator)(*(s->second));
      }
}

void Stepper::initialize()
{
  assert(_owner);
}

////////////////////////// MasterStepper

MasterStepper::MasterStepper()
{
  _pace = 1;
  _allocator = NULL;
  theRootSystem->stepperLeader().registerMasterStepper(this);  
}

void MasterStepper::initialize()
{
  Stepper::initialize();
  registerSlaves(_owner);

  distributeIntegrator((IntegratorAllocator)_allocator);

  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->initialize();
}

void MasterStepper::distributeIntegrator(IntegratorAllocator allocator)
{
  Stepper::distributeIntegrator(&allocator);
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->distributeIntegrator(&allocator);
}


void MasterStepper::registerSlaves(System* system)
{
  MetaSystem* metasystem = dynamic_cast<MetaSystem*>(system);
  if(metasystem)
    {
      for(SystemListIterator s = metasystem->firstSystem() ;
	  s != metasystem->lastSystem() ; s++)
	{
	  System* it = s->second;
	  SlaveStepper* slave;
	  if((slave = dynamic_cast<SlaveStepper*>(it->stepper())))
	    {
	      _slavesList.insert(_slavesList.end(),slave);
	      slave->masterIs(this);
	      registerSlaves(it);
#ifdef DEBUG_STEPPER
  cerr << "MasterStepper(on " << owner()->fqen() << "): registered " << it->fqen()  << endl;
#endif /* DEBUG_STEPPER */
	    }
	}
    }
}

Float MasterStepper::deltaT()
{
  return theRootSystem->stepperLeader().deltaT();

}


////////////////////////// StepperLeader

int StepperLeader::_DEFAULT_UPDATE_DEPTH(1);

StepperLeader::StepperLeader() : 
_updateDepth(_DEFAULT_UPDATE_DEPTH),_baseClock(1)
{
}

void StepperLeader::setBaseClock(int clock)
{
  _baseClock = clock;
}

Float StepperLeader::deltaT()
{
  return theTimeManager->stepInterval();
}


void StepperLeader::registerMasterStepper(MasterStepper* newone)
{
  _stepperList.insert(pair<int,MasterStepper*>(newone->pace(),newone));
  setBaseClock(lcm(newone->pace(),baseClock()));

#ifdef DEBUG_STEPPER
  cerr << "registered new master stepper (pace: " << newone->pace() << ")." << endl;
  cerr << "base clock: " << baseClock() << endl;
#endif  
}

void StepperLeader::initialize()
{
  for(MasterStepperMap::iterator it = _stepperList.begin();
      it != _stepperList.end() ; it++)
    {
      ((*it)).second->initialize();
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

  for (MasterStepperMap::iterator it = _stepperList.begin();
       it != _stepperList.end();++it)
    it->second->clear();
}

void StepperLeader::react()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: react()" << endl;
#endif /* DEBUG_STEPPER */

  for(MasterStepperMap::iterator it = _stepperList.begin();
      it != _stepperList.end(); it++)
    (*it).second->react();
}

void StepperLeader::transit()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: transit()" << endl;
#endif /* DEBUG_STEPPER */

  for (MasterStepperMap::iterator it = _stepperList.begin();
       it != _stepperList.end();++it)
    it->second->transit();

}

void StepperLeader::postern()
{
#ifdef DEBUG_STEPPER
  cerr << "StepperLeader: transit()" << endl;
#endif /* DEBUG_STEPPER */

  for (MasterStepperMap::iterator it = _stepperList.begin();
       it != _stepperList.end();++it)
    it->second->postern();

}

void StepperLeader::update()
{
  for(int i = _updateDepth ; i > 0 ; i--)
    {
      for (MasterStepperMap::iterator it = _stepperList.begin();
	   it != _stepperList.end();++it)
	it->second->postern();
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
  _owner->clear();
}

void SlaveStepper::react()
{
  _owner->react();
}

void SlaveStepper::turn()
{
  _owner->turn();
}

void SlaveStepper::transit()
{
  _owner->transit();
}

void SlaveStepper::postern()
{
  _owner->postern();
}


////////////////////////// Eular1Stepper

Eular1Stepper::Eular1Stepper()
{
  _allocator = (IntegratorAllocator)&Eular1Stepper::newEular1;
}

Integrator* Eular1Stepper::newEular1(Substance& substance)
{
  return new Eular1Integrator(substance);
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
  _owner->clear();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->clear();
}

void Eular1Stepper::react()
{  
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: react()" << endl;
#endif /* DEBUG_STEPPER */

  _owner->react();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    {
#ifdef DEBUG_STEPPER
      cerr << "react slaves: owner: "<< (*i)->owner()->entryname() << endl;
#endif /* DEBUG_STEPPER */
      (*i)->react();
    }
  _owner->turn();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->turn();
}

void Eular1Stepper::transit()
{
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: transit()" << endl;
#endif /* DEBUG_STEPPER */
  _owner->transit();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    {
      (*i)->transit();
    }
}

void Eular1Stepper::postern()
{
#ifdef DEBUG_STEPPER
  cerr << "Eular1Stepper: transit()" << endl;
#endif /* DEBUG_STEPPER */
  _owner->postern();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    {
      (*i)->postern();
    }
}


////////////////////////// RungeKutta4Stepper

RungeKutta4Stepper::RungeKutta4Stepper()
{
  _allocator = (IntegratorAllocator)&RungeKutta4Stepper::newRungeKutta4;
}

Integrator* RungeKutta4Stepper::newRungeKutta4(Substance& substance)
{
  return new RungeKutta4Integrator(substance);
}

void RungeKutta4Stepper::initialize()
{
  MasterStepper::initialize();
}

void RungeKutta4Stepper::clear()
{
  _owner->clear();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->clear();
}

void RungeKutta4Stepper::react()
{
  // 1
  _owner->react();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->react();
  _owner->turn();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->turn();

  // 2
  _owner->react();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->react();
  _owner->turn();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->turn();

  // 3
  _owner->react();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->react();
  _owner->turn();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->turn();

  // 4
  _owner->react();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->react();
  _owner->turn();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->turn();
}

void RungeKutta4Stepper::transit()
{
  _owner->transit();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    (*i)->transit();
}

void RungeKutta4Stepper::postern()
{
  _owner->postern();
  for(SlaveStepperListIterator i = _slavesList.begin() ; 
      i != _slavesList.end() ; ++i)
    {
      (*i)->postern();
    }
}
