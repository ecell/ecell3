
char const RootSystem_C_rcsid[] = "$Id$";
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




#include <stdio.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "ecell/MessageWindow.h"
#include "ecell/ECSMainWindow.h"
#include "InfoDialogManager.h"
#include "WorkingDialogManager.h"
#include "ecscore/RootSystem.h"
#include "ecscore/CellComponents.h"
#include "util/Datafile.h"
#include "ecscore/Primitive.h"
#include "ecscore/SubstanceMaker.h"
#include "ecscore/ReactorMaker.h"
#include "ecscore/SystemMaker.h"
#include "ecscore/AccumulatorMaker.h"

RootSystem::RootSystem() :
_stepperLeader(*new StepperLeader()),
_reactorMaker(*new ReactorMaker()),
_substanceMaker(*new SubstanceMaker()),
_systemMaker(*new SystemMaker()),
_stepperMaker(*new StepperMaker()),
_accumulatorMaker(*new AccumulatorMaker())
{
  // FIXME: remove this.
//  _stepper = new Eular1Stepper(this);
}

RootSystem::~RootSystem()
{
  delete &_reactorMaker;
  delete &_substanceMaker;
  delete &_systemMaker;

  delete &_stepperLeader;
  delete &_accumulatorMaker;
}

void RootSystem::initialize()
{
  MetaSystem::initialize();
  stepperLeader().initialize();
  stepperLeader().update();
}

int RootSystem::check()
{
  bool status = true;
  
  *theMessageWindow << "Reactor initialization ";
  switch(Reactor::globalCondition())
    {
    case Reactor::Good:
      *theMessageWindow 
        << "succeeded. condition Good.\n";
      break;
    case Reactor::InitFail:
      *theMessageWindow << "condition InitFail.\n";
    default:
      *theMessageWindow << "initialization failed. "
        << "trying to continue... \n";
      theInfoDialogManager->post("Reactor initialization failed.");
      status = false;
    }
  
/*
  if(!_Cell)
    {
      *theMessageWindow << "no cell object found. " <<
	"check your rule file again.\n";
      theInfoDialogManager->post("no cell object found.");
      status = false;
    }
  if(!_Environment)
    {
      *theMessageWindow << "no environment object registered... "
	<< " ok, continue without environment\n";
    }
*/

//  *theMessageWindow << systemMaker().numInstance()
//    << " systems.\n";
//  *theMessageWindow << substanceMaker().numInstance() 
//    << " substances.\n";
//  *theMessageWindow << reactorMaker().numInstance()
//    << " reactors.\n";


  return status;
}

/*
void RootSystem::step()
{
  clear();
  react();
  transit();
}

void RootSystem::clear()
{
  // clear phase
//  MetaSystem::clear();
  _stepperLeader.clear();

}


void RootSystem::react()
{
  // react phase
//  MetaSystem::react();
  _stepperLeader.react();

}


void RootSystem::transit()
{
  // transit phase
//  MetaSystem::transit();
  _stepperLeader.transit();
}
*/

System* RootSystem::findSystem(const SystemPath& systempath) 
throw(NotFound,MalformedSystemName)
{
  if(systempath.first() != "/")
    throw MalformedSystemName(__PRETTY_FUNCTION__,
			      "system path given to this function must"
			      + string("start with '/'. ([") + 
			      systempath.systemPathString() + "].");
   
   // If the root System(this!) is requested.
   // The culture is '/', not '/CULTURE'.
  SystemPath next = systempath.next();

  if(next.systemPathString() == "")
    return this;

  return MetaSystem::findSystem(next);
}

Primitive RootSystem::getPrimitive(const FQPN& fqpn) throw(UnmatchedSystemClass,
							InvalidPrimitiveType,
							NotFound)
{
  // FIXME: handle exceptions

  System* sys = findSystem(fqpn.systemPathString());
  return sys->getPrimitive(fqpn.entrynameString(),fqpn.type());
}
