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

#include "RootSystem.hpp"
#include "Primitive.hpp"
#include "FQPN.hpp"
#include "SubstanceMaker.hpp"
#include "ReactorMaker.hpp"
#include "SystemMaker.hpp"
#include "AccumulatorMaker.hpp"

RootSystem::RootSystem() 
  :
  theStepperLeader(    *new StepperLeader()    ),
  theReactorMaker(     *new ReactorMaker()     ),
  theSubstanceMaker(   *new SubstanceMaker()   ),
  theSystemMaker(      *new SystemMaker()      ),
  theStepperMaker(     *new StepperMaker()     ),
  theAccumulatorMaker( *new AccumulatorMaker() )
{
  // FIXME: remove this.
//  _stepper = new Eular1Stepper(this);
}

RootSystem::~RootSystem()
{
  delete &theReactorMaker;
  delete &theSubstanceMaker;
  delete &theSystemMaker;

  delete &theStepperLeader;
  delete &theAccumulatorMaker;
}

void RootSystem::initialize()
{
  System::initialize();
  getStepperLeader().initialize();
  getStepperLeader().update();
}

int RootSystem::check()
{
  bool status = true;
  
  //FIXME:  *theMessageWindow << "Reactor initialization ";
//FIXME  switch(Reactor::globalCondition())
//FIXME    {
//FIXME    case Reactor::Good:
//FIXME      *theMessageWindow 
//FIXME        << "succeeded. condition Good.\n";
//FIXME      break;
//FIXME    case Reactor::InitFail:
//FIXME      *theMessageWindow << "condition InitFail.\n";
//FIXME    default:
//FIXME      *theMessageWindow << "initialization failed. "
//FIXME        << "trying to continue... \n";
//FIXME      theInfoDialogManager->post("Reactor initialization failed.");
//FIXME      status = false;
//FIXME    }
  
//  *theMessageWindow << systemMaker().numInstance()
//    << " systems.\n";
//  *theMessageWindow << substanceMaker().numInstance() 
//    << " substances.\n";
//  *theMessageWindow << reactorMaker().numInstance()
//    << " reactors.\n";

  return status;
}

SystemPtr RootSystem::getSystem( SystemPathCref systempath ) 
  throw( NotFound, MalformedSystemName )
{
  if( systempath.first() != "/" )
    {
      throw MalformedSystemName( __PRETTY_FUNCTION__,
				 "system path given to this function must" +
				 String( "start with '/'. ([" ) + 
				 systempath.getString() + "].");
    }
   
  SystemPath next = systempath.next();

  // the root System(this!) is requested.
  if( next.SystemPath::getString() == "" )
    {
      return this;
    }

  return System::getSystem( next );
}

Primitive RootSystem::getPrimitive( FQPNCref fqpn ) 
  throw( InvalidPrimitiveType, NotFound )
{
  // FIXME: handle exceptions

  SystemPtr sys = getSystem( fqpn.SystemPath::getString() );
  return sys->getPrimitive( fqpn.getId(), fqpn.getType() );
}


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
