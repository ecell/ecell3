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
#include "FQPI.hpp"

RootSystem::RootSystem() 
{
  // FIXME: remove this.
//  _stepper = new Euler1Stepper(this);
}

RootSystem::~RootSystem()
{
  ; // do nothing
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
  
  return status;
}

SystemPtr RootSystem::getSystem( SystemPathCref systempath ) 
  throw( NotFound, BadID )
{
  if( systempath.first() != "/" )
    {
      throw BadID( __PRETTY_FUNCTION__,
		   "Fully qualified system path must" +
		   String( "start with '/'. ([" ) + 
		   systempath.getString() + "].");
    }
   
  SystemPath next( systempath.next() );

  // the root System(this!) is requested.
  if( next.getSystemPathString() == "" )
    {
      return this;
    }

  return System::getSystem( next );
}

#if 0
EntityPtr RootSystem::getEntity( FQPICref fqpi ) 
  throw( InvalidPrimitiveType, NotFound )
{
  // FIXME: handle exceptions?

  SystemPtr sys = getSystem( fqpi.getSystemPathString() );
  return sys->getEntity( fqpi.getPrimitiveType(), fqpi.getIdString() );
}
#endif /* 0 */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
