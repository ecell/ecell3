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

#include "SubstanceMaker.hpp"
#include "ReactorMaker.hpp"
#include "SystemMaker.hpp"
#include "StepperMaker.hpp"
#include "Stepper.hpp"
#include "AccumulatorMaker.hpp"
#include "FQPI.hpp"

#include "RootSystem.hpp"


namespace libecs
{ 

  void RootSystem::makeSlots()
  {
    makeMessageSlot( "CurrentTime", RootSystem, *this,
		 NULLPTR, &RootSystem::getCurrentTime );
  }
  

  RootSystem::RootSystem() 
    :
    //    theStepperLeader( *new StepperLeader ),
    theReactorMaker( *new ReactorMaker ),
    theSubstanceMaker( *new SubstanceMaker ),
    theSystemMaker( *new SystemMaker ),
    theStepperMaker( *new StepperMaker ),
    theAccumulatorMaker( *new AccumulatorMaker )
  {
    makeSlots();

    setId( "/" );
    setName( "The RootSystem" );
    setRootSystem( this );
  }

  RootSystem::~RootSystem()
  {
    delete &theAccumulatorMaker;
    delete &theStepperMaker;
    delete &theSystemMaker;
    delete &theSubstanceMaker;
    delete &theReactorMaker;
    delete &theStepperLeader;
  }


  const Message RootSystem::getCurrentTime( StringCref keyword )
  {
    return Message( keyword, 
		    UVariable( theStepperLeader.getCurrentTime() ) );
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
  
  
    return status;
  }

  SystemPtr RootSystem::getSystem( SystemPathCref systempath ) 
  {
    if( systempath.first() != "/" )
      {
	throw BadID( __PRETTY_FUNCTION__,
		     "Fully qualified system path must start with '/'. ([" + 
		     systempath.getString() + "].");
      }

    return getSystem( systempath.getSystemPathString() );
  }

  SystemPtr RootSystem::getSystem( StringCref id ) 
  {
    // the root System(this!) is requested.
    if( id == "/" )
      {
	return this;
      }
  
    return System::getSystem( id );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
