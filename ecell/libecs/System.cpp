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

#include <algorithm>

#include "Process.hpp"
#include "Model.hpp"
#include "Variable.hpp"
#include "Stepper.hpp"
#include "FullID.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"

#include "System.hpp"


namespace libecs
{


  /////////////////////// System

  void System::makeSlots()
  {

    registerSlot( "SystemList",  
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Polymorph>(),
				      NULLPTR,
				      &System::getSystemList ) );

    registerSlot( "VariableList",  
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Polymorph>(),
				      NULLPTR,
				      &System::getVariableList ) );

    registerSlot( "ProcessList", 
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Polymorph>(),
				      NULLPTR,
				      &System::getProcessList ) );

    registerSlot( "StepperID", 
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<String>(),
				      &System::setStepperID,
				      &System::getStepperID ) );

    registerSlot( "Volume",  
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Real>(),
				      &System::setVolume, 
				      &System::getVolume ) );
  }


  // Property slots

  const Polymorph System::getSystemList() const
  {
    PolymorphVector aVector;
    aVector.reserve( getSystemMap().size() );

    for( SystemMapConstIterator i = getSystemMap().begin() ;
	 i != getSystemMap().end() ; ++i )
      {
	aVector.push_back( i->second->getID() );
      }

    return aVector;
  }

  const Polymorph System::getVariableList() const
  {
    PolymorphVector aVector;
    aVector.reserve( getVariableMap().size() );

    for( VariableMapConstIterator i( getVariableMap().begin() );
	 i != getVariableMap().end() ; ++i )
      {
	aVector.push_back( i->second->getID() );
      }

    return aVector;
  }

  const Polymorph System::getProcessList() const
  {
    PolymorphVector aVector;
    aVector.reserve( getProcessMap().size() );

    for( ProcessMapConstIterator i( getProcessMap().begin() );
	 i != getProcessMap().end() ; ++i )
      {
	aVector.push_back( i->second->getID() );
      }

    return aVector;
  }


  System::System()
    :
    theStepper( NULLPTR ),
    theModel( NULLPTR ),
    theEntityListChanged( false )
  {
    makeSlots();
  }

  System::~System()
  {
    getStepper()->removeSystem( this );
    
    for( SystemMapIterator i( theSystemMap.begin() );
	 i != theSystemMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void System::setStepperID( StringCref anID )
  {
    theStepper = getModel()->getStepper( anID );
    theStepper->registerSystem( this );
  }

  const String System::getStepperID() const
  {
    return getStepper()->getID();
  }

  void System::initialize()
  {
    // do not need to call subsystems' initialize() -- caller does this

  }

  void System::registerProcess( ProcessPtr aProcess )
  {
    getSuperSystem()->registerProcess( aProcess );
  }



  ProcessPtr System::getProcess( StringCref anID ) 
  {
    ProcessMapConstIterator i( getProcessMap().find( anID ) );

    if( i == getProcessMap().end() )
      {
	THROW_EXCEPTION( NotFound, 
			 "[" + getFullID().getString() + 
			 "]: Process [" + anID + 
			 "] not found in this System." );
      }

    return i->second;
  }

  void System::registerVariable( VariablePtr aVariable )
  {
    getSuperSystem()->registerVariable( aVariable );
  }

  VariablePtr System::getVariable( StringCref anID ) 
  {
    VariableMapConstIterator i( getVariableMap().find( anID ) );
    if( i == getVariableMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: Variable [" + anID + 
			 "] not found in this System.");
      }

    return i->second;
  }


  void System::registerSystem( SystemPtr aSystem )
  {
    const String anID( aSystem->getID() );

    if( getSystemMap().find( anID ) != getSystemMap().end() )
      {
	delete aSystem;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: System [" + anID + "] already exist." );
      }

    theSystemMap[ anID ] = aSystem;
    aSystem->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  SystemPtr System::getSystem( StringCref anID ) 
  {
    SystemMapConstIterator i( getSystemMap().find( anID ) );
    if( i == getSystemMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: System [" + anID + 
			 "] not found in this System." );
      }
    return i->second;
  }

  void System::notifyChangeOfEntityList()
  {
    //    getStepper()->getMasterStepper()->setEntityListChanged();
  }

  const SystemPath System::getSystemPath() const
  {
    if( isRootSystem() )
      {
	return SystemPath();
      }
    else
      {
	return Entity::getSystemPath();
      }
  }

  VirtualSystem::VirtualSystem()
  {
    makeSlots();
  }

  VirtualSystem::~VirtualSystem()
  {
    for( ProcessMapIterator i( theProcessMap.begin() );
	 i != theProcessMap.end() ; ++i )
      {
	delete i->second;
      }

  }

  void VirtualSystem::makeSlots()
  {
    registerSlot( "ProcessList",  
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Polymorph>(),
				      NULLPTR,
				      &System::getProcessList ) );
  }


  void VirtualSystem::initialize()
  {
    System::initialize();

    //
    // Process::initialize()
    //
    for( ProcessMapConstIterator i( getProcessMap().begin() );
	 i != getProcessMap().end() ; ++i )
      {
	ProcessPtr aProcessPtr( i->second );

	if( aProcessPtr->getStepper() == NULLPTR )
	  {
	    aProcessPtr->setStepper( getStepper() );
	  }

	aProcessPtr->initialize();
      }

  }


  void VirtualSystem::registerProcess( ProcessPtr aProcess )
  {
    const String anID( aProcess->getID() );

    if( getProcessMap().find( anID ) != getProcessMap().end() )
      {
	delete aProcess;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Process [" + anID + "] already exist." );
      }

    theProcessMap[ anID ] = aProcess;
    aProcess->setSuperSystem( this );

    notifyChangeOfEntityList();
  }



  LogicalSystem::LogicalSystem()
  {
    makeSlots();
  }

  LogicalSystem::~LogicalSystem()
  {
    for( VariableMapIterator i( theVariableMap.begin() );
	 i != theVariableMap.end() ; ++i )
      {
	delete i->second;
      }
  }


  void LogicalSystem::makeSlots()
  {
    registerSlot( "VariableList",  
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Polymorph>(),
				      NULLPTR,
				      &System::getVariableList ) );

  }

  void LogicalSystem::initialize()
  {
    VirtualSystem::initialize();

    //
    // Variable::initialize()
    //
    for( VariableMapConstIterator i( getVariableMap().begin() );
	 i != getVariableMap().end() ; ++i )
      {
	i->second->initialize();
      }

  }

  void LogicalSystem::registerVariable( VariablePtr aVariable )
  {
    const String anID( aVariable->getID() );

    if( getVariableMap().find( anID ) != getVariableMap().end() )
      {
	delete aVariable;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Variable [" + anID + "] already exist." );
      }

    theVariableMap[ anID ] = aVariable;
    aVariable->setSuperSystem( this );

    notifyChangeOfEntityList();
  }


  CompartmentSystem::CompartmentSystem()
    :
    theVolume( 1.0 )
  {
    makeSlots();
  }

  CompartmentSystem::~CompartmentSystem()
  {
    ; // do nothing
  }

  void CompartmentSystem::makeSlots()
  {
    registerSlot( "Volume",  
		  getPropertySlotMaker()->
		  createPropertySlot( *this, Type2Type<Real>(),
				      &System::setVolume, 
				      &System::getVolume ) );
  }

  void CompartmentSystem::initialize()
  {
    LogicalSystem::initialize();
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
