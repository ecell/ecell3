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

#include "Reactor.hpp"
#include "Model.hpp"
#include "Substance.hpp"
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

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SystemList", *this,
				      Type2Type<PolymorphVectorRCPtr>(),
				      NULLPTR,
				      &System::getSystemList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SubstanceList", *this,
				      Type2Type<PolymorphVectorRCPtr>(),
				      NULLPTR,
				      &System::getSubstanceList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ReactorList", *this,
				      Type2Type<PolymorphVectorRCPtr>(),
				      NULLPTR,
				      &System::getReactorList ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "StepperID", *this,
				      Type2Type<String>(),
				      &System::setStepperID,
				      &System::getStepperID ) );

    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Volume", *this,
				      Type2Type<Real>(),
				      &System::setVolume, 
				      &System::getVolume ) );
  }


  // Property slots

  const PolymorphVectorRCPtr System::getSystemList() const
  {
    PolymorphVectorRCPtr aVectorPtr( new PolymorphVector );
    aVectorPtr->reserve( getSystemMap().size() );

    for( SystemMapConstIterator i = getSystemMap().begin() ;
	 i != getSystemMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const PolymorphVectorRCPtr System::getSubstanceList() const
  {
    PolymorphVectorRCPtr aVectorPtr( new PolymorphVector );
    aVectorPtr->reserve( getSubstanceMap().size() );

    for( SubstanceMapConstIterator i( getSubstanceMap().begin() );
	 i != getSubstanceMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const PolymorphVectorRCPtr System::getReactorList() const
  {
    PolymorphVectorRCPtr aVectorPtr( new PolymorphVector );
    aVectorPtr->reserve( getReactorMap().size() );

    for( ReactorMapConstIterator i( getReactorMap().begin() );
	 i != getReactorMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }


  System::System()
    :
    theStepper( NULLPTR ),
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
    if( theStepper == NULLPTR )
      {
	THROW_EXCEPTION( InitializationFailed,
			 getFullID().getString() + ": Stepper not set." );
      }

    // do not need to call subsystems' initialize() -- Stepper does this

  }

  void System::registerReactor( ReactorPtr aReactor )
  {
    getSuperSystem()->registerReactor( aReactor );
  }



  ReactorPtr System::getReactor( StringCref anID ) 
  {
    ReactorMapConstIterator i( getReactorMap().find( anID ) );

    if( i == getReactorMap().end() )
      {
	THROW_EXCEPTION( NotFound, 
			 "[" + getFullID().getString() + 
			 "]: Reactor [" + anID + 
			 "] not found in this System." );
      }

    return i->second;
  }

  void System::registerSubstance( SubstancePtr aSubstance )
  {
    getSuperSystem()->registerSubstance( aSubstance );
  }

  SubstancePtr System::getSubstance( StringCref anID ) 
  {
    SubstanceMapConstIterator i( getSubstanceMap().find( anID ) );
    if( i == getSubstanceMap().end() )
      {
	THROW_EXCEPTION( NotFound,
			 "[" + getFullID().getString() + 
			 "]: Substance [" + anID + 
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
    aSystem->setModel( getModel() );

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


  const Real System::getActivityPerSecond() const
  {
    return getActivity() * getStepper()->getStepsPerSecond();
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
    for( ReactorMapIterator i( theReactorMap.begin() );
	 i != theReactorMap.end() ; ++i )
      {
	delete i->second;
      }

  }

  void VirtualSystem::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "ReactorList", *this,
				      Type2Type<PolymorphVectorRCPtr>(),
				      NULLPTR,
				      &System::getReactorList ) );
  }


  void VirtualSystem::initialize()
  {
    System::initialize();

    //
    // Reactor::initialize()
    //
    for( ReactorMapConstIterator i( getReactorMap().begin() );
	 i != getReactorMap().end() ; ++i )
      {
	i->second->initialize();
      }

  }


  void VirtualSystem::registerReactor( ReactorPtr aReactor )
  {
    const String anID( aReactor->getID() );

    if( getReactorMap().find( anID ) != getReactorMap().end() )
      {
	delete aReactor;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Reactor [" + anID + "] already exist." );
      }

    theReactorMap[ anID ] = aReactor;
    aReactor->setSuperSystem( this );
    aReactor->setModel( getModel() );

    notifyChangeOfEntityList();
  }



  LogicalSystem::LogicalSystem()
  {
    makeSlots();
  }

  LogicalSystem::~LogicalSystem()
  {
    for( SubstanceMapIterator i( theSubstanceMap.begin() );
	 i != theSubstanceMap.end() ; ++i )
      {
	delete i->second;
      }
  }


  void LogicalSystem::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "SubstanceList", *this,
				      Type2Type<PolymorphVectorRCPtr>(),
				      NULLPTR,
				      &System::getSubstanceList ) );

  }

  void LogicalSystem::initialize()
  {
    VirtualSystem::initialize();

    //
    // Substance::initialize()
    //
    for( SubstanceMapConstIterator i( getSubstanceMap().begin() );
	 i != getSubstanceMap().end() ; ++i )
      {
	i->second->initialize();
      }

  }

  void LogicalSystem::registerSubstance( SubstancePtr aSubstance )
  {
    const String anID( aSubstance->getID() );

    if( getSubstanceMap().find( anID ) != getSubstanceMap().end() )
      {
	delete aSubstance;

	THROW_EXCEPTION( AlreadyExist, 
			 "[" + getFullID().getString() + 
			 "]: Substance [" + anID + "] already exist." );
      }

    theSubstanceMap[ anID ] = aSubstance;
    aSubstance->setSuperSystem( this );
    aSubstance->setModel( getModel() );

    notifyChangeOfEntityList();
  }


  CompartmentSystem::CompartmentSystem()
    :
    theVolume( 1.0 ),
    theVolumeBuffer( theVolume )
  {
    makeSlots();
  }

  CompartmentSystem::~CompartmentSystem()
  {
    ; // do nothing
  }

  void CompartmentSystem::makeSlots()
  {
    registerSlot( getPropertySlotMaker()->
		  createPropertySlot( "Volume", *this,
				      Type2Type<Real>(),
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
