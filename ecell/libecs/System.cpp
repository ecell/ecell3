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

#include <algorithm>

#include "System.hpp"
#include "Reactor.hpp"
#include "RootSystem.hpp"
#include "SubstanceMaker.hpp"
#include "ReactorMaker.hpp"
#include "SystemMaker.hpp"
#include "Substance.hpp"
#include "Stepper.hpp"
#include "StepperMaker.hpp"
#include "FullID.hpp"


namespace libecs
{


  /////////////////////// System

  void System::makeSlots()
  {
    createPropertySlot( "SystemList", *this,
			NULLPTR, &System::getSystemList );
    createPropertySlot( "SubstanceList", *this,
			NULLPTR, &System::getSubstanceList );
    createPropertySlot( "ReactorList", *this,
			NULLPTR, &System::getReactorList );

    createPropertySlot( "StepperClass", *this,
			&System::setStepperClass, &System::getStepperClass );

    createPropertySlot( "Volume", *this,
			&System::setVolume, &System::getVolume );

    createPropertySlot( "StepInterval", *this,
			&System::setStepInterval, &System::getStepInterval );
  }


  // Message slots

  const UVariableVectorRCPtr System::getSystemList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSystemMap.size() );

    for( SystemMapConstIterator i = getSystemMap().begin() ;
	 i != getSystemMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr System::getSubstanceList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSubstanceMap.size() );

    for( SubstanceMapConstIterator i( getSubstanceMap().begin() );
	 i != getSubstanceMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr System::getReactorList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theReactorMap.size() );

    for( ReactorMapConstIterator i( getReactorMap().begin() );
	 i != getReactorMap().end() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }


  void System::setStepperClass( UVariableVectorCref aMessage )
  {
    //FIXME: range check
    setStepperClass( aMessage[0].asString() );
  }

  const UVariableVectorRCPtr System::getStepperClass() const
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    aVector->push_back( UVariable( getStepper()->getClassName() ) );
    return aVector;
  }

  System::System()
    :
    theVolume( 1 ),
    theVolumeBuffer( 1 ),
    theStepper( NULLPTR ),
    theRootSystem( NULLPTR ),
    theEntityListChanged( false )
  {
    makeSlots();
    theFirstRegularReactorIterator = getReactorMap().begin();
  }

  System::~System()
  {
    delete theStepper;
    
    for( ReactorMapIterator i( theReactorMap.begin() );
	 i != theReactorMap.end() ; ++i )
      {
	delete i->second;
      }

    for( SubstanceMapIterator i( theSubstanceMap.begin() );
	 i != theSubstanceMap.end() ; ++i )
      {
	delete i->second;
      }

    for( SystemMapIterator i( theSystemMap.begin() );
	 i != theSystemMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void System::setSuperSystem( SystemPtr aSystem )
  {
    Entity::setSuperSystem( aSystem );
    theRootSystem = getSuperSystem()->getRootSystem();
  }

  void System::setStepperClass( StringCref aClassname )
  {
    StepperPtr aStepper( getRootSystem()->
			 getStepperMaker().make( aClassname ) );
    aStepper->setOwner( this );

    theStepper = aStepper;
    theStepper->initialize();
  }

  void System::setStepInterval( RealCref aStepInterval )
  {
    theStepper->setStepInterval( aStepInterval );
  }

  const Real System::getStepInterval() const
  {
    return theStepper->getStepInterval();
  }

  const Real System::getStepsPerSecond() const
  {
    return theStepper->getStepsPerSecond();
  }

  void System::initialize()
  {
    updateVolume();

    if( theStepper == NULLPTR )
      {
	setStepperClass("SlaveStepper");
      }

    //
    // Substance::initialize()
    //
    for( SubstanceMapConstIterator i( getSubstanceMap().begin() );
	 i != getSubstanceMap().end() ; ++i )
      {
	i->second->initialize();
      }

    //
    // Reactor::initialize()
    //
    for( ReactorMapConstIterator i( getReactorMap().begin() );
	 i != getReactorMap().end() ; ++i )
      {
	i->second->initialize();
      }

    theFirstRegularReactorIterator = find_if( getReactorMap().begin(),
					      getReactorMap().end(),
					      isRegularReactorItem() );

    //
    // System::initialize()
    //
    for( SystemMapConstIterator i( getSystemMap().begin() );
	 i != getSystemMap().end() ; ++i )
      {
	i->second->initialize();
      }
  }

  void System::registerReactor( ReactorPtr aReactor )
  {
    if( getReactorMap().find( aReactor->getID() ) != getReactorMap().end() )
      {
	delete aReactor;
	//FIXME: throw exception
	return;
      }

    theReactorMap[ aReactor->getID() ] = aReactor;
    aReactor->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  ReactorPtr System::getReactor( StringCref anID ) 
  {
    ReactorMapConstIterator i( getReactorMap().find( anID ) );
    if( i == getReactorMap().end() )
      {
	throw NotFound( __PRETTY_FUNCTION__, "[" + getFullID().getString() + 
			"]: Reactor [" + anID + 
			"] not found in this System." );
      }
    return i->second;
  }

  void System::registerSubstance( SubstancePtr aSubstance )
  {
    if( getSubstanceMap().find( aSubstance->getID() ) 
	!= getSubstanceMap().end() )
      {
	delete aSubstance;
	//FIXME: throw exception
	return;
      }
    theSubstanceMap[ aSubstance->getID() ] = aSubstance;
    aSubstance->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  SubstancePtr System::getSubstance( StringCref anID ) 
  {
    SubstanceMapConstIterator i( getSubstanceMap().find( anID ) );
    if( i == getSubstanceMap().end() )
      {
	throw NotFound(__PRETTY_FUNCTION__, "[" + getFullID().getString() + 
		       "]: Substance [" + anID + 
		       "] not found in this System.");
      }

    return i->second;
  }


  void System::registerSystem( SystemPtr aSystem )
  {
    if( getSystemMap().find( aSystem->getID() ) != getSystemMap().end() )
      {
	delete aSystem;
	//FIXME: throw exception
	return;
      }

    theSystemMap[ aSystem->getID() ] = aSystem;
    aSystem->setSuperSystem( this );

    notifyChangeOfEntityList();
  }

  SystemPtr System::getSystem( SystemPathCref aSystemPath )
  {
    if( aSystemPath.empty() )
      {
	return this;
      }


    SystemPath aSystemPathCopy( aSystemPath );
    SystemPtr aSystem( this );

    // looping is faster than recursive search
    do
      {
	aSystem = aSystem->getSystem( aSystemPathCopy.front() );
	aSystemPathCopy.pop_front();
      }
    while( ! aSystemPathCopy.empty() );

    return aSystem;  
  }

  SystemPtr System::getSystem( StringCref anID ) 
  {
    SystemMapConstIterator i( getSystemMap().find( anID ) );
    if( i == getSystemMap().end() )
      {
	throw NotFound( __PRETTY_FUNCTION__, "[" + getFullID().getString() + 
			"]: System [" + anID + "] not found in this System." );
      }
    return i->second;
  }

  EntityPtr System::getEntity( FullIDCref aFullID )
  {
    SystemPtr aSystem ( getSystem( aFullID.getSystemPath() ) );

    switch( aFullID.getPrimitiveType() )
      {
      case PrimitiveType::SUBSTANCE:
	return aSystem->getSubstance( aFullID.getID() );
      case PrimitiveType::REACTOR:
	return aSystem->getReactor( aFullID.getID() );
      case PrimitiveType::SYSTEM:
	return aSystem->getSystem( aFullID.getID() );
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );
      }

    // NEVER_GET_HERE
    assert( 0 );
  }


  void System::createEntity( StringCref aClassname,
			     FullIDCref aFullID,
			     StringCref aName )
  {
    SystemPtr aSuperSystemPtr( getSystem( aFullID.getSystemPath() ) );

    ReactorPtr   aReactorPtr  ( NULLPTR );
    SystemPtr    aSystemPtr   ( NULLPTR );
    SubstancePtr aSubstancePtr( NULLPTR );

    switch( aFullID.getPrimitiveType() )
      {
      case PrimitiveType::SUBSTANCE:
	aSubstancePtr = getRootSystem()->
	  getSubstanceMaker().make( aClassname );
	aSubstancePtr->setID( aFullID.getID() );
	aSubstancePtr->setName( aName );
	aSuperSystemPtr->registerSubstance( aSubstancePtr );
	break;
      case PrimitiveType::REACTOR:
	aReactorPtr = getRootSystem()->getReactorMaker().make( aClassname );
	aReactorPtr->setID( aFullID.getID() );
	aReactorPtr->setName( aName );
	aSuperSystemPtr->registerReactor( aReactorPtr );
	break;
      case PrimitiveType::SYSTEM:
	aSystemPtr = getRootSystem()->getSystemMaker().make( aClassname );
	aSystemPtr->setID( aFullID.getID() );
	aSystemPtr->setName( aName );
	aSuperSystemPtr->registerSystem( aSystemPtr );
	break;

      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );

      }

  }


  const Real System::getActivityPerSecond() const
  {
    return getActivity() * getStepper()->getStepsPerSecond();
  }

  void System::notifyChangeOfEntityList()
  {
    //    getStepper()->getMasterStepper()->setEntityListChanged();
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
