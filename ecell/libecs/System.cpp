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

    createPropertySlot( "Stepper", *this,
			&System::setStepperClass, &System::getStepperClass );

    createPropertySlot( "Volume", *this,
			&System::setVolume, &System::getVolume );

    createPropertySlot( "StepInterval", *this,
			NULLPTR, &System::getStepInterval );
  }


  // Message slots

  const UVariableVectorRCPtr System::getSystemList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSystemMap.size() );

    for( SystemMapConstIterator i = getFirstSystemIterator() ;
	 i != getLastSystemIterator() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr System::getSubstanceList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theSubstanceMap.size() );

    for( SubstanceMapConstIterator i = getFirstSubstanceIterator() ;
	 i != getLastSubstanceIterator() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }

  const UVariableVectorRCPtr System::getReactorList() const
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->reserve( theReactorMap.size() );

    for( ReactorMapConstIterator i = getFirstReactorIterator() ;
	 i != getLastReactorIterator() ; ++i )
      {
	aVectorPtr->push_back( i->second->getID() );
      }

    return aVectorPtr;
  }


  void System::setStepperClass( UVariableVectorCref uvector )
  {
    //FIXME: range check
    setStepperClass( uvector[0].asString() );
  }

  const UVariableVectorRCPtr System::getStepperClass() const
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    aVector->push_back( UVariable( getStepper()->className() ) );
    return aVector;
  }

  System::System()
    :
    theVolume( 1 ),
    theVolumeBuffer( 1 ),
    theStepper( NULLPTR ),
    theRootSystem( NULLPTR )
  {
    makeSlots();
    theFirstRegularReactorIterator = theReactorMap.begin();
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

  void System::setSuperSystem( SystemPtr const supersystem )
  {
    Entity::setSuperSystem( supersystem );
    theRootSystem = getSuperSystem()->getRootSystem();
  }

  void System::setStepperClass( StringCref classname )
  {
    StepperPtr aStepper( getRootSystem()->
			 getStepperMaker().make( classname ) );
    aStepper->setOwner( this );

    MasterStepperPtr 
      aMasterStepper( dynamic_cast<MasterStepperPtr>( aStepper ) );
    if( aMasterStepper != NULLPTR )
      {
	getRootSystem()->getStepperLeader().
	  registerMasterStepper( aMasterStepper );
      }

    theStepper = aStepper;
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
    if( theStepper == NULLPTR )
      {
	//FIXME: make this default user customizable
	setStepperClass( "Euler1Stepper" );
      }

    theStepper->initialize();

    //
    // Substance::initialize()
    //
    for( SubstanceMapConstIterator i( getFirstSubstanceIterator() ); 
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->initialize();
      }

    //
    // Reactor::initialize()
    //
    for( ReactorMapConstIterator i( getFirstReactorIterator() );
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->initialize();
      }

    theFirstRegularReactorIterator = find_if( theReactorMap.begin(),
					      theReactorMap.end(),
					      isRegularReactorItem() );

    //
    // System::initialize()
    //
    for( SystemMapConstIterator i( getFirstSystemIterator() );
	 i != getLastSystemIterator(); ++i )
      {
	i->second->initialize();
      }
  }

  void System::clear()
  {
    //
    // Substance::clear()
    //
    for( SubstanceMapConstIterator i( getFirstSubstanceIterator() );
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->clear();
      }
  }

  void System::react()
  {
    for( ReactorMapConstIterator i( getFirstRegularReactorIterator() ); 
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->react();
      }
  }

  void System::turn()
  {
    for( SubstanceMapConstIterator i( getFirstSubstanceIterator() );
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->turn();
      }
  }

  void System::transit()
  {
    for( ReactorMapConstIterator i( getFirstRegularReactorIterator() ); 
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->transit();
      }

    for( SubstanceMapConstIterator i( getFirstSubstanceIterator() );
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->transit();
      }

    updateVolume();
  }

  void System::compute()
  {
    for( ReactorMapConstIterator i( getFirstReactorIterator() ); 
	 i != getFirstRegularReactorIterator() ; ++i )
      {
	i->second->react();
      }

    // update activity of posterior reactors by buffered values 
    for( ReactorMapConstIterator i( getFirstReactorIterator() ); 
	 i != getFirstRegularReactorIterator() ; ++i )
      {
	i->second->transit();
      }
  }

  void System::registerReactor( ReactorPtr reactor )
  {
    if( containsReactor( reactor->getID() ) )
      {
	delete reactor;
	//FIXME: throw exception
	return;
      }

    theReactorMap[ reactor->getID() ] = reactor;
    reactor->setSuperSystem( this );
  }

  ReactorPtr System::getReactor( StringCref id ) 
  {
    ReactorMapIterator i( getReactorIterator( id ) );
    if( i == getLastReactorIterator() )
      {
	throw NotFound( __PRETTY_FUNCTION__, "[" + getFullID().getString() + 
			"]: Reactor [" + id + "] not found in this System." );
      }
    return i->second;
  }

  void System::registerSubstance( SubstancePtr newone )
  {
    if( containsSubstance( newone->getID() ) )
      {
	delete newone;
	//FIXME: throw exception
	return;
      }
    theSubstanceMap[ newone->getID() ] = newone;
    newone->setSuperSystem( this );
  }

  SubstancePtr System::getSubstance( StringCref id ) 
  {
    SubstanceMapIterator i = getSubstanceIterator( id );
    if( i == getLastSubstanceIterator() )
      {
	throw NotFound(__PRETTY_FUNCTION__, "[" + getFullID().getString() + 
		       "]: Substance [" + id + "] not found in this System.");
      }

    return i->second;
  }


  void System::registerSystem( SystemPtr system )
  {
    if( containsSystem( system->getID() ) )
      {
	delete system;
	//FIXME: throw exception
	return;
      }

    theSystemMap[ system->getID() ] = system;
    system->setSuperSystem( this );

  }

  SystemPtr System::getSystem( SystemPathCref systempath )
  {
    if( systempath.empty() )
      {
	return this;
      }

    SystemPath aSystemPath( systempath );
    SystemPtr aSystem( this );

    // looping is faster than recursive search
    do
      {
	aSystem = aSystem->getSystem( aSystemPath.front() );
	aSystemPath.pop_front();
      }
    while( ! aSystemPath.empty() );

    return aSystem;  
  }

  SystemPtr System::getSystem( StringCref id ) 
  {
    SystemMapIterator i( getSystemIterator( id ) );
    if( i == getLastSystemIterator() )
      {
	throw NotFound( __PRETTY_FUNCTION__, "[" + getFullID().getString() + 
			"]: System [" + id + "] not found in this System." );
      }
    return i->second;
  }

  EntityPtr System::getEntity( FullIDCref fullid )
  {
    SystemPtr aSystem ( getSystem( fullid.getSystemPath() ) );

    switch( fullid.getPrimitiveType() )
      {
      case PrimitiveType::SUBSTANCE:
	return aSystem->getSubstance( fullid.getID() );
      case PrimitiveType::REACTOR:
	return aSystem->getReactor( fullid.getID() );
      case PrimitiveType::SYSTEM:
	return aSystem->getSystem( fullid.getID() );
      default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__, 
				    "bad PrimitiveType specified." );
      }

    // NEVER_GET_HERE
    assert( 0 );
  }


  void System::createEntity( StringCref classname,
			     FullIDCref fullid, 
			     StringCref name )
  {
    SystemPtr aSuperSystemPtr( getSystem( fullid.getSystemPath() ) );

    ReactorPtr   aReactorPtr  ( NULLPTR );
    SystemPtr    aSystemPtr   ( NULLPTR );
    SubstancePtr aSubstancePtr( NULLPTR );

    switch( fullid.getPrimitiveType() )
      {
      case PrimitiveType::SUBSTANCE:
	aSubstancePtr = getRootSystem()->getSubstanceMaker().make( classname );
	aSubstancePtr->setID( fullid.getID() );
	aSubstancePtr->setName( name );
	aSuperSystemPtr->registerSubstance( aSubstancePtr );
	break;
      case PrimitiveType::REACTOR:
	aReactorPtr = getRootSystem()->getReactorMaker().make( classname );
	aReactorPtr->setID( fullid.getID() );
	aReactorPtr->setName( name );
	aSuperSystemPtr->registerReactor( aReactorPtr );
	break;
      case PrimitiveType::SYSTEM:
	aSystemPtr = getRootSystem()->getSystemMaker().make( classname );
	aSystemPtr->setID( fullid.getID() );
	aSystemPtr->setName( name );
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

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
