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
    makePropertySlot( "SystemList", System, *this,
		      NULLPTR, &System::getSystemList );
    makePropertySlot( "SubstanceList", System, *this,
		      NULLPTR, &System::getSubstanceList );
    makePropertySlot( "ReactorList", System, *this,
		      NULLPTR, &System::getReactorList );

    makePropertySlot( "Stepper", System, *this,
		      &System::setStepper, &System::getStepper );
    makePropertySlot( "VolumeIndex", System, *this,
		      &System::setVolumeIndex, &System::getVolumeIndex );

    makePropertySlot( "Volume", System, *this,
		      NULLPTR, &System::getVolume );

    makePropertySlot( "StepInterval", System, *this,
		      NULLPTR, &System::getStepInterval );
  }


  // Message slots

  const Message System::getSystemList( StringCref keyword )
  {
    UConstantVector aVector;

    for( SystemMapIterator i = getFirstSystemIterator() ;
	 i != getLastSystemIterator() ; ++i )
      {
	aVector.push_back( UConstant( i->second->getID() ) );
      }

    return Message( keyword, aVector );
  }

  const Message System::getSubstanceList( StringCref keyword )
  {
    UConstantVector aVector;

    for( SubstanceMapIterator i = getFirstSubstanceIterator() ;
	 i != getLastSubstanceIterator() ; ++i )
      {
	aVector.push_back( UConstant( i->second->getID() ) );
      }

    return Message( keyword, aVector );
  }

  const Message System::getReactorList( StringCref keyword )
  {
    UConstantVector aVector;

    for( ReactorMapIterator i = getFirstReactorIterator() ;
	 i != getLastReactorIterator() ; ++i )
      {
	aVector.push_back( UConstant( i->second->getID() ) );
      }

    return Message( keyword, aVector );
  }


  void System::setStepper( const Message& message )
  {
    //FIXME: range check
    setStepper( message[0].asString() );
  }

  const Message System::getStepper( StringCref keyword )
  {
    return Message( keyword, 
		    UConstant( getStepper()->className() ) );
  }

  void System::setVolumeIndex( const Message& message )
  {
    //FIXME: range check
    setVolumeIndex( FullID( message[0].asString() ) );
  }

  const Message System::getVolumeIndex( StringCref keyword )
  {
    if( ! haveVolumeIndex() )
      {
	return Message( keyword );
      }

    return Message( keyword, 
		    UConstant( getVolumeIndex()->getFullID().getString() ) );
  }

  const Message System::getVolume( StringCref keyword )
  {
    if( haveVolumeIndex() )
      {
	return Message( keyword, 
			UConstant( getVolume() ) ) ;
      }
    else
      {
	return Message( keyword );
      }
  }

  const Message System::getStepInterval( StringCref keyword )
  {
    return Message( keyword, 
		    UConstant( getStepInterval() ) ) ;
  }




  System::System()
    :
    theVolumeIndex( NULLPTR ),
    theStepper( NULLPTR ),
    theRootSystem( NULLPTR )
  {
    makeSlots();
    theFirstRegularReactorIterator = getFirstReactorIterator();
  }

  System::~System()
  {
    delete theStepper;
  }

  void System::setSuperSystem( SystemPtr const supersystem )
  {
    Entity::setSuperSystem( supersystem );
    theRootSystem = getSuperSystem()->getRootSystem();
  }

  void System::setStepper( StringCref classname )
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

  Real System::getVolume() 
  {
    return theVolumeIndex->getActivityPerSecond();
  }

  RealCref System::getStepInterval() const
  {
    return theStepper->getStepInterval();
  }

  RealCref System::getStepsPerSecond() const
  {
    return theStepper->getStepsPerSecond();
  }

  void System::setVolumeIndex( FullIDCref volumeindex )
  {
    SystemPtr aSystem( theRootSystem->
		       getSystem( volumeindex.getSystemPath() ) );
    theVolumeIndex = aSystem->getReactor( volumeindex.getID() );
  }

  void System::initialize()
  {
    if( theStepper == NULLPTR )
      {
	//FIXME: make this default user customizable
	setStepper( "Euler1Stepper" );
      }

    theStepper->initialize();

    //
    // Substance::initialize()
    //
    for( SubstanceMapIterator i( getFirstSubstanceIterator() ); 
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->initialize();
      }

    //
    // Reactor::initialize()
    //
    for( ReactorMapIterator i( getFirstReactorIterator() );
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->initialize();
      }

    theFirstRegularReactorIterator = find_if( getFirstReactorIterator(), 
					      getLastReactorIterator(),
					      isRegularReactorItem() );

    //
    // System::initialize()
    //
    for( SystemMapIterator i( getFirstSystemIterator() );
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
    for( SubstanceMapIterator i( getFirstSubstanceIterator() );
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->clear();
      }
  }

  void System::react()
  {
    for( ReactorMapIterator i( getFirstRegularReactorIterator() ); 
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->react();
      }
  }

  void System::turn()
  {
    for( SubstanceMapIterator i( getFirstSubstanceIterator() );
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->turn();
      }
  }

  void System::transit()
  {
    for( ReactorMapIterator i( getFirstRegularReactorIterator() ); 
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->transit();
      }

    for( SubstanceMapIterator i( getFirstSubstanceIterator() );
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->transit();
      }
  }

  void System::postern()
  {
    for( ReactorMapIterator i( getFirstReactorIterator() ); 
	 i != getFirstRegularReactorIterator() ; ++i )
      {
	i->second->react();
      }

    // update activity of posterior reactors by buffered values 
    for( ReactorMapIterator i( getFirstReactorIterator() ); 
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

    theSubsystemMap[ system->getID() ] = system;
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


  Real System::getActivityPerSecond()
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
