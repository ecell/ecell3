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
#include "Substance.hpp"
#include "Stepper.hpp"
#include "StepperMaker.hpp"
#include "FQPI.hpp"



namespace libecs
{


  /////////////////////// System

  void System::makeSlots()
  {
    makeMessageSlot( "SystemList", System, *this,
		 NULLPTR, &System::getSystemList );
    makeMessageSlot( "SubstanceList", System, *this,
		 NULLPTR, &System::getSubstanceList );
    makeMessageSlot( "ReactorList", System, *this,
		 NULLPTR, &System::getReactorList );

    makeMessageSlot( "Stepper", System, *this,
		 &System::setStepper, &System::getStepper );
    makeMessageSlot( "VolumeIndex", System, *this,
		 &System::setVolumeIndex, &System::getVolumeIndex );

    makeMessageSlot( "Volume", System, *this,
		 NULLPTR, &System::getVolume );

    makeMessageSlot( "DeltaT", System, *this,
		 NULLPTR, &System::getDeltaT );
  }


  // Message slots

  const Message System::getSystemList( StringCref keyword )
  {
    UniversalVariableVector aVector;

    for( SystemMapIterator i = getFirstSystemIterator() ;
	 i != getLastSystemIterator() ; ++i )
      {
	aVector.push_back( UniversalVariable( i->second->getId() ) );
      }

    return Message( keyword, aVector );
  }

  const Message System::getSubstanceList( StringCref keyword )
  {
    UniversalVariableVector aVector;

    for( SubstanceMapIterator i = getFirstSubstanceIterator() ;
	 i != getLastSubstanceIterator() ; ++i )
      {
	aVector.push_back( UniversalVariable( i->second->getId() ) );
      }

    return Message( keyword, aVector );
  }

  const Message System::getReactorList( StringCref keyword )
  {
    UniversalVariableVector aVector;

    for( ReactorMapIterator i = getFirstReactorIterator() ;
	 i != getLastReactorIterator() ; ++i )
      {
	aVector.push_back( UniversalVariable( i->second->getId() ) );
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
		    UniversalVariable( getStepper()->className() ) );
  }

  void System::setVolumeIndex( const Message& message )
  {
    //FIXME: range check
    setVolumeIndex( FQID( message[0].asString() ) );
  }

  const Message System::getVolumeIndex( StringCref keyword )
  {
    if( getVolumeIndex() == NULLPTR )
      {
	return Message( keyword );
      }

    return Message( keyword, 
		    UniversalVariable( getVolumeIndex()->getFqid() ) );
  }

  const Message System::getVolume( StringCref keyword )
  {
    return Message( keyword, 
		    UniversalVariable( getVolume() ) ) ;
  }

  const Message System::getDeltaT( StringCref keyword )
  {
    return Message( keyword, 
		    UniversalVariable( getDeltaT() ) ) ;
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

  const String System::getFqpi() const
  {
    return PrimitiveTypeStringOf( *this ) + ":" + getFqid();
  }

  void System::setStepper( StringCref classname )
  {
    StepperPtr aStepper;
    aStepper = getRootSystem()->getStepperMaker().make( classname );
    aStepper->setOwner( this );

    MasterStepperPtr aMasterStepper( NULLPTR );
    if( ( aMasterStepper = dynamic_cast<MasterStepperPtr>(aStepper) ) 
	!= NULLPTR )
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

  Real System::getDeltaT() const
  {
    return theStepper->getDeltaT();
  }

  void System::setVolumeIndex( FQIDCref volumeindex )
  {
    SystemPtr aSystem = theRootSystem->getSystem( SystemPath( volumeindex ) );
    theVolumeIndex = aSystem->getReactor( volumeindex.getIdString() );
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
    for( SubstanceMapIterator i = getFirstSubstanceIterator() ; 
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->initialize();
      }

    //
    // Reactor::initialize()
    //
    for( ReactorMapIterator i = getFirstReactorIterator() ;
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
    for( SystemMapIterator i = getFirstSystemIterator();
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
    for( SubstanceMapIterator i = getFirstSubstanceIterator() ; 
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->clear();
      }
  }

  void System::react()
  {
    for( ReactorMapIterator i = getFirstRegularReactorIterator() ; 
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->react();
      }
  }

  void System::turn()
  {
    for( SubstanceMapIterator i = getFirstSubstanceIterator() ; 
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->turn();
      }
  }

  void System::transit()
  {
    for( ReactorMapIterator i = getFirstRegularReactorIterator() ; 
	 i != getLastReactorIterator() ; ++i )
      {
	i->second->transit();
      }

    for( SubstanceMapIterator i = getFirstSubstanceIterator() ;
	 i != getLastSubstanceIterator() ; ++i )
      {
	i->second->transit();
      }
  }

  void System::postern()
  {
    for( ReactorMapIterator i = getFirstReactorIterator() ; 
	 i != getFirstRegularReactorIterator() ; ++i )
      {
	i->second->react();
      }

    // update activity of posterior reactors by buffered values 
    for( ReactorMapIterator i = getFirstReactorIterator() ; 
	 i != getFirstRegularReactorIterator() ; ++i )
      {
	i->second->transit();
      }
  }

  void System::registerReactor( ReactorPtr reactor )
  {
    if( containsReactor( reactor->getId() ) )
      {
	delete reactor;
	//FIXME: throw exception
	return;
      }

    theReactorMap[ reactor->getId() ] = reactor;
    reactor->setSuperSystem( this );
  }

  ReactorPtr System::getReactor( StringCref id ) 
  {
    ReactorMapIterator i = getReactorIterator( id );
    if( i == getLastReactorIterator() )
      {
	throw NotFound( __PRETTY_FUNCTION__, "[" + getFqid() + 
			"]: Reactor [" + id + "] not found in this System." );
      }
    return i->second;
  }

  void System::registerSubstance( SubstancePtr newone )
  {
    if( containsSubstance( newone->getId() ) )
      {
	delete newone;
	//FIXME: throw exception
	return;
      }
    theSubstanceMap[ newone->getId() ] = newone;
    newone->setSuperSystem( this );
  }

  SubstancePtr System::getSubstance( StringCref id ) 
  {
    SubstanceMapIterator i = getSubstanceIterator( id );
    if( i == getLastSubstanceIterator() )
      {
	throw NotFound(__PRETTY_FUNCTION__, "[" + getFqid() + 
		       "]: Substance [" + id + "] not found in this System.");
      }

    return i->second;
  }


  void System::registerSystem( SystemPtr system )
  {
    if( containsSystem( system->getId() ) )
      {
	delete system;
	//FIXME: throw exception
	return;
      }

    theSubsystemMap[ system->getId() ] = system;
    system->setSuperSystem( this );

  }

  SystemPtr System::getSystem( SystemPathCref systempath ) 
  {
    SystemPtr  aSystem = getSystem( systempath.first() );
    SystemPath anNext  = systempath.next();

    if( anNext.getString() != "" ) // not a leaf
      {
	aSystem = aSystem->getSystem( anNext );
      }
  
    return aSystem;
  }

  SystemPtr System::getSystem( StringCref id ) 
  {
    SystemMapIterator i = getSystemIterator( id );
    if( i == getLastSystemIterator() )
      {
	throw NotFound(__PRETTY_FUNCTION__, "[" + getFqid() + 
		       "]: System [" + id + "] not found in this System.");
      }
    return i->second;
  }

  Real System::getActivityPerSecond()
  {
    return getActivity() / getStepper()->getDeltaT();
  }

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
