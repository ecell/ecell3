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

// instantiate primitive lists.
//template SubstanceList;
//template ReactorList;
//template SystemList;

/////////////////////// System

void System::makeSlots()
{
  MessageSlot( "SystemList", System, *this,
	       NULLPTR, &System::getSystemList );
  MessageSlot( "SubstanceList", System, *this,
	       NULLPTR, &System::getSubstanceList );
  MessageSlot( "ReactorList", System, *this,
	       NULLPTR, &System::getReactorList );

  MessageSlot( "StepperClass", System, *this,
	       &System::setStepperClass, &System::getStepperClass );
  MessageSlot( "VolumeIndex", System, *this,
	       &System::setVolumeIndex, &System::getVolumeIndex );
}


// Message slots

const Message System::getSystemList( StringCref keyword )
{
  UniversalVariableVector aVector;

  for( SystemListIterator i = getFirstSystemIterator() ;
       i != getLastSystemIterator() ; ++i )
    {
      aVector.push_back( UniversalVariable( i->second->getId() ) );
    }

  return Message( keyword, aVector );
}

const Message System::getSubstanceList( StringCref keyword )
{
  UniversalVariableVector aVector;

  for( SubstanceListIterator i = getFirstSubstanceIterator() ;
       i != getLastSubstanceIterator() ; ++i )
    {
      aVector.push_back( UniversalVariable( i->second->getId() ) );
    }

  return Message( keyword, aVector );
}

const Message System::getReactorList( StringCref keyword )
{
  UniversalVariableVector aVector;

  for( ReactorListIterator i = getFirstReactorIterator() ;
       i != getLastReactorIterator() ; ++i )
    {
      aVector.push_back( UniversalVariable( i->second->getId() ) );
    }

  return Message( keyword, aVector );
}


void System::setStepperClass( const Message& message )
{
  //FIXME: range check
  setStepper( message[0].asString() );
}

const Message System::getStepperClass( StringCref keyword )
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

  return Message( keyword, UniversalVariable( getVolumeIndex()->getFqid() ) );
}




System::System()
  :
  theVolumeIndexName( NULLPTR ),
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
  delete theVolumeIndexName;
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

  theStepper = aStepper;
}

Real System::getVolume() 
{
  return theVolumeIndex->getActivityPerSecond();
}

void System::setVolumeIndex( FQIDCref volumeindex )
{
  theVolumeIndexName = new FQID( volumeindex );
}

void System::initialize()
{
  if( theStepper == NULLPTR )
    {
      //FIXME: make this default user customizable
      setStepper( "Euler1Stepper" );
    }

  try{
    if( theVolumeIndexName != NULLPTR )
      {
	FQID fqid( *theVolumeIndexName );
	theVolumeIndex = theRootSystem->getReactor( fqid );
      }
  }
  catch( NotFound )
    {
      //FIXME: what to do in this case?
    }

  delete theVolumeIndexName;
  theVolumeIndexName = NULLPTR;

  //
  // Substance::initialize()
  //
  for( SubstanceListIterator i = getFirstSubstanceIterator() ; 
       i != getLastSubstanceIterator() ; ++i )
    {
      i->second->initialize();
    }

  //
  // Reactor::initialize()
  //
  for( ReactorListIterator i = getFirstReactorIterator() ;
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
  for( SystemListIterator i = getFirstSystemIterator();
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
  for( SubstanceListIterator i = getFirstSubstanceIterator() ; 
      i != getLastSubstanceIterator() ; ++i )
    {
      i->second->clear();
    }
}

void System::react()
{
  for( ReactorListIterator i = getFirstRegularReactorIterator() ; 
       i != getLastReactorIterator() ; ++i )
    {
      i->second->react();
    }
}

void System::turn()
{
  for( SubstanceListIterator i = getFirstSubstanceIterator() ; 
       i != getLastSubstanceIterator() ; ++i )
    {
      i->second->turn();
    }
}

void System::transit()
{
  for( ReactorListIterator i = getFirstRegularReactorIterator() ; 
       i != getLastReactorIterator() ; ++i )
    {
      i->second->transit();
    }

  for( SubstanceListIterator i = getFirstSubstanceIterator() ;
       i != getLastSubstanceIterator() ; ++i )
    {
      i->second->transit();
    }
}

void System::postern()
{
  for( ReactorListIterator i = getFirstReactorIterator() ; 
       i != getFirstRegularReactorIterator() ; ++i )
    {
      i->second->react();
    }

  // update activity of posterior reactors by buffered values 
  for( ReactorListIterator i = getFirstReactorIterator() ; 
       i != getFirstRegularReactorIterator() ; ++i )
    {
      i->second->transit();
    }
}

void System::addReactor( ReactorPtr reactor )
{
  assert(reactor);

  if( containsReactor( reactor->getId() ) )
    {
      delete reactor;
      //FIXME: throw exception
      return;
    }

  theReactorList[ reactor->getId() ] = reactor;
  reactor->setSuperSystem( this );
}

ReactorPtr System::getReactor( StringCref id ) 
{
  ReactorListIterator i = getReactorIterator( id );
  if( i == getLastReactorIterator() )
    {
      throw NotFound( __PRETTY_FUNCTION__, "[" + getFqid() + 
		      "]: Reactor [" + id + "] not found in this System." );
    }
  return i->second;
}

void System::addSubstance( SubstancePtr newone )
{
  if( containsSubstance( newone->getId() ) )
    {
      delete newone;
      //FIXME: throw exception
      return;
    }
  theSubstanceList[ newone->getId() ] = newone;
  newone->setSuperSystem( this );
}

SubstancePtr System::getSubstance( StringCref id ) 
{
  SubstanceListIterator i = getSubstanceIterator( id );
  if( i == getLastSubstanceIterator() )
    {
      throw NotFound(__PRETTY_FUNCTION__, "[" + getFqid() + 
		     "]: Substance [" + id + "] not found in this System.");
    }

  return i->second;
}


void System::addSystem( SystemPtr system )
{
  if( containsSystem( system->getId() ) )
    {
      delete system;
      //FIXME: throw exception
      return;
    }

  theSubsystemList[ system->getId() ] = system;
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
  SystemListIterator i = getSystemIterator( id );
  if( i == getLastSystemIterator() )
    {
      throw NotFound(__PRETTY_FUNCTION__, "[" + getFqid() + 
		     "]: System [" + id + "] not found in this System.");
    }
  return i->second;
}

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
