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

#include <memory>

#include "System.hpp"
#include "Reactor.hpp"
#include "RootSystem.hpp"
#include "Substance.hpp"
#include "Stepper.hpp"
#include "StepperMaker.hpp"
#include "FQPI.hpp"

// instantiate primitive lists.
template SubstanceList;
template ReactorList;
template SystemList;

/////////////////////// System

void System::makeSlots()
{
  MessageSlot( "Stepper", System, *this,
	       &System::setStepper, &System::getStepper );
  MessageSlot( "VolumeIndex", System, *this,
	       &System::setVolumeIndex, &System::getVolumeIndex );
}

System::System()
  :
  theVolumeIndexName( NULL ),
  theVolumeIndex( NULL ),
  theStepper( NULL ),
  theRootSystem( NULL )
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
  setSuperSystem( supersystem );
  theRootSystem = getSuperSystem()->getRootSystem();
}

const String System::getFqpi() const
{
  return PrimitiveTypeStringOf( *this ) + ":" + getFqid();
}

void System::setStepper( const Message& message )
{
  setStepper( message.getBody() );
}

const Message System::getStepper( StringCref keyword )
{
  return Message( keyword, getStepper()->className() );
}

void System::setVolumeIndex( const Message& message )
{
  setVolumeIndex( FQID( message.getBody() ) );
}

const Message System::getVolumeIndex( StringCref keyword )
{
  if( !getVolumeIndex() )
    return Message( keyword, "" );

  return Message( keyword, getVolumeIndex()->getFqid() );
}

void System::setStepper( StringCref classname )
{
  StepperPtr aStepper;
  aStepper = getRootSystem()->getStepperMaker().make( classname );
  aStepper->setOwner( this );

  theStepper = aStepper;
}

Float System::getVolume() 
{
  return theVolumeIndex->getActivityPerSecond();
}

void System::setVolumeIndex( FQIDCref volumeindex )
{
  theVolumeIndexName = new FQID( volumeindex );
}

void System::initialize()
{
  if( theStepper == NULL )
    {
      //FIXME: make this default user customizable
      setStepper( "Eular1Stepper" );
    }

  try{
    if( theVolumeIndexName != NULL )
      {
	FQID fqid( *theVolumeIndexName );
	//FIXME: recursive search needed
	theVolumeIndex = theRootSystem->getReactor( fqid );
	//FIXME: *theMessageWindow << getFqid() << ": volume index is [" 
	//FIXME: 	  << _volumeIndex->getFqid() << "].\n";

      }
    else
      {
	//FIXME: *theMessageWindow << getFqid() << ": no volume index is specified.\n"; 
      }
  }
  catch( NotFound )
    {
      //FIXME: *theMessageWindow << getFqid() << ": volume index [" 
	//FIXME: << _volumeIndexName->fqidString() << "] not found.\n";
    }

  delete theVolumeIndexName;
  theVolumeIndexName = NULL;

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

ReactorPtr System::getReactor( StringCref id ) throw( NotFound )
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

SubstancePtr System::getSubstance( StringCref id ) throw( NotFound )
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

SystemPtr System::getSystem( SystemPathCref systempath ) throw( NotFound )
{
  SystemPtr  aSystem = getSystem( systempath.first() );
  SystemPath anNext    = systempath.next();

  if( anNext.getString() != "" ) // not a leaf
    {
      aSystem = aSystem->getSystem( anNext );
    }
  
  return aSystem;
}

SystemPtr System::getSystem( StringCref id ) throw(NotFound)
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
