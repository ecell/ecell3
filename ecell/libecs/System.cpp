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

#include "System.h"
#include "Reactor.h"
#include "CellComponents.h"
#include "RootSystem.h"
#include "Stepper.h"
#include "StepperMaker.h"
#include "FQPN.h"

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
  theStepper( NULL ) 
{
  makeSlots();
  theFirstRegularReactorIterator = getFirstReactorIterator();
}

System::~System()
{
  delete theStepper;
  delete theVolumeIndexName;
}

const String System::getFqpn() const
{
  return Primitive::PrimitiveTypeString( Primitive::SYSTEM ) + ":" + getFqin();
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
  setVolumeIndex( FQIN( message.getBody() ) );
}

const Message System::getVolumeIndex( StringCref keyword )
{
  if( !getVolumeIndex() )
    return Message( keyword, "" );

  return Message( keyword, getVolumeIndex()->getFqin() );
}

void System::setStepper( StringCref classname )
{
  StepperPtr aStepper;
  aStepper = theRootSystem->stepperMaker().make( classname );
  aStepper->setOwner( this );

  theStepper = aStepper;
}

Float System::getVolume() 
{
  return theVolumeIndex->getActivityPerSecond();
}

void System::setVolumeIndex( FQINCref volumeindex )
{
  theVolumeIndexName = new FQIN( volumeindex );
}

Primitive System::getPrimitive( StringCref id, Primitive::Type type )
throw( InvalidPrimitiveType, NotFound )
{
  Primitive aPrimitive;

  aPrimitive.type = type;

  switch(type)
    {
    case Primitive::SUBSTANCE:
      aPrimitive.substance = getSubstance( id );
      break;
    case Primitive::REACTOR:
      aPrimitive.reactor = getReactor( id );
      break;
    case Primitive::SYSTEM:
      aPrimitive.system = getSystem( id );
      break;
    case Primitive::NONE:
    default:
	throw InvalidPrimitiveType(__PRETTY_FUNCTION__,"[" 
				   + getFqin() + "]: request type invalid.");
    }

  return aPrimitive;
}

int System::getNumberOfPrimitives( Primitive::Type type )
{
  int aNumber( 0 );
  switch( type )
    {
    case Primitive::SUBSTANCE:
      aNumber = getNumberOfSubstances();
      break;
    case Primitive::REACTOR:
      aNumber = getNumberOfReactors();
      break;
    case Primitive::SYSTEM:
      aNumber = getNumberOfSystems();
      break;
    case Primitive::NONE:
    default:
	throw InvalidPrimitiveType(__PRETTY_FUNCTION__,"[" 
				   + getFqin() + "]: request type invalid");
    }
  return aNumber;
}

void System::forAllPrimitives( Primitive::Type type, PrimitiveCallback cb,
			       void* clientData )
{
  assert( cb );

  SubstanceListIterator si( NULL );
  ReactorListIterator ri( NULL );
  SystemListIterator yi( NULL );

  PrimitivePtr aPrimitivePtr;

  switch(type)
    {
    case Primitive::SUBSTANCE:
      for( si = getFirstSubstanceIterator(); 
	   si != getLastSubstanceIterator(); ++si )
	{
	  auto_ptr< Primitive > aPrimitivePtr( new Primitive( si->second ) );
	  cb( aPrimitivePtr.get(), clientData );
	}
      break;
    case Primitive::REACTOR:
      for( ri = getFirstReactorIterator(); 
	   ri != getLastReactorIterator(); ++ri )
	{
	  auto_ptr< Primitive > aPrimitivePtr( new Primitive( ri->second ) );
	  cb( aPrimitivePtr.get(), clientData );
	}
      break;
    case Primitive::SYSTEM:
      for( yi = getFirstSystemIterator(); 
	   yi != getLastSystemIterator(); ++yi )
	{
	  auto_ptr< Primitive > aPrimitivePtr( new Primitive( yi->second ) );
	  cb( aPrimitivePtr.get(), clientData );
	}
      break;
    case Primitive::NONE:
    default:
	throw InvalidPrimitiveType( __PRETTY_FUNCTION__,"[" 
				    + getFqin() + "]: request type invalid" );
    }
} 

void System::initialize()
{
  assert(theStepper);

  try{
    if( theVolumeIndexName != NULL )
      {
	FQPN fqpn( Primitive::REACTOR, *theVolumeIndexName );
	Primitive aPrimitive( theRootSystem->getPrimitive( fqpn ) );
	theVolumeIndex = aPrimitive.reactor;
	//FIXME: *theMessageWindow << getFqin() << ": volume index is [" 
	//FIXME: 	  << _volumeIndex->getFqin() << "].\n";

      }
    else
      {
	//FIXME: *theMessageWindow << getFqin() << ": no volume index is specified.\n"; 
      }
  }
  catch( NotFound )
    {
      //FIXME: *theMessageWindow << getFqin() << ": volume index [" 
	//FIXME: << _volumeIndexName->fqinString() << "] not found.\n";
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
      //FIXME: *theMessageWindow << "multiple definition of reactor [" 
	//FIXME: << reactor->getId() << "] on [" << getId() << 
	  //FIXME: "], later one discarded.\n";

      //FIXME: throw exception
      return;
    }

  theReactorList[ reactor->getId() ] = reactor;
  return;
}

ReactorPtr System::getReactor( StringCref id ) throw( NotFound )
{
  ReactorListIterator i = getReactorIterator( id );
  if( i == getLastReactorIterator() )
    {
      throw NotFound( __PRETTY_FUNCTION__, "[" + getFqin() + 
		      "]: Reactor [" + id + "] not found in this System." );
    }
  return i->second;
}

void System::addSubstance( SubstancePtr newone )
{
  if( containsSubstance( newone->getId() ) )
    {
//FIXME:       *theMessageWindow << "multiple definition of substance [" 
//FIXME: 	<< newone->getId() << "] on [" << getId() << 
//FIXME: 	  "], name and quantity overwrote.\n";
      theSubstanceList[ newone->getId() ]->setName( newone->getName() );
      Message aMessage( "Quantity", newone->getQuantity() );
      theSubstanceList[ newone->getId() ]->set(aMessage);
      delete newone;
      //FIXME: throw exception
      return;
    }
  theSubstanceList[ newone->getId() ] = newone;
  newone->setSupersystem( this );

}

SubstancePtr System::getSubstance( StringCref id ) throw( NotFound )
{
  SubstanceListIterator i = getSubstanceIterator( id );
  if( i == getLastSubstanceIterator() )
    {
      throw NotFound(__PRETTY_FUNCTION__, "[" + getFqin() + 
		     "]: Substance [" + id + "] not found in this System.");
    }

  return i->second;
}


void System::addSystem( SystemPtr system )
{
  if( containsSystem( system->getId() ) )
    {
//FIXME:       *theMessageWindow << "multiple definition of system [" 
//FIXME: 	<< system->getId() << "] on [" << getId() << 
//FIXME: 	  "], later one discarded.\n";
      delete system;
      //FIXME: throw exception
      return;
    }

  theSubsystemList[ system->getId() ] = system;

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
      throw NotFound(__PRETTY_FUNCTION__, "[" + getFqin() + 
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
