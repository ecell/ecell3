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
// modified by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "PropertySlotMaker.hpp"

#include "PropertyInterface.hpp"

namespace libecs
{


  ///////////////////////////// PropertyInterface

  PropertySlotMakerPtr PropertyInterface::getPropertySlotMaker()
  {
    static PropertySlotMaker aPropertySlotMaker;

    return &aPropertySlotMaker;
  }

  const Polymorph PropertyInterface::getPropertyList() const
  {
    PolymorphVector aVector;
    aVector.reserve( thePropertySlotMap.size() );

    for( PropertySlotMapConstIterator i( thePropertySlotMap.begin() ); 
	 i != thePropertySlotMap.end() ; ++i )
      {
	aVector.push_back( i->first );
      }

    return aVector;
  }

  const Polymorph PropertyInterface::
  getPropertyAttributes( StringCref aPropertyName ) const  
  {
    PropertySlotPtr aPropertySlotPtr( getPropertySlot( aPropertyName ) );

    PolymorphVector aVector;

    // is setable?
    aVector.push_back( static_cast<Int>( aPropertySlotPtr->isSetable() ) );

    // is getable?
    aVector.push_back( static_cast<Int>( aPropertySlotPtr->isGetable() ) );

    return aVector;
  }


  PropertyInterface::PropertyInterface()
  {
    ; // do nothing
  }

  PropertyInterface::~PropertyInterface()
  {
    for( PropertySlotMapIterator i( thePropertySlotMap.begin() ); 
	 i != thePropertySlotMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void PropertyInterface::registerSlot( StringCref aName,
					PropertySlotPtr aPropertySlotPtr )
  {
    if( thePropertySlotMap.find( aName ) != thePropertySlotMap.end() )
      {
	// it already exists. take the latter one.
	delete thePropertySlotMap[ aName ];
	thePropertySlotMap.erase( aName );
      }

    thePropertySlotMap[ aName ] = aPropertySlotPtr;
  }

  void PropertyInterface::removeSlot( StringCref aName )
  {
    if( thePropertySlotMap.find( aName ) == thePropertySlotMap.end() )
      {
	THROW_EXCEPTION( NoSlot,
			 getClassName() + String( ":no slot for keyword [" ) +
			 aName + String( "] found.\n" ) );
      }

    delete thePropertySlotMap[ aName ];
    thePropertySlotMap.erase( aName );
  }

  void PropertyInterface::setProperty( StringCref aPropertyName, 
				       PolymorphCref aValue )
  {
    PropertySlotMapConstIterator 
      aPropertySlotMapIterator( thePropertySlotMap.find( aPropertyName ) );

    if( aPropertySlotMapIterator == thePropertySlotMap.end() )
      {
	THROW_EXCEPTION( NoSlot,
			 getClassName() + 
			 String(": No property slot found with name [")
			 + aPropertyName + "].  Set property failed." );
      }

    aPropertySlotMapIterator->second->setPolymorph( aValue );
  }

  const Polymorph
  PropertyInterface::getProperty( StringCref aPropertyName ) const
  {
    PropertySlotMapConstIterator 
      aPropertySlotMapIterator( thePropertySlotMap.find( aPropertyName ) );

    if( aPropertySlotMapIterator == thePropertySlotMap.end() )
      {
	THROW_EXCEPTION( NoSlot, 
			 getClassName() + 
			 String(": No property slot found with name [")
			 + aPropertyName + "].  Get property failed." );
      }

    return aPropertySlotMapIterator->second->getPolymorph();
  }

  /*
  void PropertyInterface::connectLogger( LoggerPtr aLoggerPtr )
  {
    theLoggerVector.push_back( aLoggerPtr );
  }

  void PropertyInterface::disconnectLogger( LoggerPtr aLoggerPtr )
  {
    theLoggerVector.erase( aLoggerPtr );
  }
  */


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
