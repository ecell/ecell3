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
// modified by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#include "PropertyInterface.hpp"


namespace libecs
{

  ///////////////////////////// PropertyInterface

  void PropertyInterface::makeSlots()
  {
    makePropertySlot( "PropertyList",PropertyInterface,*this,NULLPTR,
		     &PropertyInterface::getPropertyList);
    makePropertySlot( "PropertyAttributes",PropertyInterface,*this,NULLPTR,
		     &PropertyInterface::getPropertyAttributes);

  }

  const Message PropertyInterface::getPropertyList( StringCref keyword )
  {
    UConstantVector aPropertyList;

    for( PropertyMapConstIterator i = thePropertyMap.begin() ; 
	 i != thePropertyMap.end() ; ++i )
      {
	aPropertyList.push_back( UConstant( i->first ) );
      }

    return Message( keyword, aPropertyList );
  }

  const Message PropertyInterface::getPropertyAttributes( StringCref keyword )
  {
    UConstantVector aPropertyList;

    for( PropertyMapConstIterator i = thePropertyMap.begin() ; 
	 i != thePropertyMap.end() ; ++i )
      {
	Int anAttributeFlag( 0 );

	if( i->second->isSetable() )
	  {
	    anAttributeFlag |= SETABLE;
	  }

	if( i->second->isGetable() )
	  {
	    anAttributeFlag |= GETABLE;
	  }

	aPropertyList.push_back( UConstant( anAttributeFlag ) );
      }

    return Message( keyword, aPropertyList );
  }

  PropertyInterface::PropertyInterface()
  {
    makeSlots();
  }

  PropertyInterface::~PropertyInterface()
  {
    for( PropertyMapIterator i = thePropertyMap.begin() ; 
	 i != thePropertyMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void PropertyInterface::appendSlot( StringCref keyword, 
				     AbstractPropertySlot* func )
  {
    if( thePropertyMap.find( keyword ) != thePropertyMap.end() )
      {
	// it already exists. take the latter one.
	delete thePropertyMap[ keyword ];
	thePropertyMap.erase( keyword );
      }
    thePropertyMap[ keyword ] = func;
  }

  void PropertyInterface::deleteSlot( StringCref keyword )
  {
    if( thePropertyMap.find( keyword ) == thePropertyMap.end() )
      {
	throw NoSlot( __PRETTY_FUNCTION__,
		      className() + String( ":no slot for keyword [" ) +
		      keyword + String( "] found.\n" ) );
      }
    delete thePropertyMap[ keyword ];
    thePropertyMap.erase( keyword );
  }

  void PropertyInterface::set( MessageCref message ) 
  {
    PropertyMapIterator sm( thePropertyMap.find( message.getKeyword() ) );

    if( sm == thePropertyMap.end() )
      {
	throw NoSlot( __PRETTY_FUNCTION__,
		      className() + String(": got a Message (keyword = [")
		      + message.getKeyword() + "]) but no slot for it.");
      }

    sm->second->set( message );
  }

  const Message PropertyInterface::get( StringCref keyword ) 
  {
    PropertyMapIterator sm( thePropertyMap.find( keyword ) );

    if( sm == thePropertyMap.end() )
      {
	throw NoSlot( __PRETTY_FUNCTION__, className()
		      + String( ": got a request for Message (keyword = [" )
		      + keyword + "]) but no slot for it.\n" );
      }

    return sm->second->get( keyword );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
