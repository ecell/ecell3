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

#include "PropertyInterface.hpp"


namespace libecs
{


  ///////////////////////////// PropertyInterface

  void PropertyInterface::makeSlots()
  {

    //    appendSlot( new PropertySlot<typeof(*this),String>("ClassName", *this, NULLPTR, &PropertyInterface::getClassNameString ) );

    //    appendSlot( new PropertySlot<typeof(*this),
    //		UVariableVectorRCPtr>( "PropertyAttributes",*this,NULLPTR,
    //				       &PropertyInterface::getPropertyAttributes) );


    createPropertySlot( "ClassName", *this, NULLPTR, 
    			&PropertyInterface::getClassNameString );
    createPropertySlot( "PropertyList",*this,NULLPTR,
			&PropertyInterface::getPropertyList );
    createPropertySlot( "PropertyAttributes",*this,NULLPTR,
			&PropertyInterface::getPropertyAttributes);

  }

  const UVariableVectorRCPtr PropertyInterface::getPropertyList() const
  {
    UVariableVectorRCPtr aPropertyVectorPtr( new UVariableVector );
    aPropertyVectorPtr->reserve( thePropertySlotMap.size() );

    for( PropertySlotMapConstIterator i( thePropertySlotMap.begin() ); 
	 i != thePropertySlotMap.end() ; ++i )
      {
	aPropertyVectorPtr->push_back( i->first );
      }

    return aPropertyVectorPtr;
  }

  const UVariableVectorRCPtr PropertyInterface::getPropertyAttributes() const
  {
    UVariableVectorRCPtr aPropertyAttributesVector( new UVariableVector );
    aPropertyAttributesVector->reserve( thePropertySlotMap.size() );

    for( PropertySlotMapConstIterator i( thePropertySlotMap.begin() ); 
	 i != thePropertySlotMap.end() ; ++i )
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

	aPropertyAttributesVector->push_back( anAttributeFlag );
      }

    return aPropertyAttributesVector;
  }

  PropertyInterface::PropertyInterface()
  {
    makeSlots();
  }

  PropertyInterface::~PropertyInterface()
  {
    for( PropertySlotMapIterator i( thePropertySlotMap.begin() ); 
	 i != thePropertySlotMap.end() ; ++i )
      {
	delete i->second;
      }
  }

  void PropertyInterface::appendSlot( PropertySlotPtr slot )
  {
    String keyword = slot->getName();
    if( thePropertySlotMap.find( keyword ) != thePropertySlotMap.end() )
      {
	// it already exists. take the latter one.
	delete thePropertySlotMap[ keyword ];
	thePropertySlotMap.erase( keyword );
      }

    thePropertySlotMap[ keyword ] = slot;
  }

  void PropertyInterface::deleteSlot( StringCref keyword )
  {
    if( thePropertySlotMap.find( keyword ) == thePropertySlotMap.end() )
      {
	THROW_EXCEPTION( NoSlot,
			 getClassName() + String( ":no slot for keyword [" ) +
			 keyword + String( "] found.\n" ) );
      }

    delete thePropertySlotMap[ keyword ];
    thePropertySlotMap.erase( keyword );
  }

  void PropertyInterface::setMessage( MessageCref message ) 
  {
    PropertySlotMapConstIterator sm( thePropertySlotMap.find( message.getKeyword() ) );

    if( sm == thePropertySlotMap.end() )
      {
	THROW_EXCEPTION( NoSlot,
			 getClassName() + 
			 String(": got a Message (keyword = [")
			 + message.getKeyword() + "]) but no slot for it.");
      }

    sm->second->setUVariableVectorRCPtr( message.getBody() );
  }

  const Message PropertyInterface::getMessage( StringCref keyword ) const
  {
    PropertySlotMapConstIterator sm( thePropertySlotMap.find( keyword ) );

    if( sm == thePropertySlotMap.end() )
      {
	THROW_EXCEPTION( NoSlot, 
			 getClassName()
			 + String( ": got a request for Message (keyword = [" )
			 + keyword + "]) but no slot for it.\n" );
      }

    return 
      Message( keyword, 
	       UVariableVectorRCPtr( sm->second->getUVariableVectorRCPtr() ) );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
