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

#include "MessageInterface.hpp"

///////////////////////////// MessageInterface

MessageInterface::MessageInterface()
{
  ; // do nothing
}

MessageInterface::~MessageInterface()
{
  for( SlotMapIterator i = theSlotMap.begin() ; i != theSlotMap.end() ; ++i )
    {
      delete i->second;
    }
}

void MessageInterface::appendSlot( StringCref keyword, 
				   AbstractMessageCallback* func )
{
  if( theSlotMap.find( keyword ) != theSlotMap.end() )
    {
      //      *theMessageWindow << "MessageSlot: appendSlot(): slot for keyword [" 
      //	<< keyword << "] already exists. Taking later one.\n";
      delete theSlotMap[ keyword ];
      theSlotMap.erase( keyword );
    }
  theSlotMap[ keyword ] = func;
}

void MessageInterface::deleteSlot( StringCref keyword )
{
  if( theSlotMap.find( keyword ) == theSlotMap.end() )
    {
      //      *theMessageWindow << "MessageSlot: deleteSlot(): no slot for keyword [" 
      //	<< keyword << "] found.\n";
      return;
    }
  delete theSlotMap[ keyword ];
  theSlotMap.erase( keyword );
}

void MessageInterface::set( MessageCref message ) throw( NoSuchSlot )
{
  SlotMapIterator sm( theSlotMap.find( message.getKeyword() ) );

  if( sm == theSlotMap.end() )
    {
      throw NoSuchSlot(__PRETTY_FUNCTION__,
		       className() + String(": got a Message (keyword = [")
		       + message.getKeyword() + "]) but no slot for it.");
    }

  try {
    sm->second->set( message );
  }
  catch( ExceptionCref e )
    {
      //      *theMessageWindow << className() << ": Callback has failed (keyword = [" 
      //	<< message.getKeyword() << "]):\n\t" << e.message() << "\n";
      return;
    }
  catch( ... )
    {
      //      *theMessageWindow << __PRETTY_FUNCTION__ << ": " 
      //<< "callback has failed.(keyword = [" << message.getKeyword() << "])\n";
      return;
    }

}

const Message MessageInterface::get( StringCref keyword ) throw( NoSuchSlot )
{
  SlotMapIterator sm( theSlotMap.find( keyword ) );

  if( sm == theSlotMap.end() )
    {
      throw NoSuchSlot( __PRETTY_FUNCTION__, className()
			+ String( ": got a request for Message (keyword = [" )
			+ keyword + "] but no slot for it.\n" );
    }

  try {
    return sm->second->get( keyword );
  }
  catch(ExceptionCref e)
    {
      //      *theMessageWindow << className() << ": Callback has failed (keyword = [" 
      //	<< keyword << "]):\n\t" << e.message() << "\n";
      return Message( keyword, "" );
    }
  catch(...)
    {
      //      *theMessageWindow << __PRETTY_FUNCTION__ << ": " 
      //	<< "callback has failed.(keyword = [" << keyword << "])\n";
      return Message( keyword, "" );
    }
}

StringList MessageInterface::slotList()
{
  StringList sl;

  for( SlotMapIterator i = theSlotMap.begin() ; i != theSlotMap.end() ; ++i )
    {
      sl.insert( sl.end(), i->first );
    }

  return sl;
}

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
