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

#ifndef __MESSAGEINTERFACE_HPP
#define __MESSAGEINTERFACE_HPP

#include "Message.hpp"

/**
  Common base class for classes which receive Messages.

  NOTE:  Subclasses of MessageInterface MUST call their own makeSlots(),
  if any, to make their slots in their constructors.
  (virtual functions won't work in constructors)

  @see Message
  @see MessageCallback
*/
class MessageInterface
{
public:  

  typedef map< const String, AbstractMessageCallback* > SlotMap;
  typedef SlotMap::iterator SlotMapIterator;

  // exceptions

  class NoSuchSlot : public Message::MessageException
    {
    public: 
      NoSuchSlot( StringCref method, StringCref what )
	: MessageException( method, what ){}
      const String what() const { return "No appropriate slot found"; }
    };

public:

  MessageInterface();

  virtual ~MessageInterface();

  void set( MessageCref ) throw( NoSuchSlot );
  const Message get( StringCref ) throw( NoSuchSlot );
  StringList slotList();

  virtual void makeSlots() = 0;

  virtual const char* const className() const { return "MessageInterface"; }

protected:

  void appendSlot( StringCref keyword, AbstractMessageCallback* );
  void deleteSlot( StringCref keyword );

private:

  SlotMap theSlotMap;

};


#define MessageSlot( KEY, CLASS, OBJ, SETMETHOD, GETMETHOD )\
appendSlot( KEY, new MessageCallback< CLASS >\
	   ( OBJ, static_cast< MessageCallback< CLASS >::SetMessageFunc >\
	    ( SETMETHOD ),\
	    static_cast< MessageCallback< CLASS >::GetMessageFunc >\
	    ( GETMETHOD ) ) )

#endif /* __MESSAGEINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
