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

#ifndef __MESSAGE_HPP
#define __MESSAGE_HPP
#include <string>
#include <map>
#include <vector>
#include "libecs.hpp"
#include "Exceptions.hpp"
#include "StringList.hpp"
#include "Defs.hpp"

DECLARE_CLASS( Message )
DECLARE_CLASS( AbstractMessageSlot )
DECLARE_CLASS( MessageSlot )
DECLARE_CLASS( MessageInterface )

  /**
     A string data packet for communication among C++ objects.

     @see MessageInterface
     @see AbstractMessageSlot
   */
class Message : private StringPair
{

public:

  Message( StringCref keyword, StringCref body ); 
  Message( StringCref keyword, const Float f );
  Message( StringCref keyword, const Int i );
  Message( StringCref message ); 

  // copy procedures
  Message( MessageCref message );
  Message& operator=( MessageCref );
  
  virtual ~Message();

  /**
    Returns keyword string of this Message.

    @return keyword string.
    @see body()
   */
  StringCref getKeyword() const { return first; }

  /**
    Returns body string of this Message.

    @return body string.
    @see keyword()
   */
  StringCref getBody() const { return second; }

  /**
    Returns nth field of body string using FIELD_SEPARATOR as delimiter.  

    @return nth field of body string.
    @see FIELD_SEPARATOR
   */
  const String getBody( int n ) const;

  /**
     return keyword + ' ' + body for debug.
   */
  const String dump() const { return first + ' ' + second; }


};


/**
   A base class for MessageCallback class.

   @see MessageCallback
   @see MessageInterface
   @see Message
 */
class AbstractMessageCallback
{

public:

  virtual void set( MessageCref message ) = 0;
  virtual const Message get( StringCref keyword ) = 0;

  virtual void operator()( MessageCref message ) 
    { set( message ); }
  virtual const Message operator()( StringCref keyword ) 
    { return get(keyword); }
};


/**
   Calls callback methods for getting and sending Message objects.

   @see Message
   @see MessageInterface
   @see AbstractMessageCallback
 */
template <class T>
class MessageCallback : public AbstractMessageCallback
{

public:

  typedef void ( T::* SetMessageFunc )( MessageCref );
  typedef const Message ( T::* GetMessageFunc )( StringCref );

public:

  MessageCallback( T& object, const SetMessageFunc setmethod,
		   const GetMessageFunc getmethod )
    : 
    theObject( object ), 
    theSetMethod( setmethod ), 
    theGetMethod( getmethod ) 
    {
      ; // do nothing
    }
  
  virtual void set( MessageCref message ) 
    {
      if( theSetMethod == NULLPTR )
	{
	  //FIXME: throw an exception
	  return;
	}
      ( theObject.*theSetMethod )( message );
    }

  virtual const Message get( StringCref keyword ) 
    {
      if( theGetMethod == NULLPTR )
	{
	  //FIXME: throw an exception
	  return Message( keyword, "" );
	}
      return ( ( theObject.*theGetMethod )( keyword ));
    }

private:

  T& theObject;
  const SetMessageFunc theSetMethod;
  const GetMessageFunc theGetMethod;

};


#endif /* ___MESSAGE_H___*/


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
