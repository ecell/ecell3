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

#include "libecs.hpp"
#include "Defs.hpp" 

#include "Exceptions.hpp"
#include "UniversalVariable.hpp"

#include <vector>
DECLARE_VECTOR( UniversalVariable, UniversalVariableVector );


  /**
     A data packet for communication among C++ objects consisting
     of a keyword and a body. The body is a list of UniversalVariables.

     @see MessageInterface
     @see AbstractMessageSlot
   */
class Message
{

public:

  Message( StringCref keyword )
    :
    theKeyword( keyword )
  {
    ; // do nothing
  }

  Message( StringCref keyword, UniversalVariableVectorCref uvl ); 
  Message( StringCref keyword, UniversalVariableCref uv ); 

  // copy procedures
  Message( MessageCref message );
  MessageRef operator=( MessageCref );
  
  virtual ~Message();

  /**
    Returns keyword string of this Message.

    @return keyword string.
    @see body()
   */
  StringCref getKeyword() const { return theKeyword; }

  /**
    @return 
   */
  UniversalVariableVectorRef getBody()
  {
    return theBody;
  }

  UniversalVariableVectorCref getBody() const
  {
    return theBody;
  }

  UniversalVariableRef operator[]( int i )
  {
    return theBody.operator[]( i );
  }

  UniversalVariableCref operator[]( int i ) const
  {
    return theBody.operator[]( i );
  }

private:

  String theKeyword;
  UniversalVariableVector theBody;

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
	  return Message( keyword );
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
