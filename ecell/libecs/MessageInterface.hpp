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

#include <map>

#include "libecs.hpp"

#include "Message.hpp"

namespace libecs
{

  DECLARE_CLASS( AbstractMessageCallback );

  DECLARE_MAP( const String, AbstractMessageCallbackPtr, 
	       less<const String>, PropertyMap );


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
      theGetMethod( getmethod ),
      theLogger( 0 )
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

    virtual void setLogger( LoggerRef logger )
    {
      theLogger = &logger;
    }

  private:

    T& theObject;
    const SetMessageFunc theSetMethod;
    const GetMessageFunc theGetMethod;
    LoggerPtr theLogger;
  };

  /**
     Common base class for classes which receive Messages.

     NOTE:  Subclasses of MessageInterface MUST call their own makeSlots(),
     if any, to make their slots in their constructors.
     (virtual functions won't work in constructors)

     FIXME: class-static slots?

     @see Message
     @see MessageCallback
  */

  class MessageInterface
  {
  public:

    MessageInterface();
    virtual ~MessageInterface();

    void set( MessageCref );
    const Message get( StringCref );

    PropertyMapIterator getMessageCallback( StringCref property )
    {
      return thePropertyMap.find( property );
    }

    virtual void makeSlots();

    virtual const char* const className() const { return "MessageInterface"; }

  public: // message slots

    const Message getPropertyList( StringCref keyword );

  protected:

    void appendSlot( StringCref keyword, AbstractMessageCallbackPtr );
    void deleteSlot( StringCref keyword );

  private:

    PropertyMap thePropertyMap;

  };


#define MessageSlot( KEY, CLASS, OBJ, SETMETHOD, GETMETHOD )\
appendSlot( KEY, new MessageCallback< CLASS >\
	   ( OBJ, static_cast< MessageCallback< CLASS >::SetMessageFunc >\
	    ( SETMETHOD ),\
	    static_cast< MessageCallback< CLASS >::GetMessageFunc >\
	    ( GETMETHOD ) ) )


} // namespace libecs

#endif /* __MESSAGEINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
