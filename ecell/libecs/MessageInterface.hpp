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
#include "Logger.hpp"
#include "Message.hpp"


namespace libecs
{

  DECLARE_CLASS( AbstractMessageSlot );

  DECLARE_MAP( const String, AbstractMessageSlotPtr, 
	       less<const String>, PropertyMap );

  class Logger;
  class ProxyMessageSlot;


  /**
     A base class for MessageSlot class.

     @see MessageSlot
     @see MessageInterface
     @see Message
  */
  class AbstractMessageSlot
  {

  public:

    virtual void set( MessageCref message ) = 0;
    virtual const Message get( StringCref keyword ) = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    virtual void operator()( MessageCref message ) 
    { set( message ); }
    virtual const Message operator()( StringCref keyword ) 
    { return get(keyword); }

    virtual ProxyMessageSlot* getProxy( void ) = 0;

  };


  class ProxyMessageSlot
  {
    
  public:
      
    ProxyMessageSlot( AbstractMessageSlotPtr aMessageSlot )
      :
      theMessageSlot( aMessageSlot ),
      theLogger( NULLPTR )
    {
      ;
    }

    // copy constructor
    
    ProxyMessageSlot( ProxyMessageSlot& rhs )
      :
      theMessageSlot( rhs.theMessageSlot ),
      theLogger( rhs.theLogger )
    {
      ;
    }

    // copy constructor 

    ProxyMessageSlot( const ProxyMessageSlot& rhs )
      :
      theMessageSlot( rhs.theMessageSlot ),
      theLogger( rhs.theLogger )
    {
      ;
    }


    void setLogger( LoggerPtr aLogger )
    {
      theLogger = aLogger;
    }
    
    
    void update( UVariableCref v )
    {
      const Real temp(0.0);
      theLogger->appendData( temp, v );
    }


    bool operator!=( const ProxyMessageSlot& rhs ) const
    {
      if( rhs.theMessageSlot != this->theMessageSlot )
	{
	  return true;
	}
      return false;
    }

      
  private:
      
    ProxyMessageSlot( void );
      
  private:    

    AbstractMessageSlotPtr   theMessageSlot;
    LoggerPtr                theLogger;
    
    
    
  };

  
  /**
     Calls callback methods for getting and sending Message objects.

     @see Message
     @see MessageInterface
     @see AbstractMessageSlot
  */


  template<class T>
  class MessageSlot : public AbstractMessageSlot
  {

  public:

    typedef void ( T::* SetMessageFunc )( MessageCref );
    typedef const Message ( T::* GetMessageFunc )( StringCref );

  public:

    MessageSlot( T& object, const SetMessageFunc setmethod,
		 const GetMessageFunc getmethod )
      : 
      theObject( object ), 
      theSetMethod( setmethod ), 
      theGetMethod( getmethod ),
      theProxy( this )
    {
      ; // do nothing
    }
  

    virtual const bool isSetable() const
    {
      return theSetMethod != NULLPTR;
    }

    virtual const bool isGetable() const
    {
      return theGetMethod != NULLPTR;
    }

    virtual void set( MessageCref message )
    {
      if( ! isSetable() )
	{
	  //FIXME: throw an exception
	  return;
	}

      ( theObject.*theSetMethod )( message );
      
      if( theProxy != NULL )
      	{
	  //	  Message m = get("nothing");
	  //	  theProxy.update( m.getBody() );
	} 
    }

    virtual const Message get( StringCref keyword ) 
    {
      if( ! isGetable() )
	{
	  //FIXME: throw an exception
	  return Message( keyword );
	}

      return ( ( theObject.*theGetMethod )( keyword ));
    }

    virtual ProxyMessageSlot* getProxy( void )
    {
      return new ProxyMessageSlot( theProxy );
    }


  private:

    T&                   theObject;
    ProxyMessageSlot     theProxy;
    const SetMessageFunc theSetMethod;
    const GetMessageFunc theGetMethod;
  };



  /**
     Common base class for classes which receive Messages.

     NOTE:  Subclasses of MessageInterface MUST call their own makeSlots(),
     if any, to make their slots in their constructors.
     (virtual functions won't work in constructors)

     FIXME: class-static slots?

     @see Message
     @see MessageSlot
  */

  class MessageInterface
  {
  public:

    enum PropertyAttribute
      {
	SETABLE = ( 1 << 0 ),
	GETABLE = ( 1 << 1 )
      };


    MessageInterface();
    virtual ~MessageInterface();

    void set( MessageCref );
    const Message get( StringCref );

    PropertyMapIterator getMessageSlot( StringCref property )
    {
      return thePropertyMap.find( property );
    }

    virtual void makeSlots();

    virtual const char* const className() const { return "MessageInterface"; }

  public: // message slots

    const Message getPropertyList( StringCref keyword );
    const Message getPropertyAttributes( StringCref keyword );

  protected:

    void appendSlot( StringCref keyword, AbstractMessageSlotPtr );
    void deleteSlot( StringCref keyword );

  private:

    PropertyMap thePropertyMap;

  };


#define makeMessageSlot( KEY, CLASS, OBJ, SETMETHOD, GETMETHOD )\
appendSlot( KEY, new MessageSlot< CLASS >\
	   ( OBJ, static_cast< MessageSlot< CLASS >::SetMessageFunc >\
	    ( SETMETHOD ),\
	    static_cast< MessageSlot< CLASS >::GetMessageFunc >\
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
