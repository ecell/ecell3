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

#ifndef __PROPERTYINTERFACE_HPP
#define __PROPERTYINTERFACE_HPP

#include <map>


#include "libecs.hpp"
#include "Logger.hpp"
#include "Message.hpp"


namespace libecs
{

  DECLARE_MAP( const String, AbstractPropertySlotPtr, 
	       less<const String>, PropertyMap );

  class Logger;
  class ProxyPropertySlot;


  /**
     A base class for PropertySlot class.

     @see PropertySlot
     @see PropertyInterface
     @see Message
  */
  class AbstractPropertySlot
  {

  public:

    virtual void set( MessageCref message ) = 0;
    virtual const Message get( StringCref keyword ) = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    virtual void operator()( MessageCref message ) 
    { 
      set( message ); 
    }

    virtual const Message operator()( StringCref keyword ) 
    { 
      return get(keyword); 
    }

    virtual PropertySlotProxyPtr createProxy( void ) = 0;

  };


  class PropertySlotProxy
  {
    
  public:
      
    PropertySlotProxy( AbstractPropertySlotPtr aPropertySlot )
      :
      thePropertySlot( aPropertySlot ),
      theLogger( NULLPTR )
    {
      ; // do nothing
    }

    // copy constructor
    
    PropertySlotProxy( PropertySlotProxyRef rhs )
      :
      thePropertySlot( rhs.thePropertySlot ),
      theLogger( rhs.theLogger )
    {
      ;
    }

    // copy constructor 

    PropertySlotProxy( PropertySlotProxyCref rhs )
      :
      thePropertySlot( rhs.thePropertySlot ),
      theLogger( rhs.theLogger )
    {
      ;
    }


    void setLogger( LoggerPtr aLogger )
    {
      theLogger = aLogger;
    }
    
    
    void update( UConstantCref v )
    {
      const Real temp(0.0);
      theLogger->appendData( temp, v );
    }


    bool operator!=( PropertySlotProxyCref rhs ) const
    {
      if( rhs.thePropertySlot != this->thePropertySlot )
	{
	  return true;
	}
      return false;
    }

      
  private:
      
    PropertySlotProxy( void );
      
  private:    

    AbstractPropertySlotPtr   thePropertySlot;
    LoggerPtr                 theLogger;
    
    
    
  };

  
  /**
     Calls callback methods for getting and sending Message objects.

     @see Message
     @see PropertyInterface
     @see AbstractPropertySlot
  */


  template<class T>
  class PropertySlot
    : 
    public AbstractPropertySlot
  {

  public:

    typedef void ( T::* SetPropertyFuncPtr )( MessageCref );
    typedef const Message ( T::* GetPropertyFuncPtr )( StringCref );

  public:

    PropertySlot( T& object, const SetPropertyFuncPtr setmethod,
		  const GetPropertyFuncPtr getmethod )
      : 
      theObject( object ), 
      theSetMethod( setmethod ), 
      theGetMethod( getmethod )
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
	  throw AttributeError( "[" + theName + "] is not settable." );
	}

      ( theObject.*theSetMethod )( message );
    }

    virtual const Message get( StringCref keyword ) 
    {
      if( ! isGetable() )
	{
	  throw AttributeError( "[" + theName + "] is not gettable." );
	}

      return ( ( theObject.*theGetMethod )( keyword ) );
    }

    virtual PropertySlotProxyPtr createProxy( void )
    {
      return new PropertySlotProxy( this );
    }


  private:

    T&                       theObject;
    const SetPropertyFuncPtr theSetMethod;
    const GetPropertyFuncPtr theGetMethod;
    String                   theName;
  };



  /**
     Common base class for classes which receive Messages.

     NOTE:  Subclasses of PropertyInterface MUST call their own makeSlots(),
     if any, to make their slots in their constructors.
     (virtual functions won't work in constructors)

     FIXME: class-static slots?

     @see Message
     @see PropertySlot
  */

  class PropertyInterface
  {
  public:

    enum PropertyAttribute
      {
	SETABLE = ( 1 << 0 ),
	GETABLE = ( 1 << 1 )
      };


    PropertyInterface();
    virtual ~PropertyInterface();

    void set( MessageCref );
    const Message get( StringCref );

    PropertyMapIterator getPropertySlot( StringCref property )
    {
      return thePropertyMap.find( property );
    }

    virtual void makeSlots();

    virtual const char* const className() const { return "PropertyInterface"; }

  public: // message slots

    const Message getPropertyList( StringCref keyword );
    const Message getPropertyAttributes( StringCref keyword );

  protected:

    void appendSlot( StringCref keyword, AbstractPropertySlotPtr );
    void deleteSlot( StringCref keyword );

  private:

    PropertyMap thePropertyMap;

  };
  

#define makePropertySlot( KEY, CLASS, OBJ, SETMETHOD, GETMETHOD )\
appendSlot( KEY, new PropertySlot< CLASS >\
	   ( OBJ, static_cast< PropertySlot< CLASS >::SetPropertyFuncPtr >\
	    ( SETMETHOD ),\
	    static_cast< PropertySlot< CLASS >::GetPropertyFuncPtr >\
	    ( GETMETHOD ) ) )


} // namespace libecs

#endif /* __PROPERTYINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
