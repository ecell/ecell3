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

#include "Util.hpp"

#include "libecs.hpp"
#include "Logger.hpp"
#include "Message.hpp"




namespace libecs
{

  DECLARE_MAP( const String, PropertySlotPtr, 
	       less<const String>, PropertyMap );

  class Logger;
  class ProxyPropertySlot;


  /**
     A base class for PropertySlot classes.

     @see PropertyInterface
     @see Message
  */
  class PropertySlot
  {

  public:

  public:

    PropertySlot( StringCref name )
      :
      theName( name )
    {
      ; // do nothing
    }
    
    virtual void setUVariableVector( UVariableVectorCref ) = 0;
    virtual UVariableVectorRCPtr getUVariableVector() const = 0;
    
    virtual void setReal( const Real real ) = 0;
    virtual const Real getReal() const = 0;

    virtual void setString( StringCref string ) = 0;
    virtual const String getString() const = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    //    virtual PropertySlotProxyPtr createProxy( void ) = 0;

    StringCref getName() const
    {
      return theName;
    }

    void setLogger( LoggerPtr logger )
    {
      theLogger = logger;
    }

    LoggerCptr getLogger() const
    {
      return theLogger;
    }

    void pushData()
    {
      theLogger->appendData( getCurrentTime(), getReal() );
    }

  protected:

    virtual Real getCurrentTime() = 0;

  protected:

    String                   theName;
    LoggerPtr                theLogger;
    

  };


#if 0 
  class PropertySlotProxy
  {
    
  public:
      
    PropertySlotProxy( PropertySlotPtr aPropertySlot )
      :
      thePropertySlot( aPropertySlot )
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

    PropertySlotPtr   thePropertySlot;
    
    
  };
#endif // 0

  template<class T>
  class ClassPropertySlot
    :
    public PropertySlot
  {

    typedef void ( T::* SetUVariableVectorMethodPtr )( UVariableVectorCref );
    typedef const UVariableVectorRCPtr 
    ( T::* GetUVariableVectorMethodPtr )() const;
    typedef void ( T::* SetRealMethodPtr )( const Real );
    typedef const Real ( T::* GetRealMethodPtr )() const;
    typedef void ( T::* SetStringMethodPtr )( StringCref );
    typedef const String ( T::* GetStringMethodPtr )() const;

  public:

    ClassPropertySlot( StringCref name, T& object );

    virtual Real getCurrentTime()
    {
      //FIXME:
      return 0;
    }

  protected:

    T& theObject;

  };

  /**

     @see UVariable
     @see PropertyInterface
     @see PropertySlot
  */


  template<class T>
  class UVariableVectorPropertySlot
    : 
    public ClassPropertySlot<T>
  {

  public:

    UVariableVectorPropertySlot( StringCref name, T& object, 
				 const SetUVariableVectorMethodPtr setmethod,
				 const GetUVariableVectorMethodPtr getmethod );
  
    virtual const bool isSetable() const
    {
      return theSetUVariableVectorMethod != NULLPTR;
    }

    virtual const bool isGetable() const
    {
      return theGetUVariableVectorMethod != NULLPTR;
    }

    virtual void setUVariableVector( UVariableVectorCref message )
    {
      if( ! isSetable() )
	{
	  throw AttributeError( "[" + theName + "] is not settable." );
	}

      ( theObject.*theSetUVariableVectorMethod )( message );
    }

    virtual UVariableVectorRCPtr getUVariableVector() const
    {
      if( ! isGetable() )
	{
	  throw AttributeError( "[" + theName + "] is not gettable." );
	}

      return ( ( theObject.*theGetUVariableVectorMethod )() );
    }

    virtual void setReal( const Real real );

    virtual const Real getReal() const;

    virtual void setString( StringCref string );

    virtual const String getString() const;

    //    virtual PropertySlotProxyPtr createProxy( void )
    //    {
    //      return new PropertySlotProxy( this );
    //    }

  private:

    const SetUVariableVectorMethodPtr theSetUVariableVectorMethod;
    const GetUVariableVectorMethodPtr theGetUVariableVectorMethod;

  };



  /**

     @see Message
     @see PropertyInterface
     @see PropertySlot
  */


  template<class T>
  class RealPropertySlot
    : 
    public ClassPropertySlot<T>
  {

  public:

  public:

    RealPropertySlot( StringCref name, T& object, 
		      const SetRealMethodPtr setmethod,
		      const GetRealMethodPtr getmethod );

    virtual const bool isSetable() const
    {
     return theSetRealMethod != NULLPTR;
    }

    virtual const bool isGetable() const
    {
      return theGetRealMethod != NULLPTR;
    }

    virtual void setReal( const Real real )
    {
      if( ! isSetable() )
	{
	  throw AttributeError( "[" + theName + "] is not settable." );
	}
      
      ( theObject.*theSetRealMethod )( real );
    }

    virtual const Real getReal() const
    {
      if( ! isGetable() )
	{
	  throw AttributeError( "[" + theName + "] is not gettable." );
	}
      
      return ( ( theObject.*theGetRealMethod )() );
    }

    virtual void setUVariableVector( UVariableVectorCref uvector );

    virtual UVariableVectorRCPtr getUVariableVector() const;

    virtual void setString( StringCref string );

    virtual const String getString() const;


  private:

    const SetRealMethodPtr    theSetRealMethod;
    const GetRealMethodPtr    theGetRealMethod;

  };


  /**

     @see Message
     @see PropertyInterface
     @see PropertySlot
  */


  template<class T>
  class StringPropertySlot
    : 
    public ClassPropertySlot<T>
  {

  public:

    StringPropertySlot( StringCref name, T& object, 
			const SetStringMethodPtr setmethod,
			const GetStringMethodPtr getmethod );
  
    virtual const bool isSetable() const
    {
      return theSetStringMethod != NULLPTR;
    }

    virtual const bool isGetable() const
    {
      return theGetStringMethod != NULLPTR;
    }

    virtual void setString( StringCref string )
    {
      if( ! isSetable() )
	{
	  throw AttributeError( "[" + theName + "] is not settable." );
	}

      ( theObject.*theSetStringMethod )( string );
    }

    virtual const String getString() const
    {
      if( ! isGetable() )
	{
	  throw AttributeError( "[" + theName + "] is not gettable." );
	}

      return ( ( theObject.*theGetStringMethod )() );
    }


    virtual void setUVariableVector( UVariableVectorCref uvector );

    virtual UVariableVectorRCPtr getUVariableVector() const;

    virtual void setReal( const Real real );

    virtual const Real getReal() const;



  private:

    const SetStringMethodPtr    theSetStringMethod;
    const GetStringMethodPtr    theGetStringMethod;

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
	SETABLE =    ( 1 << 0 ),
	GETABLE =    ( 1 << 1 ),
	CUMULATIVE = ( 1 << 2 )
      };


    PropertyInterface();
    virtual ~PropertyInterface();

    void setMessage( MessageCref );
    const Message getMessage( StringCref ) const;

    PropertyMapConstIterator getPropertySlot( StringCref property ) const
    {
      return thePropertyMap.find( property );
    }

    virtual void makeSlots();

    virtual const char* const className() const { return "PropertyInterface"; }

  public: // message slots

    const UVariableVectorRCPtr getPropertyList() const;
    const UVariableVectorRCPtr getPropertyAttributes() const;


    /**

     createPropertySlot template method provides a standard way 
     to create a new slot.  It is template so that it can accept methods
     of class T (the template parameter class).

    */

    template<class T>
    void
    createPropertySlot( StringCref name,
			T& object,
			ClassPropertySlot<T>::SetUVariableVectorMethodPtr set,
			ClassPropertySlot<T>::GetUVariableVectorMethodPtr get )
    {
      appendSlot( new UVariableVectorPropertySlot<T>( name, 
						      object, 
						      set, 
						      get ) );
    }

    template<class T>
    void
    createPropertySlot( StringCref name,
			T& object,
			ClassPropertySlot<T>::SetRealMethodPtr set,
			ClassPropertySlot<T>::GetRealMethodPtr get )
    {
      appendSlot( new RealPropertySlot<T>( name, 
					   object, 
					   set, 
					   get ) );
    }

    template<class T>
    void
    createPropertySlot( StringCref name,
			T& object,
			ClassPropertySlot<T>::SetStringMethodPtr set,
			ClassPropertySlot<T>::GetStringMethodPtr get )
    {
      appendSlot( new StringPropertySlot<T>( name, 
					   object, 
					   set, 
					   get ) );
    }

    void appendSlot( PropertySlotPtr );
    void deleteSlot( StringCref keyword );

  private:

    PropertyMap thePropertyMap;

  };
  


  //////////// implementation

  ///////////////////////////// ClassPropertySlot

  template< class T >
  ClassPropertySlot<T>::ClassPropertySlot( StringCref name, T& object )
    :
    PropertySlot( name ),
    theObject( object )
  {
    ; // do nothing
  }
  

  ///////////////////////////// UVariableVectorPropertySlot

  template< class T >
  UVariableVectorPropertySlot<T>::
  UVariableVectorPropertySlot( StringCref name, T& object, 
			       const SetUVariableVectorMethodPtr setmethod,
			       const GetUVariableVectorMethodPtr getmethod )
    : 
    ClassPropertySlot<T>( name, object ),
    theSetUVariableVectorMethod( setmethod ), 
    theGetUVariableVectorMethod( getmethod )
  {
    ; // do nothing
  }

  template< class T >
  void UVariableVectorPropertySlot<T>::setReal( const Real real )
  {
    UVariableVector aVector;
    aVector.push_back( real );
    setUVariableVector( aVector );
  }

  template< class T >
  const Real UVariableVectorPropertySlot<T>::getReal() const
  {
    UVariableVectorRCPtr aVectorPtr( getUVariableVector() );
    return (*aVectorPtr)[0].asReal();
  }

  template< class T >
  void UVariableVectorPropertySlot<T>::setString( StringCref string )
  {
    UVariableVector aVector;
    aVector.push_back( string );
    setUVariableVector( aVector );
  }

  template< class T >  
  const String UVariableVectorPropertySlot<T>::getString() const
  {
    UVariableVectorRCPtr aVectorPtr( getUVariableVector() );
    return (*aVectorPtr)[0].asString();
  }

  ///////////////////////////// RealPropertySlot

  
  template< class T >
  RealPropertySlot<T>::RealPropertySlot( StringCref name, T& object, 
					 const SetRealMethodPtr setmethod,
					 const GetRealMethodPtr getmethod )
    : 
    ClassPropertySlot<T>( name, object ),
    theSetRealMethod( setmethod ), 
    theGetRealMethod( getmethod )
  {
    ; // do nothing
  }

  template<class T> 
  void RealPropertySlot<T>::setUVariableVector( UVariableVectorCref uvector ) 
  {
    setReal( uvector[0].asReal() );
  }
  
  template<class T> 
  UVariableVectorRCPtr RealPropertySlot<T>::getUVariableVector() const
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    aVector->push_back( UVariable( getReal() ) );
    return aVector;
  }

  template<class T> 
  void RealPropertySlot<T>::setString( StringCref string )
  {
    setReal( stringTo<Real>( string ) );
  }

  template<class T> 
  const String RealPropertySlot<T>::getString() const
  {
    return toString( getReal() );
  }

  ///////////////////////////// StringPropertySlot


  template< class T >
  StringPropertySlot<T>::
  StringPropertySlot( StringCref name, T& object, 
		      const SetStringMethodPtr setmethod,
		      const GetStringMethodPtr getmethod )
    :
    ClassPropertySlot<T>( name, object ),
    theSetStringMethod( setmethod ), 
    theGetStringMethod( getmethod )
  {
    ; // do nothing
  }


  template<class T> 
  void StringPropertySlot<T>::setUVariableVector( UVariableVectorCref uvector )
  {
    setString( uvector[0].asString() );
  }
  
  template<class T> 
  UVariableVectorRCPtr StringPropertySlot<T>::getUVariableVector() const
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    aVector->push_back( UVariable( getString() ) );
    return aVector;
  }

  template<class T> 
  void StringPropertySlot<T>::setReal( const Real real )
  {
    setString( toString( real ) );
  }

  template<class T> 
  const Real StringPropertySlot<T>::getReal() const
  {
    return stringTo<Real>( getString() );
  }



} // namespace libecs

#endif /* __PROPERTYINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
