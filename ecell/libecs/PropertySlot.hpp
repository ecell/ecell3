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

#ifndef __PROPERTYSLOT_HPP
#define __PROPERTYSLOT_HPP

#include <iostream>
#include <signal.h>

#include "libecs.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "UVariable.hpp"

namespace libecs
{

  template< typename FromType, typename ToType >
  ToType convertTo( const FromType& aValue )
  {
    convertTo( aValue, Type2Type< ToType >() );
  }

  template< typename FromType, typename ToType >
  ToType convertTo( const FromType& aValue, Type2Type<ToType> )
  {
    DefaultSpecializationInhibited();
    //#warning "unexpected default specialization of convertTo()"
    //    return ToType( aValue );
  }


  // to UVariableVectorRCPtr

  // identity

  template<>
  inline const UVariableVectorRCPtr 
  convertTo( UVariableVectorRCPtrCref aValue, 
	     Type2Type< const UVariableVectorRCPtr > )
  {
    return aValue;
  }

  // from Real
  template<>
  inline const UVariableVectorRCPtr 
  convertTo( RealCref aValue,
	     Type2Type< const UVariableVectorRCPtr > )
  {
    UVariableVectorRCPtr aVector( new UVariableVector );
    aVector->push_back( aValue );
    return aVector;
  }

  // from String
  template<>
  inline const UVariableVectorRCPtr 
  convertTo( StringCref aValue,
	     Type2Type< const UVariableVectorRCPtr > )
  {
    UVariableVectorRCPtr aVectorPtr( new UVariableVector );
    aVectorPtr->push_back( aValue );
    return aVectorPtr;
  }

  // to String

  // identity
  template<>
  inline const String
  convertTo( StringCref aValue,
	     Type2Type< const String > )
  {
    return aValue;
  }



  template<>
  inline const String convertTo( UVariableVectorRCPtrCref aValue,
				 Type2Type< const String > )
  {
    return (*aValue)[0].asString();
  }

  template<>
  inline const String convertTo( RealCref aValue,
				 Type2Type< const String > )
  {
    return toString( aValue );
  }


  // to Real


  // identity
  template<>
  inline const Real
  convertTo( RealCref aValue,
	     Type2Type< const Real > )
  {
    return aValue;
  }

  template<>
  inline const Real convertTo( UVariableVectorRCPtrCref aValue,
			       Type2Type< const Real > )
  {
    return (*aValue)[0].asReal();
  }
    
  template<>
  inline const Real convertTo( StringCref aValue,
			       Type2Type< const Real > )
  {
    return stringTo< Real >( aValue );
  }


  DECLARE_VECTOR( ProxyPropertySlotPtr, ProxyPropertySlotVector );

  /**
     Base class for PropertySlot classes.

     \see PropertyInterface
     \ingroup property
  */

  class PropertySlot
  {

  public:

    PropertySlot( StringCref aName )
      :
      theName( aName ),
      //      // FIXME: dummyLogger ?
      theLogger( NULLPTR )
    {
      ; // do nothing
    }
    
    //    virtual ~PropertySlotBase()
    virtual ~PropertySlot()
    {
      ; // do nothing
    }

    virtual void setUVariableVectorRCPtr( UVariableVectorRCPtrCref ) = 0;
    virtual const UVariableVectorRCPtr getUVariableVectorRCPtr() const = 0;
    
    virtual void setReal( RealCref real ) = 0;
    virtual const Real getReal() const = 0;

    virtual void setString( StringCref string ) = 0;
    virtual const String getString() const = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;


    virtual void sync() = 0;
    virtual void push() = 0;

    virtual ProxyPropertySlotPtr createProxy() = 0;

    StringCref getName() const
    {
      return theName;
    }

    const bool isLogged()
    {
      return theLogger != NULLPTR;
    }

    // this method is here so that ProxyPropertySlot can call this.
    virtual void disconnectProxy( ProxyPropertySlotPtr aProxyPtr ) 
    {
      NEVER_GET_HERE;
    }

    void connectLogger( LoggerPtr logger );

    void disconnectLogger();

    LoggerCptr getLogger() const
    {
      return theLogger;
    }

    void updateLogger();


    template < typename TYPE >
    inline void set( TYPE aValue )
    {
      DefaultSpecializationInhibited();
    }

    template < typename TYPE >
    inline TYPE get()
    {
      DefaultSpecializationInhibited();
    }

  protected:

    String                   theName;
    LoggerPtr                theLogger;

  };


  template <>
  inline void PropertySlot::set( UVariableVectorRCPtrCref aValue )
  {
    setUVariableVectorRCPtr( aValue );
  }

  template <>
  inline void PropertySlot::set( RealCref aValue )
  {
    setReal( aValue );
  }

  template <>
  inline void PropertySlot::set( StringCref aValue )
  {
    setString( aValue );
  }

  template <>
  inline const UVariableVectorRCPtr PropertySlot::get()
  {
    return getUVariableVectorRCPtr();
  }

  template <>
  inline const String PropertySlot::get()
  {
    return getString();
  }

  template <>
  inline const Real PropertySlot::get()
  {
    return getReal();
  }


  /**

  \ingroup property
  */

  class ProxyPropertySlot
    :
    public PropertySlot
  {
    
  public:
      
    ProxyPropertySlot( PropertySlotRef aPropertySlot )
      :
      PropertySlot( aPropertySlot.getName() ),
      thePropertySlot( aPropertySlot ),
      theIsSet( false )
    {
      std::cerr << aPropertySlot.getName() << std::endl;
    }

    virtual ~ProxyPropertySlot()
    {
      thePropertySlot.disconnectProxy( this );
    }


    virtual const bool isSetable() const
    {
      return thePropertySlot.isSetable();
    }
    virtual const bool isGetable() const
    {
      return thePropertySlot.isGetable();
    }

    virtual ProxyPropertySlotPtr createProxy()
    {
      assert( 0 );
    }

    bool isSet() const
    {
      return theIsSet;
    }

    void setIsSet()
    {
      theIsSet = false;
    }

    void clearIsSet()
    {
      theIsSet = false;
    }

    bool operator==( ProxyPropertySlotCref rhs ) const
    {
      if( &(rhs.thePropertySlot) == &(this->thePropertySlot) )
	{
	  return true;
	}
      return false;
    }


  private:
      
    ProxyPropertySlot( void );
    ProxyPropertySlot( ProxyPropertySlotRef  );
    ProxyPropertySlot( ProxyPropertySlotCref );
    void operator=( ProxyPropertySlotCref );
      
  protected:

    PropertySlotRef   thePropertySlot;
    bool     theIsSet;
    
  };



  /**

  \ingroup property
  */

  template
  < 
    class T,
    typename SlotType_
  >
  class ConcreteProxyPropertySlot
    :
    public ProxyPropertySlot
  {

  public:

    DECLARE_TYPE( SlotType_, SlotType );

    ConcreteProxyPropertySlot( PropertySlotRef aPropertySlot )
      :
      ProxyPropertySlot( aPropertySlot ),
      // FIXME: should be something like null_value<SlotType>
      theOriginalValue( 0 ),
      theCachedValue( 0 )
    {
      ; // do nothing
    }

    virtual ~ConcreteProxyPropertySlot()
    {
      ; // do nothing
    }



    virtual void setUVariableVectorRCPtr( UVariableVectorRCPtrCref aValue )
    {
      setImpl( aValue );
    }

    virtual const UVariableVectorRCPtr getUVariableVectorRCPtr() const
    {
      return getImpl< const UVariableVectorRCPtr >();
    }

    virtual void setReal( RealCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Real getReal() const
    {
      return getImpl< const Real >();
    }

    virtual void setString( StringCref aValue )
    {
      setImpl( aValue );
    }

    virtual const String getString() const
    {
      return getImpl< const String >();
    }

    virtual void sync()
    {
      NEVER_GET_HERE;
    }

    virtual void push() 
    {
      NEVER_GET_HERE;
    }


    void setOriginalValue( const SlotType& aValue )
    {
      theOriginalValue = aValue;
    }


  protected:

    template < typename TYPE >
    inline void setImpl( TYPE aValue )
    {
      setIsSet();
      theCachedValue = convertTo( aValue, Type2Type<const SlotType>() );
    }

    template < typename TYPE >
    inline TYPE getImpl() const
    {
      return convertTo( theOriginalValue, Type2Type<TYPE>() );
    }

  private:

    SlotType theOriginalValue;
    SlotType theCachedValue;

  };


  /**

  \ingroup property
  */

  template
  < 
    class T,
    typename SlotType_
  >
  class ConcretePropertySlot
    :
    public PropertySlot
  {

  public:

    DECLARE_TYPE( SlotType_, SlotType );

    //???
    typedef const SlotType, GetType;
    typedef SlotTypeCref, SetType;

    typedef GetType ( T::* GetMethodPtr )() const;
    typedef void    ( T::* SetMethodPtr )( SlotTypeCref );

    typedef GetType ( ConcretePropertySlot::* CallGetMethodPtr )() const;
    typedef void    ( ConcretePropertySlot::* CallSetMethodPtr )( SetType );

    typedef ConcreteProxyPropertySlot<T,SlotType> ProxySlot_;
    DECLARE_TYPE( ProxySlot_, ProxySlot );

    typedef std::vector<ProxySlotPtr> ProxyVector;
    typedef typename ProxyVector::iterator ProxyVectorIterator;
    typedef typename ProxyVector::const_iterator ProxyVectorConstIterator;


    ConcretePropertySlot( StringCref aName, T& anObject, 
			  const SetMethodPtr aSetMethodPtr,
			  const GetMethodPtr aGetMethodPtr )
      :
      PropertySlot( aName ),
      theObject( anObject ),
      theSetMethodPtr( aSetMethodPtr ),
      theGetMethodPtr( aGetMethodPtr )
    {
      ; // do nothing
    }

    virtual ~ConcretePropertySlot()
    {
      ; // do nothing
    }

    virtual void connectProxy( ProxySlotPtr aProxyPtr )
    {
      theProxyVector.push_back( aProxyPtr );
    }

    // this takes ProxyPropertySlotPtr.  
    // In this way ~ProxyPropertySlot can call PropertySlot's disconnectProxy.
    virtual void disconnectProxy( ProxyPropertySlotPtr aProxyPtr )
    {
      std::remove( theProxyVector.begin(),
		   theProxyVector.end(),
		   aProxyPtr );

      //      if( theProxyVector.empty() )
      //	{
      //            **should remove itself from Stepper's slot vector**
      //	}

    }

    virtual void setUVariableVectorRCPtr( UVariableVectorRCPtrCref aValue )
    {
      setImpl( aValue );
    }

    virtual const UVariableVectorRCPtr getUVariableVectorRCPtr() const
    {
      return getImpl< const UVariableVectorRCPtr >();
    }

    virtual void setReal( RealCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Real getReal() const
    {
      return getImpl< const Real >();
    }

    virtual void setString( StringCref aValue )
    {
      setImpl( aValue );
    }

    virtual const String getString() const
    {
      return getImpl< const String >();
    }

    virtual void sync()
    {
      for( ProxyVectorConstIterator i( theProxyVector.begin() ); 
	   i != theProxyVector.end(); ++i )
	{
	  ProxySlotPtr aProxySlotPtr( *i );
	  
	  // get the value only if the proxy has a value set.
	  if( aProxySlotPtr->isSet() )
	    {
	      PropertySlot::set<SlotTypeCref>( aProxySlotPtr->PropertySlot::
					       get<const SlotType>() );
	      aProxySlotPtr->clearIsSet();
	    }
	}
    }

    virtual void push()
    {
      for( ProxyVectorConstIterator i( theProxyVector.begin() ); 
	   i != theProxyVector.end(); ++i )
	{
	  ProxySlotPtr aProxySlotPtr( *i );

	  aProxySlotPtr->setOriginalValue( getImpl<GetType>() );
	}
    }

    virtual const bool isSetable() const
    {
      const SetMethodPtr aNullMethodPtr( &PropertyInterface::nullSet );
      return theSetMethodPtr != aNullMethodPtr;
    }

    virtual const bool isGetable() const
    {
      const GetMethodPtr
	aNullMethodPtr( &PropertyInterface::nullGet<SlotType> );
      return theGetMethodPtr != aNullMethodPtr;
    }


    virtual ProxyPropertySlotPtr createProxy()
    {
      ProxySlotPtr aProxySlotPtr( new ProxySlot( *this ) );
    
      connectProxy( aProxySlotPtr );

      return aProxySlotPtr;
    }
      


  protected:

    inline void callSetMethod( SetType aValue )    
    {
      ( theObject.*theSetMethodPtr )( aValue );
    }

    inline GetType callGetMethod() const
    {
      return ( ( theObject.*theGetMethodPtr )() );
    }


    template < typename TYPE >
    inline void setImpl( TYPE aValue )
    {
      callSetMethod( convertTo( aValue, Type2Type< SetType >() ) );
    }

    template < typename TYPE >
    inline TYPE getImpl() const
    {
      return convertTo( callGetMethod(), Type2Type< TYPE >() );
    }

  private:

    T& theObject;
    const SetMethodPtr theSetMethodPtr;
    const GetMethodPtr theGetMethodPtr;

    ProxyVector        theProxyVector;

  };




  /** @} */ //end of libecs_module 

}


#endif /* __PROPERTYSLOT_HPP */
