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
#include "convertTo.hpp"

#include "Polymorph.hpp"

namespace libecs
{


  /** @addtogroup property
      
  @ingroup libecs
  @{
  */

  /** @file */

  DECLARE_VECTOR( ProxyPropertySlotPtr, ProxyPropertySlotVector );

  /**
     Base class for PropertySlot classes.

     @see PropertyInterface
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

    virtual void setPolymorphVectorRCPtr( PolymorphVectorRCPtrCref ) = 0;
    virtual const PolymorphVectorRCPtr getPolymorphVectorRCPtr() const = 0;
    
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


    template < typename Type >
    inline void set( const Type& aValue )
    {
      DefaultSpecializationInhibited();
    }

    template < typename Type >
    inline const Type get() const
    {
      DefaultSpecializationInhibited();
    }

  protected:

    String                   theName;
    LoggerPtr                theLogger;

  };


  template <>
  inline void PropertySlot::set( PolymorphVectorRCPtrCref aValue )
  {
    setPolymorphVectorRCPtr( aValue );
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
  inline const PolymorphVectorRCPtr PropertySlot::get() const
  {
    return getPolymorphVectorRCPtr();
  }

  template <>
  inline const String PropertySlot::get() const
  {
    return getString();
  }

  template <>
  inline const Real PropertySlot::get() const
  {
    return getReal();
  }



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
      NEVER_GET_HERE;
    }

    bool isSet() const
    {
      return theIsSet;
    }

    void setIsSet()
    {
      theIsSet = true;
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



    virtual void setPolymorphVectorRCPtr( PolymorphVectorRCPtrCref aValue )
    {
      setImpl( aValue );
    }

    virtual const PolymorphVectorRCPtr getPolymorphVectorRCPtr() const
    {
      return getImpl<PolymorphVectorRCPtr>();
    }

    virtual void setReal( RealCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Real getReal() const
    {
      return getImpl<Real>();
    }

    virtual void setString( StringCref aValue )
    {
      setImpl( aValue );
    }

    virtual const String getString() const
    {
      return getImpl<String>();
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

    template < typename Type >
    inline void setImpl( const Type& aValue )
    {
      setIsSet();
      theCachedValue = convertTo<SlotType>( aValue );
    }

    template < typename Type >
    inline const Type getImpl() const
    {
      return convertTo<Type>( theOriginalValue );
    }

  private:

    SlotType theOriginalValue;
    SlotType theCachedValue;

  };


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
      std::remove( theProxyVector.begin(), theProxyVector.end(), aProxyPtr );

      //      if( theProxyVector.empty() )
      //	{
      //            **should remove itself from Stepper's slot vector**
      //	}

    }

    virtual void setPolymorphVectorRCPtr( PolymorphVectorRCPtrCref aValue )
    {
      setImpl( aValue );
    }

    virtual const PolymorphVectorRCPtr getPolymorphVectorRCPtr() const
    {
      return getImpl<PolymorphVectorRCPtr>();
    }

    virtual void setReal( RealCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Real getReal() const
    {
      return getImpl<Real>();
    }

    virtual void setString( StringCref aValue )
    {
      setImpl( aValue );
    }

    virtual const String getString() const
    {
      return getImpl<String>();
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
	      callSetMethod( aProxySlotPtr->PropertySlot::get<SlotType>() );
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

	  aProxySlotPtr->setOriginalValue( getImpl<SlotType>() );
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

    template < typename Type >
    inline void setImpl( Type aValue )
    {
      callSetMethod( convertTo<SlotType>( aValue ) );
    }

    template < typename Type >
    inline const Type getImpl() const
    {
      return convertTo<Type>( callGetMethod() );
    }

  protected:

    T& theObject;
    const SetMethodPtr theSetMethodPtr;
    const GetMethodPtr theGetMethodPtr;

    ProxyVector        theProxyVector;

  };



  template
  < 
    class T,
    typename SlotType_
  >
  class ConcretePropertySlotWithSyncMethod
    :
    public ConcretePropertySlot<T,SlotType_>
  {

  public:


    typedef ConcretePropertySlot<T,SlotType_> ConcreteSlot;

    typedef typename ConcreteSlot::SlotType SlotType;
    typedef typename ConcreteSlot::GetType  GetType;
    typedef typename ConcreteSlot::SetType  SetType;

    typedef typename ConcreteSlot::GetMethodPtr GetMethodPtr;
    typedef typename ConcreteSlot::SetMethodPtr SetMethodPtr;


    typedef typename ConcreteSlot::ProxySlotPtr ProxySlotPtr;
    typedef typename ConcreteSlot::ProxyVectorIterator ProxyVectorIterator;
    typedef typename ConcreteSlot::ProxyVectorConstIterator
    ProxyVectorConstIterator;


    ConcretePropertySlotWithSyncMethod( StringCref aName, T& anObject, 
					const SetMethodPtr aSetMethodPtr,
					const GetMethodPtr aGetMethodPtr,
					const SetMethodPtr aSyncMethodPtr )
      :
      ConcreteSlot( aName, anObject, aSetMethodPtr, aGetMethodPtr ),
      theSyncMethodPtr( aSyncMethodPtr )
    {
      ; // do nothing
    }

    virtual ~ConcretePropertySlotWithSyncMethod()
    {
      ; // do nothing
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
	      callSyncMethod( aProxySlotPtr->PropertySlot::get<SlotType>() );
	      aProxySlotPtr->clearIsSet();
	    }
	}
    }


  protected:

    inline void callSyncMethod( SetType aValue )    
    {
      ( theObject.*theSyncMethodPtr )( aValue );
    }

  protected:

    const SetMethodPtr theSyncMethodPtr;

  };



  /*@}*/

}


#endif /* __PROPERTYSLOT_HPP */
