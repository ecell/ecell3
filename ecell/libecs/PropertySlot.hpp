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
#include <functional>

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

  /**
     Base class for PropertySlot classes.

     @see PropertyInterface
  */

  class PropertySlot
  {

  public:

    class LogCaller 
      :
      std::binary_function< PropertySlotPtr, Real, void > 
    {
    public:
      LogCaller(){}

      void operator()( const PropertySlotPtr& aPropertySlotPtr, 
		       RealCref aTime ) const
      {
	aPropertySlotPtr->log( aTime );
      }

    };


  public:

    PropertySlot()
      :
      theLogger( NULLPTR )
    {
      ; // do nothing
    }
    
    virtual ~PropertySlot()
    {
      ; // do nothing
    }

    virtual void setPolymorph( PolymorphCref ) = 0;
    virtual const Polymorph getPolymorph() const = 0;
    
    virtual void setReal( RealCref real ) = 0;
    virtual const Real getReal() const = 0;

    virtual void setInt( IntCref real ) = 0;
    virtual const Int getInt() const = 0;

    virtual void setString( StringCref string ) = 0;
    virtual const String getString() const = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    const bool isLogged()
    {
      return theLogger != NULLPTR;
    }

    void connectLogger( LoggerPtr logger );

    void disconnectLogger();

    LoggerCptr getLogger() const
    {
      return theLogger;
    }

    void log( const Real aTime ) const;

    virtual PropertyInterfaceCref getPropertyInterface() const = 0;


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

    LoggerPtr                theLogger;

  };


  template <>
  inline void PropertySlot::set( PolymorphCref aValue )
  {
    setPolymorph( aValue );
  }

  template <>
  inline void PropertySlot::set( RealCref aValue )
  {
    setReal( aValue );
  }

  template <>
  inline void PropertySlot::set( IntCref aValue )
  {
    setInt( aValue );
  }

  template <>
  inline void PropertySlot::set( StringCref aValue )
  {
    setString( aValue );
  }

  template <>
  inline const Polymorph PropertySlot::get() const
  {
    return getPolymorph();
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


  template <>
  inline const Int PropertySlot::get() const
  {
    return getInt();
  }



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

    ConcretePropertySlot( T& anObject, 
			  const SetMethodPtr aSetMethodPtr,
			  const GetMethodPtr aGetMethodPtr )
      :
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

    virtual void setPolymorph( PolymorphCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Polymorph getPolymorph() const
    {
      return getImpl<Polymorph>();
    }

    virtual void setReal( RealCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Real getReal() const
    {
      return getImpl<Real>();
    }

    virtual void setInt( IntCref aValue )
    {
      setImpl( aValue );
    }

    virtual const Int getInt() const
    {
      return getImpl<Int>();
    }

    virtual void setString( StringCref aValue )
    {
      setImpl( aValue );
    }

    virtual const String getString() const
    {
      return getImpl<String>();
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

    virtual PropertyInterfaceCref getPropertyInterface() const
    {
      return static_cast<PropertyInterfaceCref>( theObject );
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

  };


  /*@}*/

}


#endif /* __PROPERTYSLOT_HPP */
