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
#include "PropertiedClass.hpp"
#include "convertTo.hpp"
#include "Logger.hpp"
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

  class PropertySlotBase
  {

  public:
    PropertySlotBase()
    {
      ; // do nothing
    }
    
    virtual ~PropertySlotBase()
    {
      ; // do nothing
    }

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    //    virtual PropertyInterfaceCref getPropertyInterface() const = 0;


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

  };


  template
  < 
    class T
  >
  class PropertySlot
    :
    public PropertySlotBase
  {

  public:

    PropertySlot()
    {
      ; // do nothing
    }

    virtual ~PropertySlot()
    {
      ; // do nothing
    }


#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual void set ## TYPE( T& anObject, TYPE ## Cref aValue ) = 0;

#define _PROPERTYSLOT_GETMETHOD( TYPE )\
    virtual const TYPE get ## TYPE( const T& anObject ) const = 0;

    _PROPERTYSLOT_SETMETHOD( Polymorph );
    _PROPERTYSLOT_GETMETHOD( Polymorph );

    _PROPERTYSLOT_SETMETHOD( Real );
    _PROPERTYSLOT_GETMETHOD( Real );

    _PROPERTYSLOT_SETMETHOD( Int );
    _PROPERTYSLOT_GETMETHOD( Int );

    _PROPERTYSLOT_SETMETHOD( String );
    _PROPERTYSLOT_GETMETHOD( String );

#undef _PROPERTYSLOT_SETMETHOD
#undef _PROPERTYSLOT_GETMETHOD

  };


  template
  < 
    class T,
    typename SlotType_
  >
  class ConcretePropertySlot
    :
    public PropertySlot<T>
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

    ConcretePropertySlot( const SetMethodPtr aSetMethodPtr,
			  const GetMethodPtr aGetMethodPtr )
      :
      theSetMethodPtr( SetMethod( aSetMethodPtr ) ),
      theGetMethodPtr( GetMethod( aGetMethodPtr ) )
    {
      ; // do nothing
    }

    virtual ~ConcretePropertySlot()
    {
      ; // do nothing
    }


    virtual const bool isSetable() const
    {
      const SetMethodPtr aNullMethodPtr( &PropertiedClass::nullSet );
      return theSetMethodPtr != aNullMethodPtr;
    }

    virtual const bool isGetable() const
    {
      const GetMethodPtr
	aNullMethodPtr( &PropertiedClass::nullGet<SlotType> );
      return theGetMethodPtr != aNullMethodPtr;
    }

#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual void set ## TYPE( T& anObject, TYPE ## Cref aValue )\
    {\
      setImpl( anObject, aValue );\
    }

#define _PROPERTYSLOT_GETMETHOD( TYPE )\
    virtual const TYPE get ## TYPE( const T& anObject ) const\
    {\
      return getImpl<TYPE>( anObject );\
    }

    _PROPERTYSLOT_SETMETHOD( Polymorph );
    _PROPERTYSLOT_GETMETHOD( Polymorph );

    _PROPERTYSLOT_SETMETHOD( Real );
    _PROPERTYSLOT_GETMETHOD( Real );

    _PROPERTYSLOT_SETMETHOD( Int );
    _PROPERTYSLOT_GETMETHOD( Int );

    _PROPERTYSLOT_SETMETHOD( String );
    _PROPERTYSLOT_GETMETHOD( String );

#undef _PROPERTYSLOT_SETMETHOD
#undef _PROPERTYSLOT_GETMETHOD


  protected:

    inline void callSetMethod( T& anObject, SetType aValue )    
    {
      ( anObject.*theSetMethodPtr )( aValue );
    }

    inline GetType callGetMethod( const T& anObject ) const
    {
      return ( ( anObject.*theGetMethodPtr )() );
    }

    template < typename Type >
    inline void setImpl( T& anObject, Type aValue )
    {
      callSetMethod( anObject, convertTo<SlotType>( aValue ) );
    }
    
    template < typename Type >
    inline const Type getImpl( const T& anObject ) const
    {
      return convertTo<Type>( callGetMethod( anObject ) );
    }


    static SetMethodPtr SetMethod( SetMethodPtr aSetMethodPtr )
    {
      if( aSetMethodPtr == NULLPTR )
	{
	  return &PropertiedClass::nullSet;
	}
      else
	{
	  return aSetMethodPtr;
	}
    }

    static GetMethodPtr GetMethod( GetMethodPtr aGetMethodPtr )
    {
      if( aGetMethodPtr == NULLPTR )
	{
	  return &PropertiedClass::nullGet<SlotType>;
	}
      else
	{
	  return aGetMethodPtr;
	}
    }


  protected:

    const SetMethodPtr theSetMethodPtr;
    const GetMethodPtr theGetMethodPtr;

  };





  /*@}*/

}


#endif /* __PROPERTYSLOT_HPP */
