//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __PROPERTYSLOT_HPP
#define __PROPERTYSLOT_HPP

#include <functional>

#include "libecs.hpp"
//#include "Util.hpp"
#include "PropertiedClass.hpp"
#include "convertTo.hpp"
#include "Polymorph.hpp"


namespace libecs
{


  /** @addtogroup property
      
  @ingroup libecs
  @{
  */

  /** @file */



  // convert std::map to a Polymorph which is a nested-list.
  //  template< class MAP >

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
    
    ECELL_API virtual ~PropertySlotBase();

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    ECELL_API virtual const bool isLoadable() const;
    ECELL_API virtual const bool isSavable()  const;

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

    typedef void    ( T::* SetPolymorphMethodPtr )( PolymorphCref );
    typedef const Polymorph ( T::* GetPolymorphMethodPtr )() const;

    PropertySlot()
    {
      ; // do nothing
    }

    virtual ~PropertySlot()
    {
      ; // do nothing
    }


#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual void set ## TYPE( T& anObject, Param<TYPE>::type value ) = 0;

#define _PROPERTYSLOT_GETMETHOD( TYPE )\
    virtual const TYPE get ## TYPE( const T& anObject ) const = 0;

    _PROPERTYSLOT_SETMETHOD( Polymorph );
    _PROPERTYSLOT_GETMETHOD( Polymorph );

    _PROPERTYSLOT_SETMETHOD( Real );
    _PROPERTYSLOT_GETMETHOD( Real );

    _PROPERTYSLOT_SETMETHOD( Integer );
    _PROPERTYSLOT_GETMETHOD( Integer );

    _PROPERTYSLOT_SETMETHOD( String );
    _PROPERTYSLOT_GETMETHOD( String );

#undef _PROPERTYSLOT_SETMETHOD
#undef _PROPERTYSLOT_GETMETHOD


    virtual void loadPolymorph( T& anObject, Param<Polymorph>::type aValue )
    {
      setPolymorph( anObject, aValue );
    }

    virtual const Polymorph savePolymorph( const T& anObject ) const
    {
      return getPolymorph( anObject );
    }

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

    typedef typename Param<SlotType>::type  SetType;
    typedef const SlotType                  GetType;

    typedef void    ( T::* SetMethodPtr )( SetType );
    typedef GetType ( T::* GetMethodPtr )() const;

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
      return isSetableMethod( theSetMethodPtr );
    }

    virtual const bool isGetable() const
    {
      return isGetableMethod( theGetMethodPtr );
    }

#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual void set ## TYPE( T& anObject, Param<TYPE>::type aValue )\
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

    _PROPERTYSLOT_SETMETHOD( Integer );
    _PROPERTYSLOT_GETMETHOD( Integer );

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
      return ( anObject.*theGetMethodPtr )();
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


    static const bool isSetableMethod( const SetMethodPtr aSetMethodPtr )
    {
      const SetMethodPtr aNullMethodPtr( &PropertiedClass::nullSet<SlotType> );
      return aSetMethodPtr != aNullMethodPtr;
    }

    static const bool isGetableMethod( const GetMethodPtr aGetMethodPtr )
    {
      const GetMethodPtr
	aNullMethodPtr( &PropertiedClass::nullGet<SlotType> );
      return aGetMethodPtr != aNullMethodPtr;
    }


    static SetMethodPtr SetMethod( SetMethodPtr aSetMethodPtr )
    {
      if( aSetMethodPtr == NULLPTR )
	{
	  return &PropertiedClass::nullSet<SlotType>;
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

  template
  < 
    class T,
    typename SlotType_
  >
  class LoadSaveConcretePropertySlot
    :
    public ConcretePropertySlot<T,SlotType_>
  {

  public:

    DECLARE_TYPE( SlotType_, SlotType );

    typedef ConcretePropertySlot<T,SlotType> ConcretePropertySlot;

    typedef typename ConcretePropertySlot::SetType SetType;
    typedef typename ConcretePropertySlot::GetType GetType;

    typedef typename ConcretePropertySlot::SetMethodPtr SetMethodPtr;
    typedef typename ConcretePropertySlot::GetMethodPtr GetMethodPtr;

    LoadSaveConcretePropertySlot( const SetMethodPtr aSetMethodPtr,
				  const GetMethodPtr aGetMethodPtr,
				  const SetMethodPtr aLoadMethodPtr,
				  const GetMethodPtr aSaveMethodPtr )
      :
      ConcretePropertySlot( aSetMethodPtr, aGetMethodPtr ),
      theLoadMethodPtr( SetMethod( aLoadMethodPtr ) ),
      theSaveMethodPtr( GetMethod( aSaveMethodPtr ) )
    {
      ; // do nothing
    }

    ~LoadSaveConcretePropertySlot()
    {
      ; // do nothing
    }


    virtual const bool isLoadable() const
    {
      return isSetableMethod( theLoadMethodPtr );
    }

    virtual const bool isSavable()  const
    {
      return isGetableMethod( theSaveMethodPtr );
    }

    virtual void loadPolymorph( T& anObject, Param<Polymorph>::type aValue )
    {
      loadImpl( anObject, aValue );
    }

    virtual const Polymorph savePolymorph( const T& anObject ) const
    {
      return saveImpl( anObject );
    }


  protected:

    inline void callLoadMethod( T& anObject, SetType aValue )    
    {
      ( anObject.*theLoadMethodPtr )( aValue );
    }

    inline GetType callSaveMethod( const T& anObject ) const
    {
      return ( anObject.*theSaveMethodPtr )();
    }

    inline void loadImpl( T& anObject, PolymorphCref aValue )
    {
      callLoadMethod( anObject, convertTo<SlotType>( aValue ) );
    }
    
    inline const Polymorph saveImpl( const T& anObject ) const
    {
      return convertTo<Polymorph>( callSaveMethod( anObject ) );
    }

  protected:

    const SetMethodPtr theLoadMethodPtr;
    const GetMethodPtr theSaveMethodPtr;

  };

  /*@}*/

}


#endif /* __PROPERTYSLOT_HPP */
