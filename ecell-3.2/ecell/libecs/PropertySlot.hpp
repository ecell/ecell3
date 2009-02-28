//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#include <functional>

#include "dmtool/DMObject.hpp"

#include "libecs/Defs.hpp"
#include "libecs/convertTo.hpp"
#include "libecs/Polymorph.hpp"

#ifndef __LIBECS_PROPERTYSLOTBASE_DEFINED
#define __LIBECS_PROPERTYSLOTBASE_DEFINED
namespace libecs
{
/**
   Base class for PropertySlot classes.

   @see PropertyInterface
*/
class DM_IF PropertySlotBase
{
public:
    enum Type
    {
        Polymorph = 0,
        POLYMORPH = 0,
        Real      = 1,
        REAL      = 1,
        Integer   = 2,
        INTEGER   = 2,
        String    = 3,
        STRING    = 3
    };

    template< typename T_ >
    struct TypeToTypeCode
    {
    };

public:
    PropertySlotBase()
    {
        ; // do nothing
    }
    
    virtual ~PropertySlotBase();

    virtual StringCref getName() const = 0;

    virtual enum Type getType() const = 0;

    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    virtual const bool isDynamic() const { return false; }

    virtual const bool isLoadable() const;
    virtual const bool isSavable()    const;
};

template<>
struct PropertySlotBase::TypeToTypeCode< Polymorph >
{
    static const enum PropertySlotBase::Type value = PropertySlotBase::POLYMORPH;
};

template<>
struct PropertySlotBase::TypeToTypeCode< Real >
{
    static const enum PropertySlotBase::Type value = PropertySlotBase::REAL;
};

template<>
struct PropertySlotBase::TypeToTypeCode< Integer >
{
    static const enum PropertySlotBase::Type value = PropertySlotBase::INTEGER;
};


template<>
struct PropertySlotBase::TypeToTypeCode< String >
{
    static const enum PropertySlotBase::Type value = PropertySlotBase::STRING;
};

template< class T >
class PropertySlot: public PropertySlotBase
{
public:
    typedef void  ( T::* SetPolymorphMethodPtr )( PolymorphCref );
    typedef const libecs::Polymorph ( T::* GetPolymorphMethodPtr )() const;

    PropertySlot()
    {
        ; // do nothing
    }

    virtual ~PropertySlot()
    {
        ; // do nothing
    }

#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual void set ## TYPE( T& anObject, Param<libecs::TYPE>::type value ) const = 0;

#define _PROPERTYSLOT_GETMETHOD( TYPE )\
    virtual const libecs::TYPE get ## TYPE( const T& anObject ) const = 0;

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


    DM_IF virtual void loadPolymorph( T& anObject, Param<libecs::Polymorph>::type aValue ) const
    {
        setPolymorph( anObject, aValue );
    }

    DM_IF virtual const libecs::Polymorph savePolymorph( const T& anObject ) const
    {
        return getPolymorph( anObject );
    }

};

} // namespace libecs
#endif /* __LIBECS_PROPERTYSLOTBASE_DEFINED */

#ifndef __LIBECS_CONCRETEPROPERTYSLOT_DEFINED
#define __LIBECS_CONCRETEPROPERTYSLOT_DEFINED
namespace libecs
{

template< class T, typename SlotType_ >
class ConcretePropertySlot: public PropertySlot<T>
{

public:

    DECLARE_TYPE( SlotType_, SlotType );

    typedef typename Param<SlotType>::type SetType;
    typedef const SlotType GetType;

    typedef void ( T::* SetMethodPtr )( SetType );
    typedef GetType ( T::* GetMethodPtr )() const;

    ConcretePropertySlot( StringCref aName,
                          const SetMethodPtr aSetMethodPtr,
                          const GetMethodPtr aGetMethodPtr )
        : theName( aName ),
          theType( PropertySlotBase::TypeToTypeCode< SlotType_ >::value ),
          theSetMethodPtr( SetMethod( aSetMethodPtr ) ),
          theGetMethodPtr( GetMethod( aGetMethodPtr ) )
    {
        ; // do nothing
    }

    DM_IF virtual ~ConcretePropertySlot()
    {
        ; // do nothing
    }

    DM_IF virtual StringCref getName() const
    {
        return theName;
    }

    DM_IF virtual enum PropertySlotBase::Type getType() const
    {
        return theType;
    }

    DM_IF virtual const bool isSetable() const
    {
        return isSetableMethod( theSetMethodPtr );
    }

    DM_IF virtual const bool isGetable() const
    {
        return isGetableMethod( theGetMethodPtr );
    }

#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual void set ## TYPE( T& anObject, Param<libecs::TYPE>::type aValue ) const\
    {\
        setImpl( anObject, aValue );\
    }

#define _PROPERTYSLOT_GETMETHOD( TYPE )\
    virtual const libecs::TYPE get ## TYPE( const T& anObject ) const\
    {\
        return getImpl<libecs::TYPE>( anObject );\
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
    inline void callSetMethod( T& anObject, SetType aValue ) const
    {
        ( anObject.*theSetMethodPtr )( aValue );
    }

    inline GetType callGetMethod( const T& anObject ) const
    {
        return ( anObject.*theGetMethodPtr )();
    }

    template < typename Type >
    inline void setImpl( T& anObject, Type aValue ) const
    {
        callSetMethod( anObject, convertTo<SlotType>( aValue ) );
    }
    
    template < typename Type >
    inline const Type getImpl( const T& anObject ) const
    {
        return convertTo<Type>( callGetMethod( anObject ) );
    }

    static const bool isSetableMethod( const SetMethodPtr aSetMethodPtr );

    static const bool isGetableMethod( const GetMethodPtr aGetMethodPtr );

    static SetMethodPtr SetMethod( SetMethodPtr aSetMethodPtr );

    static GetMethodPtr GetMethod( GetMethodPtr aGetMethodPtr );

protected:
    const libecs::String theName;
    const enum PropertySlotBase::Type theType;
    const SetMethodPtr theSetMethodPtr;
    const GetMethodPtr theGetMethodPtr;

};

} // namespace libecs
#endif /* __LIBECS_CONCRETEPROPERTYSLOT_DEFINED */

#ifndef __LIBECS_LOADSAVECONCRETEPROPERTYSLOT_DEFINED
#define __LIBECS_LOADSAVECONCRETEPROPERTYSLOT_DEFINED
namespace libecs
{

template< class T, typename SlotType_ >
class LoadSaveConcretePropertySlot: public ConcretePropertySlot<T,SlotType_>
{
public:
    DECLARE_TYPE( SlotType_, SlotType );

    typedef ConcretePropertySlot<T,SlotType> ConcretePropertySlot;

    typedef typename ConcretePropertySlot::SetType SetType;
    typedef typename ConcretePropertySlot::GetType GetType;

    typedef typename ConcretePropertySlot::SetMethodPtr SetMethodPtr;
    typedef typename ConcretePropertySlot::GetMethodPtr GetMethodPtr;

    DM_IF LoadSaveConcretePropertySlot( StringCref aName,
                                        const SetMethodPtr aSetMethodPtr,
                                        const GetMethodPtr aGetMethodPtr,
                                        const SetMethodPtr aLoadMethodPtr,
                                        const GetMethodPtr aSaveMethodPtr )
        : ConcretePropertySlot( aName, aSetMethodPtr, aGetMethodPtr ),
          theLoadMethodPtr( SetMethod( aLoadMethodPtr ) ),
          theSaveMethodPtr( GetMethod( aSaveMethodPtr ) )
    {
        ; // do nothing
    }

    DM_IF ~LoadSaveConcretePropertySlot()
    {
        ; // do nothing
    }


    DM_IF virtual const bool isLoadable() const
    {
        return isSetableMethod( theLoadMethodPtr );
    }

    DM_IF virtual const bool isSavable()    const
    {
        return isGetableMethod( theSaveMethodPtr );
    }

    DM_IF virtual void loadPolymorph( T& anObject, Param< libecs::Polymorph >::type aValue ) const
    {
        loadImpl( anObject, aValue );
    }

    DM_IF virtual const libecs::Polymorph savePolymorph( const T& anObject ) const
    {
        return saveImpl( anObject );
    }

protected:
    inline void callLoadMethod( T& anObject, SetType aValue ) const
    {
        ( anObject.*theLoadMethodPtr )( aValue );
    }

    inline GetType callSaveMethod( const T& anObject ) const
    {
        return ( anObject.*theSaveMethodPtr )();
    }

    inline void loadImpl( T& anObject, PolymorphCref aValue ) const
    {
        callLoadMethod( anObject, convertTo<SlotType>( aValue ) );
    }
    
    inline const libecs::Polymorph saveImpl( const T& anObject ) const
    {
        return convertTo< libecs::Polymorph >( callSaveMethod( anObject ) );
    }

protected:
    const SetMethodPtr theLoadMethodPtr;
    const GetMethodPtr theSaveMethodPtr;
};

} // namespace libecs
#endif /* __LIBECS_LOADSAVECONCRETEPROPERTYSLOT_DEFINED */

#ifndef __LIBECS_CONCRETEPROPERTYSLOT_MEMBER_DEFINED
#define __LIBECS_CONCRETEPROPERTYSLOT_MEMBER_DEFINED
#include "libecs/EcsObject.hpp"

namespace libecs
{

template< class T, typename SlotType_ >
const bool ConcretePropertySlot< T, SlotType_ >::isSetableMethod( const SetMethodPtr aSetMethodPtr )
{
    const SetMethodPtr aNullMethodPtr( &EcsObject::nullSet<SlotType> );
    return aSetMethodPtr != aNullMethodPtr;
}

template< class T, typename SlotType_ >
const bool ConcretePropertySlot< T, SlotType_ >::isGetableMethod( const GetMethodPtr aGetMethodPtr )
{
    const GetMethodPtr
        aNullMethodPtr( &EcsObject::nullGet<SlotType> );
    return aGetMethodPtr != aNullMethodPtr;
}

template< class T, typename SlotType_ >
typename ConcretePropertySlot< T, SlotType_ >::SetMethodPtr
ConcretePropertySlot< T, SlotType_ >::SetMethod( SetMethodPtr aSetMethodPtr )
{
    if( aSetMethodPtr == NULLPTR )
    {
        return &EcsObject::nullSet<SlotType>;
    }
    else
    {
        return aSetMethodPtr;
    }
}

template< class T, typename SlotType_ >
typename ConcretePropertySlot< T, SlotType_ >::GetMethodPtr
ConcretePropertySlot< T, SlotType_ >::GetMethod( GetMethodPtr aGetMethodPtr )
{
    if( aGetMethodPtr == NULLPTR )
    {
        return &EcsObject::nullGet<SlotType>;
    }
    else
    {
        return aGetMethodPtr;
    }
}

} // namespace libecs

#endif /* __LIBECS_CONCRETEPROPERTYSLOT_MEMBER_DEFINED */
