//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
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

/** @addtogroup property
  
@ingroup libecs
@{
*/

/** @file */

#ifndef __LIBECS_PROPERTYSLOT_DEFINED
#define __LIBECS_PROPERTYSLOT_DEFINED

#include <functional>

#include "libecs.hpp"
#include "Converters.hpp"
#include "Polymorph.hpp"
#include "PropertyType.hpp"

namespace libecs {

class PropertiedClass;

class PropertySlot
{
public:
    PropertySlot( const String& name, const PropertyType& type )
        : theName( name ), theType( type )
    {
        ; // do nothing
    }

    virtual ~PropertySlot()
    {
        ; // do nothing
    }

public:
    const String& getName() const 
    {
        return theName;
    }

    const PropertyType& getType() const
    {
        return theType;
    }

    virtual bool isSetable() const
    {
        return true;
    }

    virtual bool isGetable() const
    {
        return true;
    }

    virtual bool isLoadable() const
    {
        return true;
    }

    virtual bool isSavable() const
    {
        return true;
    }

    template < typename Type >
    void set( PropertiedClass& anObject,
            typename Param<Type>::type aValue ) const
    {
    }

    template < typename Type >
    const Type get( const PropertiedClass& anObject ) const
    {
    }

    virtual void load( PropertiedClass& anObject,
        Param<Polymorph>::type aValue ) const
    {
        setPolymorph( anObject, aValue );
    }

    virtual const Polymorph save( const PropertiedClass& anObject ) const
    {
        return getPolymorph( anObject );
    }

protected:
    void setPolymorph(
        PropertiedClass& anObject,
        Param<Polymorph>::type aValue ) const;

    const Polymorph getPolymorph(
        const PropertiedClass& anObject ) const;

    virtual void setInteger(
        PropertiedClass& anObject,
        Param<Integer>::type aValue ) const = 0;
    virtual const Integer getInteger(
        const PropertiedClass& anObject ) const = 0;

    virtual void setReal(
        PropertiedClass& anObject,
        Param<Real>::type aValue ) const = 0;
    virtual const Real getReal(
        const PropertiedClass& anObject ) const = 0;

    virtual void setString(
        PropertiedClass& anObject,
        Param<String>::type aValue ) const = 0;
    virtual const String getString(
        const PropertiedClass& anObject ) const = 0;

protected:
    const String theName;
    const PropertyType& theType;
};

template<> inline void
PropertySlot::set<Integer>( PropertiedClass& anObject,
        Param<Integer>::type aValue ) const
{
    setInteger( anObject, aValue );
}

template<> inline const Integer
PropertySlot::get<Integer>( const PropertiedClass& anObject ) const
{
    return getInteger( anObject );
}

template<> inline void
PropertySlot::set<Real>( PropertiedClass& anObject,
        Param<Real>::type aValue ) const
{
    setReal( anObject, aValue );
}

template<> inline const Real
PropertySlot::get<Real>( const PropertiedClass& anObject ) const
{
    return getReal( anObject );
}

template<> inline void
PropertySlot::set<String>( PropertiedClass& anObject,
        Param<String>::type aValue ) const
{
    setString( anObject, aValue );
}

template<> inline const String
PropertySlot::get<String>( const PropertiedClass& anObject ) const
{
    return getString( anObject );
}

template<> inline void
PropertySlot::set<Polymorph>( PropertiedClass& anObject,
        Param<Polymorph>::type aValue ) const
{
    setPolymorph( anObject, aValue );
}

template<> inline const Polymorph
PropertySlot::get<Polymorph>( const PropertiedClass& anObject ) const
{
    return getPolymorph( anObject );
}

} // namespace libecs
#endif /* __LIBECS_PROPERTYSLOT_DEFINED */

#ifndef __LIBECS_CONCRETEPROPERTYSLOT_DEFINED
#define __LIBECS_CONCRETEPROPERTYSLOT_DEFINED
namespace libecs {

template< class T, typename SlotType_ >
class ConcretePropertySlot: public PropertySlot
{
public:
    DECLARE_TYPE( SlotType_, SlotType );

    typedef void ( T::* SetMethod )( typename Param<SlotType>::type );
    typedef const SlotType ( T::* GetMethod )() const;
    typedef void ( T::* LoadMethod )( typename Param<Polymorph>::type );
    typedef const Polymorph ( T::* SaveMethod )() const;

    ConcretePropertySlot(
        const String& name,
        const SetMethod aSetMethodPtr,
        const GetMethod aGetMethodPtr,
        const LoadMethod aLoadMethodPtr = 0,
        const SaveMethod aSaveMethodPtr = 0 )
        : PropertySlot( name, Type2PropertyType< SlotType >::value ),
          theSetMethodPtr( aSetMethodPtr ),
          theGetMethodPtr(  aGetMethodPtr ),
          theLoadMethodPtr( aLoadMethodPtr ),
          theSaveMethodPtr( aSaveMethodPtr )
    {
        ; // do nothing
    }

    virtual ~ConcretePropertySlot()
    {
        ; // do nothing
    }

    virtual bool isSetable() const;

    virtual bool isGetable() const;

    virtual bool isLoadable() const;

    virtual bool isSavable() const;

protected:
    virtual void load( PropertiedClass& anObject, const Polymorph& aValue ) const
    {
        if ( !theLoadMethodPtr )
        {
            ( dynamic_cast<T&>(anObject).*theSetMethodPtr )(
                    convertTo< SlotType >( aValue ) );
            return;
        }

        ( dynamic_cast<T&>(anObject).*theLoadMethodPtr )( aValue );
    }

    virtual const Polymorph save( const PropertiedClass& anObject ) const
    {
        if ( !theSaveMethodPtr )
        {
            return convertTo< Polymorph >(
                ( dynamic_cast<const T&>(anObject).*theGetMethodPtr )() );
        }
        return ( dynamic_cast<const T&>(anObject).*theSaveMethodPtr )();
    }

    virtual void setInteger(
        PropertiedClass& anObject,
        Param<Integer>::type aValue ) const
    {
        ( dynamic_cast<T&>(anObject).*theSetMethodPtr )(
                convertTo< SlotType >( aValue ) );
    }

    virtual const Integer getInteger(
        const PropertiedClass& anObject ) const
    {
        return convertTo< Integer >(
            ( dynamic_cast<const T&>(anObject).*theGetMethodPtr )() );
    }

    virtual void setReal(
        PropertiedClass& anObject,
        Param<Real>::type aValue ) const
    {
        ( dynamic_cast<T&>(anObject).*theSetMethodPtr )(
                convertTo< SlotType >( aValue ) );
    }

    virtual const Real getReal(
        const PropertiedClass& anObject ) const
    {
        return convertTo< Real >(
            ( dynamic_cast<const T&>(anObject).*theGetMethodPtr )() );
    }

    virtual void setString(
        PropertiedClass& anObject,
        Param<String>::type aValue ) const
    {
        ( dynamic_cast<T&>(anObject).*theSetMethodPtr )(
                convertTo< SlotType >( aValue ) );
    }

    virtual const String getString(
        const PropertiedClass& anObject ) const
    {
        return convertTo< String >(
            ( dynamic_cast<const T&>(anObject).*theGetMethodPtr )() );
    }


protected:
    const SetMethod theSetMethodPtr;
    const GetMethod theGetMethodPtr;
    const LoadMethod theLoadMethodPtr;
    const SaveMethod theSaveMethodPtr;
};

} // namespace libecs

#include "PropertiedClass.hpp"

namespace libecs {

template< class T, typename SlotType_ >
bool ConcretePropertySlot<T, SlotType_>::isSetable() const
{
    return theSetMethodPtr != static_cast< SetMethod >(
            &PropertiedClass::template nullSet<SlotType> );
}

template< class T, typename SlotType_ >
bool ConcretePropertySlot<T, SlotType_>::isGetable() const
{
    return theGetMethodPtr != static_cast< GetMethod >(
            &PropertiedClass::template nullGet<SlotType> );
}

template< class T, typename SlotType_ >
bool ConcretePropertySlot<T, SlotType_>::isLoadable() const
{
    return theLoadMethodPtr != reinterpret_cast< LoadMethod >(
            &PropertiedClass::nullLoad );
}

template< class T, typename SlotType_ >
bool ConcretePropertySlot<T, SlotType_>::isSavable() const
{
    return theSaveMethodPtr != reinterpret_cast< SaveMethod >(
            &PropertiedClass::nullSave );
}

} // namespace libecs
#endif /* __LIBECS_CONCRETEPROPERTYSLOT_DEFINED */
/*@}*/
