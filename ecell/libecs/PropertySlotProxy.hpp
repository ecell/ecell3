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

#ifndef __PROPERTYSLOTPROXY_HPP
#define __PROPERTYSLOTPROXY_HPP

#include <functional>

#include "libecs/libecs.hpp"
#include "libecs/Util.hpp"
#include "libecs/PropertySlot.hpp"
#include "libecs/convertTo.hpp"
#include "libecs/Polymorph.hpp"

/**
   @addtogroup property
   @ingroup libecs
   @{
*/
namespace libecs
{
class LIBECS_API PropertySlotProxy
{

public:

    PropertySlotProxy()
    {
        ; // do nothing
    }
    
    virtual ~PropertySlotProxy();

    virtual SET_METHOD( Polymorph, Polymorph ) = 0;
    virtual GET_METHOD( Polymorph, Polymorph ) = 0;

    virtual SET_METHOD( Real, Real ) = 0;
    virtual GET_METHOD( Real, Real ) = 0;

    virtual SET_METHOD( Integer, Integer ) = 0;
    virtual GET_METHOD( Integer, Integer ) = 0;

    virtual SET_METHOD( String, String ) = 0;
    virtual GET_METHOD( String, String ) = 0;
    
    virtual const bool isSetable() const = 0;
    virtual const bool isGetable() const = 0;

    template < typename Type >
    inline void set( typename Param<Type>::type aValue );

    template < typename Type >
    inline const Type get() const;

protected:
};


template <>
inline void 
PropertySlotProxy::set<Polymorph>( Param<Polymorph>::type aValue )
{
    setPolymorph( aValue );
}

template <>
inline void PropertySlotProxy::set<Real>( Param<Real>::type aValue )
{
    setReal( aValue );
}

template <>
inline void PropertySlotProxy::set<Integer>( Param<Integer>::type aValue )
{
    setInteger( aValue );
}

template <>
inline void PropertySlotProxy::set<String>( Param<String>::type aValue )
{
    setString( aValue );
}

template <>
inline const Polymorph PropertySlotProxy::get() const
{
    return getPolymorph();
}

template <>
inline const String PropertySlotProxy::get() const
{
    return getString();
}

template <>
inline const Real PropertySlotProxy::get() const
{
    return getReal();
}


template <>
inline const Integer PropertySlotProxy::get() const
{
    return getInteger();
}


template< class T >
class ConcretePropertySlotProxy: public PropertySlotProxy
{
public:
    typedef PropertySlot<T> PropertySlot_;
    DECLARE_TYPE( PropertySlot_, PropertySlot );

    DM_IF ConcretePropertySlotProxy( T& anObject, 
                                     PropertySlotCref aPropertySlot )
        : theObject( anObject ),
          thePropertySlot( aPropertySlot )
    {
        ; // do nothing
    }

    DM_IF virtual ~ConcretePropertySlotProxy()
    {
        ; // do nothing
    }


#define _PROPERTYSLOT_SETMETHOD( TYPE )\
    virtual SET_METHOD( TYPE, TYPE )\
    {\
        thePropertySlot.set ## TYPE( theObject, value );\
    }

#define _PROPERTYSLOT_GETMETHOD( TYPE )\
    virtual GET_METHOD( TYPE, TYPE )\
    {\
        return thePropertySlot.get ## TYPE( theObject );\
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


    DM_IF virtual const bool isSetable() const
    {
        return thePropertySlot.isSetable();
    }

    DM_IF virtual const bool isGetable() const
    {
        return thePropertySlot.isGetable();
    }

private:

    DM_IF ConcretePropertySlotProxy();

    T&                             theObject;
    PropertySlotCref    thePropertySlot;
};

} // namespace libecs

/** @} */

#endif /* __PROPERTYSLOT_HPP */
