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
// modified by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "Exceptions.hpp"
#include "PropertiedClass.hpp"
#include "PropertySlotProxy.hpp"

namespace libecs
{

PropertiedClass::~PropertiedClass()
{
    ; // do nothing
}

void PropertiedClass::startup()
{
}

void PropertiedClass::initialize()
{
}

void PropertiedClass::postInitialize()
{
}

void PropertiedClass::interrupt( TimeParam )
{
}

const String& PropertiedClass::asString() const
{
    return getPropertyInterface().getClassName();
}

void PropertiedClass::initializePropertyInterface( ::libecs::PropertyInterface& thePropertyInterface )
{
}

const PropertyInterface& PropertiedClass::getPropertyInterface() const
{
    return propertyInterface_;
}

const PropertySlot*
PropertiedClass::getPropertySlot( const String& propertyName ) const
{
    return getPropertyInterface().getPropertySlot( propertyName );
}

void
PropertiedClass::loadProperty( const String& propertyName,
                               const Polymorph& aValue )
{
    const PropertySlot* slot( getPropertySlot( propertyName ) );
    if ( slot )
    {
        slot->load( *this, aValue );
    }
    else
    {
        defaultSetProperty( propertyName, aValue );
    }
}

Polymorph
PropertiedClass::saveProperty( const String& propertyName ) const
{
    const PropertySlot* slot( getPropertySlot( propertyName ) );
    if ( slot )
    {
        return slot->save( *this );
    }
    else
    {
        return defaultGetProperty( propertyName );
    }
}

PropertySlotProxy
PropertiedClass::createPropertySlotProxy( const String& propertyName )
{
    const PropertySlot* aPropertySlot( getPropertySlot( propertyName ) );
    if ( !aPropertySlot ) {
        THROW_EXCEPTION( NoSlot,
                         getPropertyInterface().getClassName() +
                         String( ": No such property slot: " )
                         + propertyName
                       );
    }

    return PropertySlotProxy( this, aPropertySlot );
}


const Polymorph PropertiedClass::
defaultGetPropertyAttributes( const String& propertyName ) const
{
    THROW_EXCEPTION( NoSlot,
                     getPropertyInterface().getClassName() +
                     String( ": No property slot [" )
                     + propertyName + "].  Get property attributes failed." );
}

const Polymorph
PropertiedClass::defaultGetPropertyList() const
{
    PolymorphVector aVector;

    return aVector;
}

void PropertiedClass::defaultSetProperty( const String& propertyName,
        const Polymorph& aValue )
{
    THROW_EXCEPTION( NoSlot,
                     getPropertyInterface().getClassName() +
                     String( ": No property slot [" )
                     + propertyName + "].  Set property failed." );
}

const Polymorph
PropertiedClass::defaultGetProperty( const String& propertyName ) const
{
    THROW_EXCEPTION( NoSlot,
                     getPropertyInterface().getClassName() +
                     String( ": No property slot [" )
                     + propertyName + "].  Get property failed." );
}

// @internal
void PropertiedClass::nullLoad( Param<Polymorph>::type )
{
    THROW_EXCEPTION( IllegalOperation, "Not loadable." );
}

/// @internal
const Polymorph PropertiedClass::nullSave() const
{
    THROW_EXCEPTION( IllegalOperation, "Not savable." );
    return Polymorph();
}


ConcretePropertyInterface< PropertiedClass >
PropertiedClass::propertyInterface_(
    "PropertiedClass", PropertiedClassKind::NONE );

#define NULLSET_SPECIALIZATION_DEF( TYPE )\
template <> void PropertiedClass::nullSet<TYPE>( Param<TYPE>::type )\
{\
    THROW_EXCEPTION( IllegalOperation, "Not settable." ); \
}

NULLSET_SPECIALIZATION_DEF( Real );
NULLSET_SPECIALIZATION_DEF( Integer );
NULLSET_SPECIALIZATION_DEF( String );
NULLSET_SPECIALIZATION_DEF( Polymorph );

#define NULLGET_SPECIALIZATION_DEF( TYPE )\
template <> const TYPE PropertiedClass::nullGet<TYPE>() const\
{\
    THROW_EXCEPTION( IllegalOperation, "Not gettable." ); \
    return TYPE(); \
}

NULLGET_SPECIALIZATION_DEF( Real );
NULLGET_SPECIALIZATION_DEF( Integer );
NULLGET_SPECIALIZATION_DEF( String );
NULLGET_SPECIALIZATION_DEF( Polymorph );

} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
