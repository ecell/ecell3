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
// but WITHOUT ANY WARRANTY; without even the implied warranty of // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
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

#ifndef __PROPERTYINTERFACE_HPP
#define __PROPERTYINTERFACE_HPP

#include <boost/assert.hpp>

#include "libecs.hpp"
#include "AssocVector.h"
#include "PropertySlot.hpp"
#include "PropertiedClassKind.hpp"

/**
   @addtogroup property The Inter-object Communication.
   The Interobject Communication.
   @{
 */
/** @file */
namespace libecs {

class LIBECS_API PropertyInterface
{
public:
    DECLARE_UNORDERED_MAP( String, const PropertySlot*, DEFAULT_HASHER( String ), PropertySlotMap );
    DECLARE_UNORDERED_MAP( String, Polymorph, DEFAULT_HASHER( String ), InfoMap );

public:
    ~PropertyInterface()
    {
        for( PropertySlotMap::const_iterator i( thePropertySlotMap.begin() );
                i != thePropertySlotMap.end() ; ++i )
        {
            delete i->second;
        }
    }

    const String& getClassName() const
    {
        return theClassName;
    }

    const PropertiedClassKind& getKind() const
    {
        return theKind;
    }

    const PropertySlot* getPropertySlot( const String& aPropertyName ) const
    {
        PropertySlotMap::const_iterator i(
                findPropertySlot( aPropertyName ) );
        if ( i == thePropertySlotMap.end() )
            return 0;
        return i->second;
    }

    /**
       get Field from info map
    */
    const Polymorph& getInfoField( const String& aFieldName )
    {
        return theInfoMap[ aFieldName ];
    }

    /**
      set Info field
      if info field key begins with "Property_" then append PropertyName to "PropertyList" infofield
     
    */
    void setInfoField( const String& aFieldName, const Polymorph& aValue )
    {
        theInfoMap[ aFieldName ] = aValue;
    }


    void
    registerPropertySlot( PropertySlot* aPropertySlotPtr )
    {
        const String& aName( aPropertySlotPtr->getName() );
        if ( findPropertySlot( aName ) != thePropertySlotMap.end() )
        {
            // it already exists. take the latter one.
            delete thePropertySlotMap[ aName ];
            thePropertySlotMap.erase( aName );
        }

        thePropertySlotMap[ aName ] = aPropertySlotPtr;
    }

protected:

    PropertyInterface(
            const String& className,
            const PropertiedClassKind& kind )
        : theClassName( className ), theKind( kind )
    {
        ; // do nothing
    }

    static void throwNoSlot( const String& aClassName, const String& aPropertyName );

    static void throwNotLoadable( const String& aClassName,
                                  const PropertiedClass& aClassName,
                                  const String& aPropertyName );
    static void throwNotSavable( const String& aClassName,
                                 const PropertiedClass& aClassName,
                                 const String& aPropertyName );
protected:
    PropertySlotMap::const_iterator
    findPropertySlot( const String& aPropertyName ) const
    {
        return thePropertySlotMap.find( aPropertyName );
    }

protected:
    PropertySlotMap thePropertySlotMap;
    InfoMap theInfoMap;
    const PropertiedClassKind& theKind;
    const String theClassName;
};


template < class T >
class ConcretePropertyInterface: public T::_LIBECS_BASE_CLASS_::ConcretePropertyInterface
{
public:
    typedef T Owner;

    ConcretePropertyInterface(
            const String& className,
            const PropertiedClassKind& kind )
        : PropertyInterface( className, kind )
    {
        T::initializePropertyInterface( *const_cast< ConcretePropertyInterface* >( this ) );
    }

    ~ConcretePropertyInterface()
    {
    }
};

} // namespace libecs

/*@}*/


#endif /* __PROPERTYINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
