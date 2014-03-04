//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

#ifndef __PROPERTYINTERFACE_HPP
#define __PROPERTYINTERFACE_HPP

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/iterator.hpp>
#include <boost/assert.hpp>

#include "dmtool/DynamicModuleInfo.hpp"

#include "libecs/Defs.hpp"
#include "libecs/AssocVector.h"
#include "libecs/PropertyAttributes.hpp"
#include "libecs/PropertySlot.hpp"
#include "libecs/PropertySlotProxy.hpp"


namespace libecs
{

class LIBECS_API PropertyInterfaceBase: public DynamicModuleInfo
{
public:
    typedef ::Loki::AssocVector< String, Polymorph, std::less<const String> >
            PolymorphAssocVector;
    typedef std::pair< PolymorphAssocVector::const_iterator,
                       PolymorphAssocVector::const_iterator >
            PolymorphAssocVectorCrange;
    typedef ::Loki::AssocVector< String, PropertySlotBase*,
                                 std::less<const String> > PropertySlotMap;
    typedef PropertySlotMap::const_iterator PropertySlotMapConstIterator;

private:
    template< typename Trange_ >
    class EntryIterator: public DynamicModuleInfo::EntryIterator
    {
    public:
        EntryIterator( const Trange_& aRange )
            : firstTime( true ),
              theEnd( boost::end( aRange ) ),
              theIter( boost::begin( aRange ) ) {}

        virtual ~EntryIterator() {}

        virtual bool next()
        {
            if ( theIter == theEnd )
                return false;
            if ( !firstTime )
            {
                ++theIter;
                if ( theIter == theEnd )
                    return false;
            }
            firstTime = false;
            return true;
        }

        virtual std::pair< String, const void* > current()
        {
            return std::pair< String, const void * >( theIter->first, &theIter->second );
        }

    private:
        bool firstTime;
        boost::range_iterator< PolymorphAssocVectorCrange >::type theEnd;
        boost::range_iterator< PolymorphAssocVectorCrange >::type theIter;
    };

public:
    virtual ~PropertyInterfaceBase()
    {
        std::for_each( thePropertySlotMap.begin(), thePropertySlotMap.end(),
                ComposeUnary( DeletePtr< PropertySlotBase >(),
                    SelectSecond< PropertySlotMap::value_type >() ) );
    }

    /**
       Get a PropertySlot by name.

       @param aPropertyName the name of the PropertySlot.

       @return a borrowed pointer to the PropertySlot with that name.
    */
    PropertySlotBase const* getPropertySlot( String const& aPropertyName ) const
    {
        PropertySlotMapConstIterator i( findPropertySlot( aPropertyName ) );

        if( i == thePropertySlotMap.end() )
        {
            throwNoSlot( aPropertyName );
        }

        return i->second;
    }


    std::vector< String > getPropertyList() const
    {
        std::vector< String > aVector;

        for( PropertySlotMapConstIterator i( thePropertySlotMap.begin() ); 
             i != thePropertySlotMap.end() ; ++i )
        {
            aVector.push_back( i->first );
        }

        return aVector;
    }

    
    void 
    registerPropertySlot( PropertySlotBase* aPropertySlotPtr )
    {
        String const& aName( aPropertySlotPtr->getName() );
        PropertySlotMapConstIterator i( findPropertySlot( aName ) );
        if ( i != thePropertySlotMap.end() )
        {
            // it already exists. take the latter one.
            delete ( *i ).second;
            thePropertySlotMap.erase( aName );
        }

        thePropertySlotMap.insert( std::make_pair( aName, aPropertySlotPtr ) );
    }


    const PropertyAttributes
    getPropertyAttributes( String const& aPropertyName ) const
    {
        PropertySlotMapConstIterator i( findPropertySlot( aPropertyName ) );

        if( i != thePropertySlotMap.end() )
        {
            PropertySlotBase const* aPropertySlotPtr( getPropertySlot( aPropertyName ) );
            
            return PropertyAttributes( *aPropertySlotPtr );
        }
        throwNoSlot( aPropertyName );
	  return PropertyAttributes();
    }


    PropertySlotMap const& getPropertySlotMap() const
    {
        return thePropertySlotMap;
    }

    void setInfoField( String const& aFieldName, Polymorph const& aFieldValue )
    {
        theInfoMap.insert( std::make_pair( aFieldName, aFieldValue ) );
    }

    virtual const void* getInfoField( String const& aFieldName ) const
    {
        PolymorphAssocVector::const_iterator i( theInfoMap.find( aFieldName ) );
        if( i == theInfoMap.end() )
            THROW_EXCEPTION( NoInfoField, "no such info field: " + aFieldName );
        return &i->second;
    }

    virtual DynamicModuleInfo::EntryIterator* getInfoFields() const
    {
        return createEntryIterator( theInfoMap.begin(), theInfoMap.end() );
    }

    String const& getClassName() const
    {
        return theClassName;
    }

    String const& getTypeName() const
    {
        return theTypeName;
    }

    void throwNoSlot( String const& aPropertyName ) const;
    void throwNoSlot( EcsObject const& obj, String const& aPropertyName ) const;
    void throwNotLoadable( EcsObject const& obj, String const& aPropertyName ) const;
    void throwNotSavable( EcsObject const& obj, String const& aPropertyName ) const;

protected:

    PropertyInterfaceBase( String const& aClassName, String aTypeName )
        : theClassName( aClassName ), theTypeName( aTypeName )
    {
        ; // do nothing
    }


    PropertySlotMapConstIterator 
    findPropertySlot( String const& aPropertyName ) const
    {
        return thePropertySlotMap.find( aPropertyName );
    }


    template< typename Titer_ >
    EntryIterator< std::pair< Titer_, Titer_ > >*
    createEntryIterator( const Titer_& begin, const Titer_& end ) const
    {
        return new EntryIterator< std::pair< Titer_, Titer_ > >( std::make_pair( begin, end ) );
    }

protected:

    PropertySlotMap    thePropertySlotMap;

    PolymorphAssocVector theInfoMap;

    String theClassName;

    String theTypeName;
};

template < class T >
class PropertyInterface
    : public PropertyInterfaceBase
{
public:
    PropertyInterface( String const& aClassName, String const& aTypeName )
        : PropertyInterfaceBase( aClassName, aTypeName )
    {
        T::initializePropertyInterface( this );
    }


    virtual ~PropertyInterface()
    {
    }


    std::vector< String > getPropertyList( const T& anObject ) const
    {
        std::vector< String > aVector1;
        // aVector.reserve( thePropertySlotMap.size() );
        
        for( PropertySlotMapConstIterator i( thePropertySlotMap.begin() ); 
             i != thePropertySlotMap.end() ; ++i )
        {
            aVector1.push_back( i->first );
        }

        const std::vector< String > aVector2( anObject.defaultGetPropertyList() );

        for( std::vector< String >::const_iterator i( aVector2.begin() );
             i != aVector2.end(); ++i )
        {
            aVector1.push_back( *i );
        }

        return aVector1;
    }

    
    PropertySlotProxy*
    createPropertySlotProxy( T& anObject,
                             String const& aPropertyName ) const
    {
        try
        {
            PropertySlotBase const* aPropertySlot( getPropertySlot( aPropertyName ) );
            return new ConcretePropertySlotProxy<T>(
                    anObject,
                    *static_cast< PropertySlot<T> const* >( aPropertySlot ) );
        }
        catch( NoSlot const& )
        {
            throwNoSlot( anObject, aPropertyName );
        }
        return 0; // never get here
    }


    /**
       Set a value of a property slot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName the name of the property.
       @param aValue the value to set as a Polymorph.
       @throw NoSlot 
    */
    void setProperty( T& anObject, String const& aPropertyName, 
                                        Polymorph const& aValue ) const
    {
        PropertySlotMapConstIterator aPropertySlotMapIterator(
                findPropertySlot( aPropertyName ) );
        
        if( aPropertySlotMapIterator != thePropertySlotMap.end() )
        {
            static_cast< PropertySlot< T > const* >( aPropertySlotMapIterator->second )->setPolymorph( anObject, aValue );
        }
        else
        {
            anObject.defaultSetProperty( aPropertyName, aValue );                    
        }
    }
    

    /**
       Get a property value from this object via a PropertySlot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName the name of the property.
       @return the value as a Polymorph.
       @throw NoSlot
    */
    const Polymorph getProperty( const T& anObject,
                                 String const& aPropertyName ) const
    {
        PropertySlotMapConstIterator 
            aPropertySlotMapIterator( findPropertySlot( aPropertyName ) );
        
        if( aPropertySlotMapIterator != thePropertySlotMap.end() )
        {
            return static_cast< PropertySlot< T > const* >( aPropertySlotMapIterator->second )->getPolymorph( anObject );
        }
        else
        {
            return anObject.defaultGetProperty( aPropertyName );
        }
    }


    void loadProperty( T& anObject, String const& aPropertyName, 
                       Polymorph const& aValue ) const
    {
        PropertySlotMapConstIterator 
            aPropertySlotMapIterator( findPropertySlot( aPropertyName ) );

        if( aPropertySlotMapIterator != thePropertySlotMap.end() )
        {
            PropertySlotBase const* aPropertySlotPtr( aPropertySlotMapIterator->second );

            if( aPropertySlotPtr->isLoadable() )
            {
                static_cast< PropertySlot< T > const* >( aPropertySlotPtr )->loadPolymorph( anObject, aValue );
            }
            else
            {
                throwNotLoadable( anObject, aPropertyName );
            }
        }
        else
        {
            anObject.defaultSetProperty( aPropertyName, aValue );
        }
    }
    

    const Polymorph
    saveProperty( const T& anObject, String const& aPropertyName ) const
    {
        PropertySlotMapConstIterator 
            aPropertySlotMapIterator( findPropertySlot( aPropertyName ) );

        if( aPropertySlotMapIterator != thePropertySlotMap.end() )
        {
            PropertySlotBase const* aPropertySlotPtr( aPropertySlotMapIterator->second );
            if( aPropertySlotPtr->isSavable() )
            {
                return static_cast< PropertySlot< T > const* >( aPropertySlotPtr )->savePolymorph( anObject );
            }
            else
            {
                throwNotSavable( anObject, aPropertyName );
            }
        }
        else
        {
            return anObject.defaultGetProperty( aPropertyName );
        }
        return Polymorph(); // never get here
    }


    const PropertyAttributes
    getPropertyAttributes( const T& anObject, String const& aPropertyName ) const
    {
        PropertySlotMapConstIterator i( findPropertySlot( aPropertyName ) );

        if( i != thePropertySlotMap.end() )
        {
            PropertySlotBase const* aPropertySlotPtr( getPropertySlot( aPropertyName ) );
            
            return PropertyAttributes( *aPropertySlotPtr );
        }
        else
        {
            return anObject.defaultGetPropertyAttributes( aPropertyName );
        }
    }
};

} // namespace libecs

#endif /* __PROPERTYINTERFACE_HPP */
