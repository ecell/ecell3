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

#ifndef __FULLID_HPP
#define __FULLID_HPP

#include "libecs.hpp"
#include "EntityType.hpp"

namespace libecs {

/** @addtogroup identifier The FullID, FullPN and SystemPath.
 The FullID, FullPN and SystemPath.
 

 @ingroup libecs
 @{
 */

/** @file */


/**
 SystemPath
 */
class LIBECS_API SystemPath : protected StringList
{
public:
    SystemPath( const StringList& systempath )
            : StringList( systempath )
    {
        ; // do nothing
    }

    SystemPath()
            : StringList()
    {
    }

    ~SystemPath() {}

    const String asString() const;

    bool isAbsolute() const
    {
        return ( ( ( ! empty() ) && ( front()[0] == DELIMITER ) ) || empty() );
    }

    /**
       Normalize a SystemPath.
       Reduce '..'s and remove extra white spaces.

       @return reference to the systempath
    */
    SystemPath normalize();

    LIBECS_API static SystemPath parse( const String& systempathstring );

    bool operator==(const SystemPath& rhs) const
    {
        return static_cast<const StringList&>(*this) == static_cast<const StringList&>(rhs);
    }

    bool operator!=(const SystemPath& rhs) const
    {
        return static_cast<const StringList&>(*this) != static_cast<const StringList&>(rhs);
    }

    bool operator<(const SystemPath& rhs) const
    {
        return static_cast<const StringList&>(*this) < static_cast<const StringList&>(rhs);
    }

public:
    static const char DELIMITER = '/';
};

/**
   LocalID is an identifier that is unique within a System.
 */
class LocalID
{
public:
    LocalID( const EntityType& type,
             const String& id )
            :
            theEntityType( type ),
            theID( id )
    {
        ; // do nothing
    }

    ~LocalID() {}

    const EntityType  getEntityType() const
    {
        return theEntityType;
    }

    const String& getID() const
    {
        return theID;
    }

    bool operator<( const LocalID& rhs ) const
    {
        // first look at the EntityType
        if ( getEntityType() != rhs.getEntityType() )
        {
            return getEntityType() < rhs.getEntityType();
        }

        // finally compare the ID strings
        return getID() < rhs.getID();
    }

    bool operator==( const LocalID& rhs ) const
    {
        return getEntityType() == rhs.getEntityType() &&
                getID() == rhs.getID();
    }

    bool operator!=( const LocalID& rhs ) const
    {
        return ! operator==( rhs );
    }

private:
    LocalID();

private:
    const EntityType& theEntityType;
    const String        theID;
};

/**
   FullID is an identifier that specifies an unique Entity in a cell model.
   The FullID consists of a EntityType, a SystemPath and an ID string.

   @see EntityType, SystemPath
*/
class FullID
{
public:
    FullID( const LocalID& localID,
            const SystemPath& systempath )
            :
            theLocalID( localID ),
            theSystemPath( systempath )
    {
        ; // do nothing
    }

    FullID( const EntityType& type,
            const SystemPath& systempath,
            const String& id )
            :
            theLocalID( type, id ),
            theSystemPath( systempath )
    {
        ; // do nothing
    }

    ~FullID() {}

    const EntityType getEntityType() const
    {
        return theLocalID.getEntityType();
    }

    const SystemPath& getSystemPath() const
    {
        return theSystemPath;
    }

    const String& getID() const
    {
        return theLocalID.getEntityType();
    }

    const LocalID& getLocalID() const
    {
        return theLocalID;
    }

    LIBECS_API const String asString() const;

    bool operator<( const FullID& rhs ) const
    {
        if ( getSystemPath() != rhs.getSystemPath() )
        {
            return getSystemPath() < rhs.getSystemPath();
        }

        return getLocalID() < rhs.getLocalID();
    }

    bool operator==( const FullID& rhs ) const
    {
        return getSystemPath() == rhs.getSystemPath() &&
                getLocalID() == rhs.getLocalID();
    }

    bool operator!=( const FullID& rhs ) const
    {
        return ! operator==( rhs );
    }

    LIBECS_API static FullID parse( const String& fullidstring );

private:
    FullID();

public:

    static const char DELIMITER = ':';

private:
    const LocalID       theLocalID;
    const SystemPath    theSystemPath;
};

class FullPN
{
public:
    FullPN( const EntityType& type,
            const SystemPath& systempath,
            const String& id,
            const String& propertyname )
            :
            theFullID( type, systempath, id ),
            thePropertyName( propertyname )
    {
        ; // do nothing
    }

    FullPN( const FullID& fullid, const String& propertyname )
            :
            theFullID( fullid ),
            thePropertyName( propertyname )
    {
        ; // do nothing
    }

    FullPN( const FullPN& fullpn )
            :
            theFullID( fullpn.getFullID() ),
            thePropertyName( fullpn.getPropertyName() )
    {
        ; // do nothing
    }

    ~FullPN()
    {
        ; // do nothing
    }

    const FullID& getFullID() const
    {
        return theFullID;
    }

    const EntityType  getEntityType() const
    {
        return getFullID().getEntityType();
    }

    const SystemPath& getSystemPath() const
    {
        return getFullID().getSystemPath();
    }

    const String& getID() const
    {
        return getFullID().getID();
    }

    const String& getPropertyName() const
    {
        return thePropertyName;
    }

    LIBECS_API static FullPN parse( const String& that );

    LIBECS_API const String asString() const;

    bool operator<( const FullPN& rhs ) const
    {
        if ( getFullID() != rhs.getFullID() )
        {
            return getFullID() < rhs.getFullID();
        }

        return getPropertyName() < rhs.getPropertyName();
    }

    bool operator==( const FullPN& rhs ) const
    {
        if ( getFullID() != rhs.getFullID() )
        {
            return false;
        }

        // finally compare the ID strings
        return getPropertyName() == rhs.getPropertyName();
    }

    bool operator!=( const FullPN& rhs ) const
    {
        return ! operator==( rhs );
    }

private:
    FullID theFullID;
    String thePropertyName;
};

/** @} */ // identifier module

} // namespace libecs

#endif // __FULLID_HPP

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
