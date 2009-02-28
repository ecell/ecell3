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

#ifndef __FULLID_HPP
#define __FULLID_HPP

#include "libecs/Defs.hpp"
#include "libecs/EntityType.hpp"

namespace libecs
{

/** 
   SystemPath 
*/
class LIBECS_API SystemPath : public StringList
{
public:
    explicit SystemPath( StringCref systempathstring = "" )
    {
        parse( systempathstring );
    }

    SystemPath( SystemPathCref systempath )
        : StringList( static_cast<StringList>( systempath ) )
    {
        ; // do nothing
    }

    ~SystemPath() {}

    String asString() const;

    /** @deprecated use asString() instead. */
    LIBECS_DEPRECATED String getString() const
    {
        return asString();
    }

    bool isAbsolute() const
    {
        return ( ( ( ! empty() ) && ( front()[0] == DELIMITER ) ) || empty() );
    }

    bool isValid() const
    {
        // FIXME: check '..'s and '.'s etc..
        return true;
    }

    operator String() const
    {
        return asString();
    }

protected:

    /**
       Standardize a SystemPath. 
       Reduce '..'s and remove extra white spaces.

       @return reference to the systempath
    */
    void standardize();

private:

    void parse( StringCref systempathstring );

public:

    static const char DELIMITER = '/';
};


/**
   FullID is an identifier of a unique Entiy in a cell model.
   The FullID consists of a EntityType, a SystemPath and an ID string.

   @see EntityType, SystemPath
*/
class FullID
{
public:
    FullID( const EntityType type,
            SystemPathCref systempath,
            StringCref id )
        : theEntityType( type ),
          theSystemPath( systempath ),
          theID( id )
    {
        ; // do nothing
    }

    explicit FullID( const EntityType type,
                     StringCref systempathstring,
                     StringCref id )
        : theEntityType( type ),
          theSystemPath( systempathstring ),
          theID( id )
    {
        ; // do nothing
    }


    FullID( StringCref fullidstring )
    {
        parse( fullidstring );
    }


    FullID( FullIDCref fullid )
        : theEntityType( fullid.getEntityType() ),
          theSystemPath( fullid.getSystemPath() ),
          theID( fullid.getID() )
    {
        ; // do nothing
    }


    ~FullID() {}


    const EntityType getEntityType() const 
    { 
        return theEntityType; 
    }


    SystemPathCref getSystemPath() const
    { 
        return theSystemPath; 
    }


    StringCref getID() const
    { 
        return theID;
    }


    void setEntityType( const EntityType type )
    {
        theEntityType = type;
    }


    void setSystemPath( SystemPathCref systempath ) 
    {
        theSystemPath = systempath;
    }


    void setID( StringCref id ) 
    {
        theID = id;
    }


    bool isValid() const;

    LIBECS_API String asString() const;

    /** @deprecated use asString() instead. */
    LIBECS_DEPRECATED String getString() const
    {
        return asString();
    }

    bool operator<( FullIDCref rhs ) const
    {
        // first look at the EntityType
        if( getEntityType() != rhs.getEntityType() )
        {
            return getEntityType() < rhs.getEntityType();
        }

        // then compare the SystemPaths
        // FIXME: should be faster is there is SystemPath::compare()
        if( getSystemPath() != rhs.getSystemPath() )
        {
            return getSystemPath() < rhs.getSystemPath();
        }

        // finally compare the ID strings
        return getID() < rhs.getID();
    }

    bool operator==( FullIDCref rhs ) const
    {
        // first look at the EntityType
        if( getEntityType() != rhs.getEntityType() )
        {
            return false;
        }

        // then compare the SystemPaths
        if( getSystemPath() != rhs.getSystemPath() )
        {
            return false;
        }

        // finally compare the ID strings
        return getID() == rhs.getID();
    }

    bool operator!=( FullIDCref rhs ) const
    {
        return ! operator==( rhs );
    }

    operator String() const
    {
        return asString();
    }

protected:
    LIBECS_API void parse( StringCref fullidstring );

private:
    FullID();

public:
    static const char DELIMITER = ':';

private:
    EntityType theEntityType;
    SystemPath theSystemPath;
    String     theID;

};

class FullPN
{
public:
    FullPN( const EntityType type, 
            SystemPathCref systempath,
            StringCref id,
            StringCref propertyname )
        : theFullID( type, systempath, id ),
          thePropertyName( propertyname )
    {
        ; // do nothing
    }

    FullPN( FullIDCref fullid, StringCref propertyname )
        : theFullID( fullid ),
          thePropertyName( propertyname )
    {
        ; // do nothing
    }

    FullPN( FullPNCref fullpn )
        : theFullID( fullpn.getFullID() ),
          thePropertyName( fullpn.getPropertyName() )
    {
        ; // do nothing
    }

    LIBECS_API FullPN( StringCref fullpropertynamestring );

    ~FullPN() 
    {
        ; // do nothing
    }

    FullIDCref getFullID() const
    {
        return theFullID;
    }

    const EntityType    getEntityType() const 
    { 
        return getFullID().getEntityType(); 
    }

    SystemPathCref getSystemPath() const
    { 
        return getFullID().getSystemPath();
    }

    StringCref getID() const
    { 
        return getFullID().getID();
    }

    StringCref getPropertyName() const
    {
        return thePropertyName;
    }

    void setEntityType( const EntityType type )
    {
        theFullID.setEntityType( type );
    }

    void setSystemPath( SystemPathCref systempath ) 
    {
        theFullID.setSystemPath( systempath );
    }

    void setID( StringCref id ) 
    {
        theFullID.setID( id );
    }

    void setPropertyName( StringCref propertyname )
    {
        thePropertyName = propertyname;
    }

    LIBECS_API String asString() const;

    /** @deprecated use asString() instead. */
    LIBECS_DEPRECATED String getString() const
    {
        return asString();
    }

    bool isValid() const;

    bool operator<( FullPNCref rhs ) const
    {
        if( getFullID() != rhs.getFullID() )
        {
            return getFullID() < rhs.getFullID();
        }

        return getPropertyName() < rhs.getPropertyName();
    }

    bool operator==( FullPNCref rhs ) const
    {
        if( getFullID() != rhs.getFullID() )
        {
            return false;
        }

        // finally compare the ID strings
        return getPropertyName() == rhs.getPropertyName();
    }

    bool operator!=( FullPNCref rhs ) const
    {
        return ! operator==( rhs );
    }

    operator String() const
    {
        return asString();
    }

private:
    FullID theFullID;
    String thePropertyName;
};

} // namespace libecs

#endif // __FULLID_HPP
