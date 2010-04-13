//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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

#ifndef __FULLID_HPP
#define __FULLID_HPP

#include <vector>
#include "libecs/Defs.hpp"
#include "libecs/EntityType.hpp"

namespace libecs
{

/** 
   SystemPath 
*/
class SystemPath
{
public:
    typedef std::vector< String > StringVector;
    typedef StringVector::const_iterator const_iterator;
    typedef StringVector::size_type size_type;

public:
    explicit SystemPath( String const& systempathstring )
        : isCanonicalized_( true )
    {
        parse( systempathstring );
    }

    SystemPath(): isCanonicalized_( true ) {}

    SystemPath( SystemPath const& that )
        : isCanonicalized_( that.isCanonicalized_  ),
          theComponents( that.theComponents )
    {
        ; // do nothing
    }

    ~SystemPath() {}

    LIBECS_API String asString() const;

    /** @deprecated use asString() instead. */
    LIBECS_DEPRECATED String getString() const
    {
        return asString();
    }

    bool operator==( SystemPath const& rhs ) const
    {
        return theComponents == rhs.theComponents;
    }

    bool operator!=( SystemPath const& rhs ) const
    {
        return !operator==( rhs );
    }

    bool operator<( SystemPath const& rhs ) const
    {
        return theComponents < rhs.theComponents;
    }

    bool operator>=( SystemPath const& rhs ) const
    {
        return !operator<( rhs );
    }

    bool operator>( SystemPath const& rhs ) const
    {
        return theComponents > rhs.theComponents;
    }

    bool operator<=( SystemPath const& rhs ) const
    {
        return !operator>( rhs );
    }

    SystemPath const& operator=( SystemPath const& rhs )
    {
        theComponents = rhs.theComponents;
        return *this;
    }

    void swap( SystemPath& that )
    {
        theComponents.swap( that.theComponents );
    }

    void push_back( String const& aComponent )
    {
        theComponents.push_back( aComponent );
        if ( aComponent == "." || aComponent == ".." )
        {
            isCanonicalized_ = false;
        }
    }

    void pop_back()
    {
        theComponents.pop_back();
    }

    const_iterator begin() const
    {
        return theComponents.begin();
    }

    const_iterator end() const
    {
        return theComponents.end();
    }

    size_type size() const
    {
        return theComponents.size();
    }

    bool isAbsolute() const
    {
        return ( ! theComponents.empty() &&
                theComponents.front()[0] == DELIMITER )
                || isModel();
    }

    bool isModel() const
    {
        return theComponents.empty();
    }

    LIBECS_API void canonicalize();

    LIBECS_API SystemPath toRelative( SystemPath const& aBaseSystemPath ) const;

    bool isValid() const
    {
        // FIXME: check '..'s and '.'s etc..
        return true;
    }

    bool isCanonicalized() const
    {
        return isCanonicalized_;
    }

    operator String() const
    {
        return asString();
    }

private:

    LIBECS_API void parse( String const& systempathstring );

private:

    bool isCanonicalized_;
    StringVector theComponents;

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
    FullID()
        : theEntityType( EntityType::NONE ),
          theSystemPath(),
          theID() {}

    FullID( EntityType const& type,
            SystemPath const& systempath,
            String const& id )
        : theEntityType( type ),
          theSystemPath( systempath ),
          theID( id )
    {
        ; // do nothing
    }

    explicit FullID( EntityType const& type,
                     String const& systempathstring,
                     String const& id )
        : theEntityType( type ),
          theSystemPath( systempathstring ),
          theID( id )
    {
        ; // do nothing
    }

    FullID( String const& fullidstring )
    {
        parse( fullidstring );
    }


    FullID( FullID const& fullid )
        : theEntityType( fullid.getEntityType() ),
          theSystemPath( fullid.getSystemPath() ),
          theID( fullid.getID() )
    {
        ; // do nothing
    }


    ~FullID() {}


    EntityType const& getEntityType() const 
    { 
        return theEntityType; 
    }


    SystemPath const& getSystemPath() const
    { 
        return theSystemPath; 
    }


    String const& getID() const
    { 
        return theID;
    }


    void setEntityType( EntityType const& type )
    {
        theEntityType = type;
    }


    void setSystemPath( SystemPath const& systempath ) 
    {
        theSystemPath = systempath;
    }


    void setID( String const& id ) 
    {
        theID = id;
    }


    LIBECS_API bool isValid() const;

    LIBECS_API String asString() const;

    /** @deprecated use asString() instead. */
    LIBECS_DEPRECATED String getString() const
    {
        return asString();
    }

    bool operator<( FullID const& rhs ) const
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

    bool operator==( FullID const& rhs ) const
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

    bool operator!=( FullID const& rhs ) const
    {
        return ! operator==( rhs );
    }

    operator String() const
    {
        return asString();
    }

protected:
    LIBECS_API void parse( String const& fullidstring );

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
            SystemPath const& systempath,
            String const& id,
            String const& propertyname )
        : theFullID( type, systempath, id ),
          thePropertyName( propertyname )
    {
        ; // do nothing
    }

    FullPN( FullID const& fullid, String const& propertyname )
        : theFullID( fullid ),
          thePropertyName( propertyname )
    {
        ; // do nothing
    }

    FullPN( FullPN const& fullpn )
        : theFullID( fullpn.getFullID() ),
          thePropertyName( fullpn.getPropertyName() )
    {
        ; // do nothing
    }

    LIBECS_API FullPN( String const& fullpropertynamestring );

    ~FullPN() 
    {
        ; // do nothing
    }

    FullID const& getFullID() const
    {
        return theFullID;
    }

    const EntityType    getEntityType() const 
    { 
        return getFullID().getEntityType(); 
    }

    SystemPath const& getSystemPath() const
    { 
        return getFullID().getSystemPath();
    }

    String const& getID() const
    { 
        return getFullID().getID();
    }

    String const& getPropertyName() const
    {
        return thePropertyName;
    }

    void setEntityType( const EntityType type )
    {
        theFullID.setEntityType( type );
    }

    void setSystemPath( SystemPath const& systempath ) 
    {
        theFullID.setSystemPath( systempath );
    }

    void setID( String const& id ) 
    {
        theFullID.setID( id );
    }

    void setPropertyName( String const& propertyname )
    {
        thePropertyName = propertyname;
    }

    LIBECS_API String asString() const;

    /** @deprecated use asString() instead. */
    LIBECS_DEPRECATED String getString() const
    {
        return asString();
    }

    LIBECS_API bool isValid() const;

    bool operator<( FullPN const& rhs ) const
    {
        if( getFullID() != rhs.getFullID() )
        {
            return getFullID() < rhs.getFullID();
        }

        return getPropertyName() < rhs.getPropertyName();
    }

    bool operator==( FullPN const& rhs ) const
    {
        if( getFullID() != rhs.getFullID() )
        {
            return false;
        }

        // finally compare the ID strings
        return getPropertyName() == rhs.getPropertyName();
    }

    bool operator!=( FullPN const& rhs ) const
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
