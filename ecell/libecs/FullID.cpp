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
#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <string>

#include "Util.hpp"
#include "Exceptions.hpp"

#include "FullID.hpp"

namespace libecs
{

///////////////////////  SystemPath

SystemPath SystemPath::parse( const String& aString )
{
    if ( aString.empty() )
    {
        return SystemPath();
    }

    SystemPath retval;

    String::size_type start( 0 ), end( 0 );

    // absolute path ( start with '/' )
    if ( aString[0] == DELIMITER )
    {
        //insert(end(), String( 1, DELIMITER ) );
        retval.push_back( String( 1, DELIMITER ) );

        if ( aString.size() == 1 )
        {
            return retval;
        }

        ++start;
    }

    for ( ;; )
    {
        end = aString.find_first_of( DELIMITER, start );
        if ( end == String::npos )
        {
            retval.push_back( aString.substr( start ) );
            break;
        }
        else
        {
            retval.push_back( aString.substr( start, end - start ) );
        }
        start = end + 1;
    }

    return retval;
}

const String SystemPath::asString() const
{
    StringList::const_iterator i = begin();
    String aString;

    if ( isAbsolute() )
    {
        if ( size() == 1 )
        {
            return "/";
        }
        else
        {
            ; // do nothing
        }
    }
    else
    {
        // isAbsolute() == false implies that this can be empty
        if ( empty() )
        {
            return aString;
        }
        else
        {
            aString = *i;
        }
    }

    if ( i == end() ) {
        return aString;
    }

    ++i;

    while ( i != end() )
    {
        aString += '/';
        aString += *i;
        ++i;
    }

    return aString;
}

///////////////// FullID

FullID FullID::parse( const String& aString )
{
    // empty FullID string is invalid
    if ( aString.empty() )
    {
        THROW_EXCEPTION( BadFormat, "empty FullID string." );
    }

    // ignore leading white spaces
    String::size_type aFieldStart( 0 );
    String::size_type aFieldEnd( aString.find_first_of( DELIMITER,
                                 aFieldStart ) );
    if ( aFieldEnd == String::npos )
    {
        THROW_EXCEPTION( BadFormat,
                         "no ':' in the FullID string \"" + aString + "\"" );
    }

    const EntityType& entityType( EntityType::get(
                                      aString.substr( aFieldStart, aFieldEnd - aFieldStart ) ) );

    aFieldStart = aFieldEnd + 1;
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );
    if ( aFieldEnd == String::npos )
    {
        THROW_EXCEPTION( BadFormat,
                         "only one ':' in the FullID string \""
                         + aString + "\"" );
    }

    SystemPath systemPath( SystemPath::parse(
                               aString.substr( aFieldStart, aFieldEnd - aFieldStart ) ) );

    aFieldStart = aFieldEnd + 1;

    // drop trailing string after extra ':'(if this is  FullPN),
    // or go to the end
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );

    return FullID( entityType, systemPath,
                   aString.substr( aFieldStart, aFieldEnd - aFieldStart ) );
}

const String FullID::asString() const
{
    return theLocalID.getEntityType() + FullID::DELIMITER
           + theSystemPath.asString() + FullID::DELIMITER
           + theLocalID.getID();
}

///////////////// FullPN

FullPN FullPN::parse( const String& fullpropertynamestring )
{
    String::size_type aPosition( 0 );

    for ( int i( 0 ) ; i < 3 ; ++i )
    {
        aPosition = fullpropertynamestring.
                    find_first_of( FullID::DELIMITER, aPosition );
        if ( aPosition == String::npos )
        {
            THROW_EXCEPTION( BadFormat,
                             "not enough fields in FullPN string \"" +
                             fullpropertynamestring + "\"" );
        }
        ++aPosition;
    }

    return FullPN(
           FullID::parse(
               fullpropertynamestring.substr( 0, aPosition - 1 ) ),
           fullpropertynamestring.substr( aPosition ) );
}

const String FullPN::asString() const
{
    return theFullID.asString() + FullID::DELIMITER + thePropertyName;
}

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
