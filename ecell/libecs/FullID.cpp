//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2016 Keio University
//       Copyright (C) 2008-2016 RIKEN
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

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include <string>

#include "Util.hpp"
#include "Exceptions.hpp"

#include "FullID.hpp"

namespace libecs
{

///////////////////////    SystemPath

void SystemPath::parse( String const& systempathstring )
{
    if( systempathstring.empty() )
    {
        return;
    }

    String aString( systempathstring );
    eraseWhiteSpaces( aString );
    
    String::size_type aCompStart( 0 );
    
    // absolute path ( start with '/' )
    if( aString[ 0 ] == DELIMITER )
    {
        push_back( String( 1, DELIMITER ) );
        ++aCompStart;
    }

    for ( String::size_type aCompEnd; aCompStart < aString.size();
            aCompStart = aCompEnd + 1 )
    {
        aCompEnd = aString.find_first_of( DELIMITER, aCompStart );
        if ( aCompEnd == String::npos )
        {
            aCompEnd = aString.size();
        }

        String aComponent( aString.substr( aCompStart, aCompEnd - aCompStart ) );
        if ( aComponent == ".." )
        {
            if ( theComponents.size() == 1 &&
                    theComponents.front()[ 0 ] == DELIMITER )
            {
                THROW_EXCEPTION( BadSystemPath, 
                                 "Too many levels of retraction with .." );
            }
            else if ( theComponents.empty() || theComponents.back() == ".." )
            {
                theComponents.push_back( aComponent );
            }
            else 
            {
                theComponents.pop_back();
            }
        }
        else if ( !aComponent.empty() && aComponent != "." )
        {
            theComponents.push_back( aComponent );
        }
    }

    if ( !aString.empty() && theComponents.empty() )
    {
        theComponents.push_back( "." );
    }
}

String SystemPath::asString() const
{
    StringVector::const_iterator i = theComponents.begin();
    String aString;

    if( isAbsolute() )
    {
        if( theComponents.size() == 1 )
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
        if( theComponents.empty() )
        {
            return aString;
        }
        else
        {
            aString = *i;
        }
    }

    if( i == theComponents.end() )
    {
        return aString;
    }

    ++i;

    while( i != theComponents.end() )
    {
        aString += '/';
        aString += *i;
        ++i;
    }

    return aString;
}

void SystemPath::canonicalize()
{
    StringVector aNewPathComponents;

    for ( StringVector::const_iterator i( theComponents.begin() );
           i != theComponents.end(); ++i )
    {
        if ( *i == "." )
        {
            continue;
        }
        else if ( *i == ".." )
        {
            if ( aNewPathComponents.empty() )
            {
                break;
            }
            aNewPathComponents.pop_back();
        }
        else
        {
            aNewPathComponents.push_back( *i );
        }
    }

    theComponents.swap( aNewPathComponents );
}

SystemPath SystemPath::toRelative( SystemPath const& aBaseSystemPath ) const
{
    // 1. "" (empty) means Model itself, which is invalid for this method.
    // 2. Not absolute is invalid (not absolute implies not empty).
    if( ! isAbsolute() || isModel() )
    {
        return *this;
    }

    if( ! aBaseSystemPath.isAbsolute() || aBaseSystemPath.isModel() )
    {
        THROW_EXCEPTION( BadSystemPath, 
                         "[" + aBaseSystemPath.asString() +
                         "] is not an absolute SystemPath" );
    }

    SystemPath aThisPathCopy;
    SystemPath const* thisPath;

    if ( !isCanonicalized() )
    {
        aThisPathCopy = *this;
        aThisPathCopy.canonicalize();
        thisPath = &aThisPathCopy;
    }
    else
    {
        thisPath = this;
    }

    SystemPath aBaseSystemPathCopy;
    SystemPath const* aCanonicalizedBaseSystemPath;
    if ( !aBaseSystemPath.isCanonicalized() )
    {
        aCanonicalizedBaseSystemPath = &aBaseSystemPath;
    }
    else
    {
        aBaseSystemPathCopy = aBaseSystemPath;
        aBaseSystemPathCopy.canonicalize();
        aCanonicalizedBaseSystemPath = &aBaseSystemPathCopy;
    }

    SystemPath aRetval;
    StringVector::const_iterator j( thisPath->theComponents.begin() ),
                               je( thisPath->theComponents.end() );
    StringVector::const_iterator
        i( aCanonicalizedBaseSystemPath->theComponents.begin() ),
        ie( aCanonicalizedBaseSystemPath->theComponents.end() );

    while ( i != ie && j != je )
    {
        String const& aComp( *i );
        if ( aComp != *j )
        {
            break;
        }
        ++i, ++j;
    }
    if ( i != ie )
    {
        while ( i != ie )
        {
            aRetval.theComponents.push_back( ".." );
            ++i;
        }
    }
    std::copy( j, je, std::back_inserter( aRetval.theComponents ) );

    if ( aRetval.theComponents.empty() )
    {
        aRetval.theComponents.push_back( "." );
    }

    return aRetval; 
}


///////////////// FullID

void FullID::parse( String const& fullidstring )
{
    // empty FullID string is invalid
    if( fullidstring == "" )
    {
        THROW_EXCEPTION( BadID, "empty FullID string" );
    }

    String aString( fullidstring );
    eraseWhiteSpaces( aString );

    // ignore leading white spaces
    String::size_type aFieldStart( 0 );
    String::size_type aFieldEnd( aString.find_first_of( DELIMITER, aFieldStart ) );
    if( aFieldEnd == String::npos )
    {
        THROW_EXCEPTION( BadID, 
                         "no ':' in the FullID string [" + aString + "]" );
    }

    String aTypeString( aString.substr( aFieldStart, aFieldEnd - aFieldStart ) );
    theEntityType = EntityType( aTypeString );
    
    aFieldStart = aFieldEnd + 1;
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );
    if( aFieldEnd == String::npos )
    {
        THROW_EXCEPTION( BadID, "only one ':' in the FullID string [" 
                                + aString + "]" );
    }

    
    theSystemPath = SystemPath( aString.substr( aFieldStart, 
                                aFieldEnd - aFieldStart ) );
    
    aFieldStart = aFieldEnd + 1;

    // drop trailing string after extra ':'(if this is    FullPN),
    // or go to the end
    aFieldEnd = aString.find_first_of( DELIMITER, aFieldStart );

    theID = aString.substr( aFieldStart, aFieldEnd - aFieldStart );
}

String FullID::asString() const
{
    if ( theID.empty() )
    {
        return String( "(invalid)" );
    }

    return theEntityType.asString() + FullID::DELIMITER 
        + theSystemPath.asString() + FullID::DELIMITER + theID;
}

bool FullID::isValid() const
{
    bool aFlag( theSystemPath.isValid() );
    aFlag &= ! theID.empty();

    return aFlag;
}


///////////////// FullPN


FullPN::FullPN( String const& fullpropertynamestring )
    : theFullID( fullpropertynamestring )
{

    String::size_type aPosition( 0 );

    for( int i( 0 ) ; i < 3 ; ++i )
    {
        aPosition = fullpropertynamestring.
            find_first_of( FullID::DELIMITER, aPosition );
        if( aPosition == String::npos ) 
        {
            THROW_EXCEPTION( BadID, "not enough fields in FullPN string [" +
                                    fullpropertynamestring + "]" );
        }
        ++aPosition;
    }

    thePropertyName = fullpropertynamestring.substr( aPosition, String::npos );
    eraseWhiteSpaces( thePropertyName );
}

String FullPN::asString() const
{
    return theFullID.asString() + FullID::DELIMITER + thePropertyName;
}

bool FullPN::isValid() const
{
    return theFullID.isValid() & ! thePropertyName.empty();
}

} // namespace libecs
