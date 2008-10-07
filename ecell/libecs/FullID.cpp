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

#include <algorithm>
#include <functional>
#include <boost/range/iterator_range.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/finder.hpp>
#include <boost/algorithm/string/split.hpp>

#include "Exceptions.hpp"
#include "FullID.hpp"

namespace libecs {

FullID FullID::parse( const String& fullIDRepr )
{
    typedef boost::iterator_range< String::const_iterator > StringRange;
    StringRange trimmedFullIDRepr(
            std::find_if( fullIDRepr.begin(), fullIDRepr.end(),
                !boost::is_space() ),
            std::find_if( fullIDRepr.rbegin(), fullIDRepr.rend(),
                !boost::is_space() ).base() );

    // empty FullID string is invalid
    if ( trimmedFullIDRepr.begin() >= trimmedFullIDRepr.end() )
    {
        THROW_EXCEPTION( BadFormat, "empty FullID string." );
    }

    std::vector< StringRange > i;
    boost::iter_split( i, trimmedFullIDRepr, boost::token_finder(
            std::bind2nd( std::equal_to< char >(), ID_DELIMITER ) ) );
    if ( i.size() < 3 )
    {
        THROW_EXCEPTION( BadFormat,
                         "Too few ':' in the FullID string \""
                         + fullIDRepr + "\"" );
    }
    else if ( i.size() > 3 )
    {
        THROW_EXCEPTION( BadFormat,
                         "Too many ':' in the FullID string \""
                         + fullIDRepr + "\"" );
    }

    try
    {
        const EntityType& entityType( EntityType::get( i[ 0 ] ) );
        SystemPath systemPath( SystemPath::parse( i[ 1 ] ) );
        return FullID( entityType, systemPath, String( i[ 2 ].begin(), i[ 2 ].end() ) );
    }
    catch ( const Exception& e )
    {
        THROW_EXCEPTION( BadFormat, e.what() );
    }
}

const String FullID::asString() const
{
    return static_cast< const String& >( localID_.getEntityType() )
           + ID_DELIMITER
           + systemPath_.asString() + ID_DELIMITER
           + localID_.getID();
}

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
