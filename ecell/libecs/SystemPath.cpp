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
#include <utility>

#include <boost/algorithm/string/join.hpp>

#include "Exceptions.hpp"
#include "SystemPath.hpp"

namespace libecs {

const char SystemPath::DELIMITER[] = "/";
const char SystemPath::CURRENT[] = ".";
const char SystemPath::PARENT[] = "..";


std::pair< const SystemPath, String > SystemPath::splitAtLast() const
{
    if ( components_.empty() )
    {
        THROW_EXCEPTION( BadFormat,
                "getParent() called against an empty SystemPath" );
    }
    SystemPath parentPath( *this );
    return std::make_pair( parentPath, parentPath.pop() );
}

const String SystemPath::asString() const
{
    return boost::algorithm::join( components_, DELIMITER );
}

SystemPath SystemPath::toAbsolute( const SystemPath& baseSystemPath ) const
{
    if ( !baseSystemPath.isAbsolute() )
    {
        THROW_EXCEPTION( ValueError,
                String( "Base system path is expected to be absolute, " )
                + baseSystemPath.asString() + " given." );
    }

    if ( isAbsolute() )
    {
        return *this;
    }

    return baseSystemPath.cat( *this );
}

SystemPath SystemPath::toRelative( const SystemPath& baseSystemPath ) const
{
    if ( !baseSystemPath.isAbsolute() )
    {
        THROW_EXCEPTION( ValueError,
                String( "Base system path is expected to be absolute, " )
                + baseSystemPath.asString() + " given." );
    }

    if ( !isAbsolute() )
    {
        return *this;
    }

    SystemPath retval;
    StringList::const_iterator i( baseSystemPath.components_.begin() ),
                               j( components_.begin() );
    while ( j != components_.end() )
    {
        if ( i == baseSystemPath.components_.end() ) {
            retval.components_.clear();
            break;
        }
       
        if ( *i != *j )
        {
            retval.components_.push_back( (*i).empty() ? CURRENT: PARENT );
            break;
        }
        ++i, ++j;
    }

    while ( i != baseSystemPath.components_.end() )
    {
        retval.components_.push_back( (*i).empty() ? CURRENT: PARENT );
        ++i;
    }

    ::std::copy( j, components_.end(),
            ::std::back_inserter( retval.components_ ) );

    return  retval;
}

const SystemPath& SystemPath::toCanonical() const
{
    if ( canonicalized_ )
    {
        return *canonicalized_;
    }

    SystemPath* retval = new SystemPath();

    StringList& components( retval->components_ );
    for ( StringList::const_iterator i( components_.begin() );
            i < components_.end(); ++i) {
        if ( *i == CURRENT )
        {
            continue;
        }
        else if ( *i == PARENT )
        {
            if ( components.size() > 0 &&
                    ( !components.front().empty() || components.size() > 1 ) )
            {
                components.pop_back(); 
            }
        }
        else
        {
            components.push_back( *i );
        }
    }

    if ( components.size() == 1 && components.front().empty() )
    {
        components.push_back( "" );
    }

    canonicalized_ = retval;

    return *retval;
}

} // namespace libecs

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
