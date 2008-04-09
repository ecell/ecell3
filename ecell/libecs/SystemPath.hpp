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

#ifndef SYSTEMPATH_HPP
#define SYSTEMPATH_HPP

#include <vector>
#include <algorithm>
#include "libecs.hpp"

/** @addtogroup identifier The FullID, FullPN and SystemPath.
 The FullID, FullPN and SystemPath.
 

 @ingroup libecs
 @{
 */

/** @file */


namespace libecs {

/**
 SystemPath
 */
class LIBECS_API SystemPath
{
protected:
    typedef ::std::vector< String > StringList;

public:
    typedef StringList::iterator iterator;
    typedef StringList::const_iterator const_iterator;
    typedef StringList::value_type value_type;
    typedef StringList::pointer pointer;
    typedef StringList::reference reference;


public:
    SystemPath() {}

    template< typename Trange_ >
    SystemPath( const Trange_& components )
    {
        ::std::copy( components.begin(), components.end(),
                ::std::back_inserter( components_) );
    }

    ~SystemPath() {}

    const String asString() const;

    bool isEmpty() const
    {
        return components_.empty();
    }

    bool isRoot() const
    {
        return components_.size() == 2 &&
                components_[ 0 ].empty() && components_[ 1 ].empty();
    }

    bool isAbsolute() const
    {
        return ( !components_.empty() && components_[ 0 ].empty() );
    }

    std::pair< const SystemPath, String> splitAtLast() const;

    String pop()
    {
        String retval( components_.back() );
        components_.pop_back();
        return retval; 
    }

    template< typename Tcomps_ > 
    SystemPath& append( const Tcomps_& comps )
    {
        typename Tcomps_::const_iterator i( comps.begin() ), end( comps.end() );

        if ( i == end )
        {
            return *this;
        }

        // is the specified path absolute and the original is not empty?
        if ( (*i).empty() && !components_.empty() )
        {
            ++i;
        }

        while ( i != end )
        {
            components_.push_back( *i );
        }

        return *this;
    }

    SystemPath& append( const SystemPath& that )
    {
        return append( that.components_ );
    }

    SystemPath& append( const String& pathRepr );

    SystemPath cat( const String& pathRepr ) const
    {
        return SystemPath( *this ).append( pathRepr );
    }

    SystemPath cat( const SystemPath& pathRepr ) const
    {
        return cat( pathRepr.components_ );
    }

    SystemPath toAbsolute( const SystemPath& baseSystemPath ) const;

    SystemPath toRelative( const SystemPath& baseSystemPath ) const;

    const_iterator begin() const
    {
        return components_.begin();
    }

    const_iterator end() const
    {
        return components_.end();
    }

    iterator begin()
    {
        return components_.begin();
    }

    iterator end()
    {
        return components_.end();
    }

    static SystemPath parse( const String& pathRepr )
    {
        return SystemPath().append( pathRepr );
    }

    SystemPath operator+( const String& rhs ) const
    {
        return cat( rhs );
    }

    SystemPath operator+( const SystemPath& rhs ) const
    {
        return cat( rhs );
    }

    SystemPath operator+=( const String& rhs )
    {
        return append( rhs );
    }

    SystemPath operator+=( const SystemPath& rhs )
    {
        return append( rhs );
    }

    SystemPath& operator=( const SystemPath& rhs )
    {
        components_ = rhs.components_;
        return *this;
    }

    void swap( SystemPath& rhs )
    {
        components_.swap( rhs.components_ );
    }

    bool operator==(const SystemPath& rhs) const
    {
        return components_ == rhs.components_;
    }

    bool operator!=(const SystemPath& rhs) const
    {
        return components_ != rhs.components_;
    }

    bool operator<(const SystemPath& rhs) const
    {
        return components_ < rhs.components_;
    }

public:
    static const char DELIMITER[];
    static const char CURRENT[];
    static const char PARENT[];

protected:
    std::vector<String> components_;
};

inline String operator+( const String& lhs, const SystemPath& rhs )
{
    return lhs + rhs.asString();
}

} // namespace libecs

/** @} */ // identifier module

#endif // SYSTEMPATH_HPP

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
