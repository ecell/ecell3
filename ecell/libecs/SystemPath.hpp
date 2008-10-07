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
#include <functional>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/rbegin.hpp>
#include <boost/range/rend.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/iterator/reverse_iterator.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/compare.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "libecs.hpp"
#include "Exceptions.hpp"

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
    typedef StringList::reverse_iterator reverse_iterator;
    typedef StringList::const_reverse_iterator const_reverse_iterator;
    typedef StringList::value_type value_type;
    typedef StringList::pointer pointer;
    typedef StringList::reference reference;
    typedef StringList::size_type size_type;

public:
    SystemPath(): canonicalized_( 0 ) {}

    template< typename Trange_ >
    SystemPath( const Trange_& components )
        : canonicalized_( 0 )
    {
        ::std::copy(
                ::boost::const_begin( components ),
                ::boost::const_end( components ),
                ::std::back_inserter( components_) );
    }

    ~SystemPath()
    {
        delete canonicalized_;
    }

    const String asString() const;

    bool isEmpty() const
    {
        return components_.empty();
    }

    bool isRoot() const
    {
        const SystemPath& canonPath( toCanonical() );
        return canonPath.size() == 2 &&
                canonPath[ 0 ].empty() && canonPath[ 1 ].empty();
    }

    bool isAbsolute() const
    {
        return components_.empty() || components_.front().empty();
    }

    std::pair< const SystemPath, String> splitAtLast() const;

    size_type size() const
    {
        return components_.size();
    }

    String pop()
    {
        if ( components_.empty() )
            THROW_EXCEPTION( IllegalOperation, "No more components to pop" );

        String retval( components_.back() );
        if ( retval.empty() )
            components_.pop_back();
        components_.pop_back();
        delete canonicalized_, canonicalized_ = 0;
        return retval; 
    }

    template< typename Tcomps_ > 
    SystemPath& append( const Tcomps_& comps )
    {
        typedef typename ::boost::range_const_iterator< Tcomps_ >::type const_iterator;
        const_iterator i( ::boost::begin( comps ) ), end( ::boost::end( comps ) );

        if ( i == end )
        {
            return *this;
        }

        // is the specified path absolute and the original is not empty?
        if ( ( *i ).empty() )
        {
            if ( components_.empty() )
            {
                components_.push_back( *i );
            }
            ++i;

            String nextItem( *i );
            ++i;
            if ( i == end || !nextItem.empty() )
            {
                components_.push_back( nextItem );
            }
        }

        while ( i < end )
        {
            if ( !( *i ).empty() )
            {
                components_.push_back( *i );
            }
            ++i;
        }

        delete canonicalized_, canonicalized_ = 0;
        return *this;
    }

    template< typename TcharRange_ >
    SystemPath& append( const TcharRange_& pathRepr, bool )
    {
        ::boost::iterator_range<String::const_iterator> trimmedPathRepr(    
                ::std::find_if(
                    ::boost::begin( pathRepr ), ::boost::end( pathRepr ),
                    !::boost::is_space() ),
                ::std::find_if(
                    ::boost::rbegin( pathRepr ), ::boost::rend( pathRepr ),
                    !::boost::is_space() ).base() );

        if ( trimmedPathRepr.begin() >= trimmedPathRepr.end() )
        {
            return *this;
        }

        StringList comps;
        ::boost::algorithm::split(
                comps, trimmedPathRepr,
                ::std::bind2nd( ::std::equal_to< char >(), DELIMITER[ 0 ] ) );
        return append( comps );
    }

    SystemPath& append( const SystemPath& that )
    {
        return append( that.components_ );
    }

    SystemPath& append( const String& pathRepr )
    {
        return append( pathRepr, true );
    }

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

    const SystemPath& toCanonical() const;

    void canonicalize()
    {
        *this = toCanonical();
    }

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
        delete canonicalized_, canonicalized_ = 0;
        return components_.begin();
    }

    iterator end()
    {
        delete canonicalized_, canonicalized_ = 0;
        return components_.end();
    }

    const_reverse_iterator rbegin() const
    {
        return components_.rbegin();
    }

    const_reverse_iterator rend() const
    {
        return components_.rend();
    }

    reverse_iterator rbegin()
    {
        delete canonicalized_, canonicalized_ = 0;
        return components_.rbegin();
    }

    reverse_iterator rend()
    {
        delete canonicalized_, canonicalized_ = 0;
        return components_.rend();
    }

    template< typename TcharRange_ >
    static SystemPath parse( const TcharRange_& pathRepr )
    {
        return SystemPath().append( pathRepr, true );
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
        delete canonicalized_;
        canonicalized_ = rhs.canonicalized_ ?
            new SystemPath( *rhs.canonicalized_ ): 0;
        return *this;
    }

    void swap( SystemPath& rhs )
    {
        components_.swap( rhs.components_ );
        ::std::swap( canonicalized_, rhs.canonicalized_ );
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

    const String& operator[]( size_type idx ) const
    {
        return components_[ idx ];
    }

    String& operator[]( size_type idx )
    {
        delete canonicalized_, canonicalized_ = 0;
        return components_[ idx ];
    }

public:
    static const char DELIMITER[];
    static const char CURRENT[];
    static const char PARENT[];

protected:
    StringList components_;
    mutable SystemPath *canonicalized_;    
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
