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
// written by Moriyoshi Koizumi <mozo@sfc.keio.ac.jp>
// E-Cell Project.
//

#ifndef __PARTITIONEDLIST_HPP
#define __PARTITIONEDLIST_HPP

#include <cstddef>
#include <stdexcept>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/range/iterator_range.hpp>

namespace libecs {

template< ::std::size_t Npart_offset_, typename Tcontainer_ >
class PartitionedList: public Tcontainer_
{
public:
    typedef PartitionedList self_type;
    typedef Tcontainer_ base_type;
    typedef ::std::size_t partition_index_type;
    typedef typename base_type::value_type value_type;
    typedef typename base_type::difference_type difference_type;
    typedef typename base_type::size_type size_type;
    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::reverse_iterator reverse_iterator;
    typedef typename base_type::const_reverse_iterator const_reverse_iterator;
    typedef typename ::boost::iterator_range<iterator> range;
    typedef typename ::boost::iterator_range<const_iterator> const_range;

public:
    void push_back( partition_index_type part, const value_type& item )
    {
        base_type::insert(
                base_type::begin() + part_offset_[ part + 1 ], item );
        for ( partition_index_type i( part + 1 ); i <= npartitions; ++i )
        {
            ++part_offset_[ i ];
        }
    }

    void push_back( const value_type& item )
    {
        push_back( npartitions - 1, item );
    }

    void insert( iterator pos_iter, const value_type& item )
    {
        partition_index_type pos( pos_iter - base_type::begin() );
        base_type::insert( pos_iter, item );
        partition_index_type i( 1 );
        while ( i <= npartitions )
        {
            if ( pos <= part_offset_[ i ] )
            {
                break;
            }
            ++i;
        }
        if ( i > npartitions )
        {
            return;
        }
        while ( i <= npartitions )
        {
            ++part_offset_[ i ];
            ++i;
        }
    }

    void insert( partition_index_type part, const value_type& item )
    {
        base_type::insert( base_type::begin() + part_offset_[ part ], item );
        for ( partition_index_type i( part + 1 ); i <= npartitions; ++i )
        {
            ++part_offset_[ i ];
        }
    }

    void insert( const value_type& item )
    {
        insert( base_type::begin(), item );
    }

    void erase( iterator first_iter, iterator second_iter )
    {
        using ::boost::lexical_cast;

        if ( first_iter >= base_type::end() || second_iter <= first_iter )
        {
            return;
        }

        if ( second_iter > base_type::end() )
        {
            second_iter = base_type::end();
        }

        size_type first_pos( first_iter - base_type::begin() ),
                  second_pos( second_iter - base_type::begin() );

        partition_index_type i( 1 );
        while ( i <= npartitions )
        {
            if ( first_pos < part_offset_[ i ] )
            {
                break;
            }
            ++i;
        }

        if ( part_offset_[ i ] < second_pos )
        {
            throw ::std::invalid_argument(
                "Specified range spans across partitions: "
                + lexical_cast< ::std::string >( first_pos )
                + "-" + lexical_cast< ::std::string >( second_pos ) );
        }


        base_type::erase( first_iter, second_iter );

        while ( i <= npartitions )
        {
            --part_offset_[ i ];
            ++i;
        }
    }

    void erase( iterator pos_iter )
    {
        erase( pos_iter, pos_iter + 1 );
    }

    void clear()
    {
        base_type::clear();
        for ( partition_index_type i( 0 ); i <= npartitions; ++i )
        {
            part_offset_[ i ] = 0;
        }
    }

    void partition( partition_index_type idx, size_type pos )
    {
        using ::boost::lexical_cast;
        if ( idx >= npartitions - 1 ) 
        {
            throw ::std::out_of_range(
                "partition index out of range: "
                + lexical_cast< ::std::string >( idx )
                + ">=" + lexical_cast< ::std::string >( npartitions - 1 ) );
        }

        if ( pos > base_type::size() )
        {
            throw ::std::out_of_range(
                "element index out of range: "
                + lexical_cast< ::std::string >( pos )
                + ">" + lexical_cast< ::std::string >( base_type::size() ) );
        }

        part_offset_[ idx + 1 ] = pos;
    }

    size_type partition( partition_index_type idx ) const
    {
        using ::boost::lexical_cast;
        if ( idx >= npartitions - 1 ) 
        {
            throw ::std::out_of_range(
                "partition index out of range: "
                + lexical_cast< ::std::string >( idx )
                + ">=" + lexical_cast< ::std::string >( npartitions - 1 ) );
        }

        return part_offset_[ idx + 1 ];
    }

    void partition( partition_index_type idx, iterator pos )
    {
        partition( idx, pos - base_type::begin() );
    }

    range partition_range( partition_index_type idx )
    {
        using ::boost::lexical_cast;
        if ( idx >= npartitions - 1 ) 
        {
            throw ::std::out_of_range(
                "partition index out of range: "
                + lexical_cast< ::std::string >( idx )
                + ">=" + lexical_cast< ::std::string >( npartitions - 1 ) );
        }

        return range( begin( idx ), end( idx ) );
    }

    const_range partition_range( partition_index_type idx ) const
    {
        using ::boost::lexical_cast;
        if ( idx >= npartitions - 1 ) 
        {
            throw ::std::out_of_range(
                "partition index out of range: "
                + lexical_cast< ::std::string >( idx )
                + ">=" + lexical_cast< ::std::string >( npartitions - 1 ) );
        }

        return const_range( begin( idx ), end( idx ) );
    }

    iterator begin( partition_index_type idx )
    {
        return base_type::begin() + part_offset_[ idx ];
    }

    iterator end( partition_index_type idx )
    {
        return base_type::begin() + part_offset_[ idx + 1 ];
    }

    const_iterator begin( partition_index_type idx ) const
    {
        return base_type::begin() + part_offset_[ idx ];
    }

    const_iterator end( partition_index_type idx ) const
    {
        return base_type::begin() + part_offset_[ idx + 1 ];
    }

    reverse_iterator rbegin( partition_index_type idx )
    {
        return base_type::rend() - part_offset_[ idx + 1 ];
    }

    reverse_iterator rend( partition_index_type idx )
    {
        return base_type::rend() - part_offset_[ idx ];
    }

    const_reverse_iterator rbegin( partition_index_type idx ) const
    {
        return base_type::rend() - part_offset_[ idx + 1 ];
    }

    const_reverse_iterator rend( partition_index_type idx ) const
    {
        return base_type::rend() - part_offset_[ idx ];
    }

    iterator begin()
    {
        return base_type::begin();
    }

    iterator end()
    {
        return base_type::end();
    }

    const_iterator begin() const
    {
        return base_type::begin();
    }

    const_iterator end() const
    {
        return base_type::end();
    }

    reverse_iterator rbegin()
    {
        return base_type::rbegin();
    }

    reverse_iterator rend()
    {
        return base_type::rend();
    }

    const_reverse_iterator rbegin() const
    {
        return base_type::rbegin();
    }

    const_reverse_iterator rend() const
    {
        return base_type::rend();
    }

    PartitionedList()
        : base_type()
    {
        for ( partition_index_type i = 0; i < ( npartitions + 1 ); ++i )
        {
            part_offset_[ i ] = 0;
        }
    }

    PartitionedList( const base_type& that )
        : base_type( that )
    {
        for ( partition_index_type i = 0; i < ( npartitions + 1 ); ++i )
        {
            part_offset_[ i ] = base_type::size();
        }
    }

public:
    static const  partition_index_type npartitions = Npart_offset_;
private:
    size_type part_offset_[ Npart_offset_ + 1 ];
};

} // namespace libecs

#endif /* __PARTITIONEDLIST_HPP */
