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

#ifndef __RANGECOMPOSITOR_HPP
#define __RANGECOMPOSITOR_HPP

#include <cstddef>
#include <stdexcept>
#include <iterator>
#include <boost/assert.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/iterator_range.hpp>

namespace libecs {

namespace detail
{
    template<typename Tptr_, typename Tvalue_>
    struct retval_constness_check
    {
        typedef Tvalue_& type;
    };

    template<typename Tptr_, typename Tvalue_>
    struct retval_constness_check<Tptr_ const*, Tvalue_>
    {
        typedef const Tvalue_ type;
    };
}

template<typename Trange_>
class RangeCompositor
{
public:
    class iterator;
    friend class iterator;
    typedef RangeCompositor self_type;
    typedef Trange_ range_type;
    typedef typename range_type::iterator each_iterator_type;
    typedef typename range_type::value_type value_type;
    typedef typename range_type::difference_type difference_type;
    typedef typename range_type::size_type size_type;
    typedef typename each_iterator_type::pointer pointer;
    typedef typename each_iterator_type::reference reference;
    typedef const typename each_iterator_type::reference const_reference;

protected:
    class iterator_info
    {
    public:
        iterator_info( RangeCompositor& impl,
                ::std::size_t range_idx, const each_iterator_type& iter )
            : impl_( impl ), range_idx_( range_idx ), iter_( iter )
        {
        }

        iterator_info( RangeCompositor& impl, ::std::size_t range_idx )
            : impl_( impl ), range_idx_( range_idx ),
              iter_( impl.range( range_idx ).begin() )
        {
        }

        bool operator<( const iterator_info& rhs ) const
        {
            return ( range_idx_ < rhs.range_idx_ )
                || ( range_idx_ == rhs.range_idx_ &&
                    iter_ < rhs.iter_ );
        }

        bool operator>( const iterator_info& rhs ) const
        {
            return ( range_idx_ > rhs.range_idx_ )
                || ( range_idx_ == rhs.range_idx_ &&
                    iter_ > rhs.iter_ );
        }

        bool operator==( const iterator_info& rhs ) const
        {
            return ( range_idx_ == rhs.range_idx_ && iter_ == rhs.iter_ );
        }

        bool operator!=( const iterator_info& rhs ) const
        {
            return !operator==( rhs );
        }

        bool operator>=( const iterator_info& rhs ) const
        {
            return !operator<( rhs );
        }

        bool operator<=( const iterator_info& rhs ) const
        {
            return !operator>( rhs );
        }

        reference operator*() const
        {
            return *iter_;
        }

        reference operator[]( difference_type off ) const
        {
            return *(*this + off);
        }

        iterator_info operator+( difference_type off ) const
        {
            difference_type o( off );
            iterator_info retval( *this );
            range_type range( impl_.range( retval.range_idx_ ) );
            for (;;)
            {
                retval.iter_ += o;
                if ( retval.iter_ < range.begin() )
                {
                    if ( retval.range_idx_ == 0 )
                    {
                        throw ::std::out_of_range("offset out of range: "
                                + boost::lexical_cast< ::std::string>( off ) );
                    }
                    o = retval.iter_ - range.begin();
                    range = impl_.range( --retval.range_idx_ );
                    retval.iter_ = range.end();
                }
                else if ( retval.iter_ >= range.end() )
                {
                    if ( retval.range_idx_ >= impl_.range_count() )
                    {
                        throw ::std::out_of_range("offset out of range: "
                                + boost::lexical_cast< ::std::string>( off ) );
                    }
                    o = retval.iter_ - range.end();
                    range = impl_.range( ++retval.range_idx_ );
                    retval.iter_ = range.begin();
                }
                else
                {
                    break;
                }
            }
            return retval;
        }

        difference_type operator-( const iterator_info& rhs ) const
        {
            difference_type retval = 0;
            each_iterator_type i( iter_ );
            ::std::size_t ridx( range_idx_ );
            range_type range( impl_.range( ridx ) );

            if ( rhs.range_idx_ > range_idx_ )
            {
                do
                {
                    retval -= range.end() - i;
                    ++ridx;
                    BOOST_ASSERT( ridx < impl_.range_count() );
                    range = impl_.range( ridx );
                    i = range.begin();
                } while ( rhs.range_idx_ > ridx );
                retval -= rhs.iter_ - i;
            }
            else if ( rhs.range_idx_ < range_idx_ )
            {
                do
                {
                    retval += i - range.begin();
                    --ridx;
                    BOOST_ASSERT( ridx >= 0 );
                    range = impl_.range( ridx );
                    i = range.end();
                } while ( rhs.range_idx_ < ridx );
                retval += i - rhs.iter_;
            }
            else
            {
                retval = i - rhs.iter_;
            }
            return retval;
        }
    public:
        RangeCompositor& impl_;
        ::std::size_t range_idx_;
        each_iterator_type iter_;
    };

public:
    class iterator: public iterator_info
    {
        friend class RangeCompositor;
    public:
        typedef typename RangeCompositor::value_type value_type;
        typedef typename RangeCompositor::difference_type difference_type;
        typedef typename RangeCompositor::pointer pointer;
        typedef typename RangeCompositor::reference reference;
        typedef ::std::bidirectional_iterator_tag iterator_cateory;

    public:
        const iterator& operator++()
        {
            ++iterator_info::iter_;
            if ( iterator_info::iter_ >= end_ )
            {
                if ( iterator_info::range_idx_ < iterator_info::impl_.range_count() - 1 )
                {
                    ++iterator_info::range_idx_;
                    iterator_info::iter_ = begin_ = iterator_info::impl_.range(
                            iterator_info::range_idx_ ).begin();
                    end_ = iterator_info::impl_.range( iterator_info::range_idx_ ).end();
                }
            }

            return *this;
        }

        const iterator& operator++(int)
        {
            iterator retval( *this, iterator_info::iter_ );

            ++iterator_info::iter_;
            if ( iterator_info::iter_ >= iterator_info::end_ )
            {
                if ( iterator_info::range_idx_ < iterator_info::impl_.range_count() - 1 )
                {
                    ++iterator_info::range_idx_;
                    iterator_info::iter_ = begin_ = iterator_info::impl_.range(
                            iterator_info::range_idx_ ).begin();
                    end_ = iterator_info::impl_.range( iterator_info::range_idx_ ).end();
                }
            }

            return retval;
        }

        const iterator& operator--()
        {
            if ( iterator_info::iter_ == begin_ )
            {
                if ( iterator_info::range_idx_ > 0 )
                {
                    --iterator_info::range_idx_;
                    iterator_info::iter_ = ( end_ = iterator_info::impl_.range(
                            iterator_info::range_idx_ ).end() );
                    begin_ = iterator_info::impl_.range( iterator_info::range_idx_ ).begin();
                }
            }

            --iterator_info::iter_;

            return *this;
        }

        const iterator& operator--(int)
        {
            iterator retval( *this, iterator_info::iter_ );

            if ( iterator_info::iter_ == begin_ )
            {
                if ( iterator_info::range_idx_ > 0 )
                {
                    --iterator_info::range_idx_;
                    iterator_info::iter_ = ( end_ = iterator_info::impl_.range(
                            iterator_info::range_idx_ ).end() );
                    begin_ = iterator_info::impl_.range( iterator_info::range_idx_ ).begin();
                }
            }

            --iterator_info::iter_;

            return retval;
        }

        reference operator[]( difference_type off )
        {
            if ( begin_ <= iterator_info::iter_ + off  &&
                    iterator_info::iter_ + off < end_ )
            {
                return *( iterator_info::iter_ + off );
            }
            return iterator_info::operator[]( off );
        }

    protected:
        iterator( const iterator_info& info )
            : iterator_info( info ),
              begin_( info.impl_.range( info.range_idx_ ).begin() ),
              end_( info.impl_.range( info.range_idx_ ).end() )
        {
            BOOST_ASSERT( begin_ <= iterator_info::iter_ );
            BOOST_ASSERT( iterator_info::iter_ <= end_ );
        }

        iterator( const iterator& proto, const each_iterator_type& iter )
            : iterator_info( proto.impl_, proto.range_idx_, iter ),
              begin_( proto.begin_ ), end_( proto.end_ )
        {
            BOOST_ASSERT( begin_ <= iterator_info::iter_ );
            BOOST_ASSERT( iterator_info::iter_ <= end_ );
        }

    protected:
        each_iterator_type begin_, end_;
    };

public:
    RangeCompositor( range_type* ranges, ::std::size_t num_ranges )
        : ranges_( ranges ), num_ranges_( num_ranges )
    {
    }

    range_type& range( ::std::size_t idx )
    {
        return ranges_[ idx ];
    }

    const range_type& range( ::std::size_t idx ) const
    {
        return ranges_[ idx ];
    }

    ::std::size_t range_count()
    {
        return num_ranges_;
    }

    range_type& range_front()
    {
        return range( 0 );
    }

    range_type& range_back()
    {
        return range( range_count() - 1 );
    }

    iterator begin()
    {
        return iterator( iterator_info( *this, 0 ) );
    }

    iterator end()
    {
        return iterator( iterator_info( *this,
                range_count() - 1, range_back().end() ) );
    }

    size_type size() const 
    {
        ::std::size_t retval = 0;

        for ( ::std::size_t i( 0 ); i < num_ranges_; ++i )
        {
            retval += static_cast< size_type >(
                    ranges_[ i ].end() - ranges_[ i ].begin() );
        }

        return retval;
    }

    reference operator[]( size_type idx )
    {
        return begin()[ idx ];
    }

private:
    range_type* ranges_;
    ::std::size_t num_ranges_;
};

} // namespace libecs

#endif /* __RANGECOMPOSITOR_HPP */
