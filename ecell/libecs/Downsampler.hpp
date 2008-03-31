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

#ifndef __DOWNSAMPLER_HPP
#define __DOWNSAMPLER_HPP

#include <cstddef>

namespace libecs {

template<typename Tdpi_, typename Tagg_>
class Downsampler
{
public:
    typedef Tagg_ Integrator;
    typedef Tdpi_::value_type value_type;
    typedef ::std::size_t size_type;

    struct AggregatedDataPoint: public value_type
    {
        AggregatedDataPoint(
                typename value_type::Time _time,
                typename value_type::Value _value,
                typename value_type::Value _min,
                typename value_type::Value _max )
            : value_type( _time, _value ), min( _min ), max( _max ) {} 

        value_type::Value min;
        value_type::Value max;
    };

public:
    Downsampler( const Tdpi_& begin, const Tdpi_& end, size_type interval )
        : begin_( begin ), end_( end ), interval_( interval ), iter_( begin ),
          consumed_( false )
    {
    }

    const Downsampler& operator++()
    {
        consume();
        consumed_ = false;
    }

    AggregatedDataPoint operator*() const
    {
        consume();
        return AggregatedDataPoint(
                agg_.get().time, agg_.get().value,
                agg_.getMin(), agg_.getMax() );
    }

private:
    void consume()
    {
        if ( !consumed_ )
        {
            for ( size_type i( 0 ); i < interval_; ++i ) {
                if ( iter_ == end_ )
                    break;
                agg_.put( *iter_ );
                ++iter_;
            }
            consumed_ = true;
        }
    }

private:
    Tagg_ agg_;
    Tdpi_ iter_;
    Tdpi_ begin_;
    Tdpi_ end_;
    bool consumed_;
    size_type interval_;
};

} // namespace libecs

#endif /* __DOWNSAMPLER_HPP */
