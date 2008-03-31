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
// written by Gabor Bereczki <gabor.bereczki@talk21.com>
// 25/03/2002

#ifndef __DATAPOINTAGGREGATOR_HPP
#define __DATAPOINTAGGREGATOR_HPP

namespace libecs {

template<typename Tdp_>
class DataPointAggregator
{
public:
    typedef Tdp_ DataPoint;

public:
    DataPointAggregator()
        : accumulator_( DataPoint::invalid ),
          last_( DataPoint::invalid ),
          min_( DataPoint::invalid ),
          max_( DataPoint::invalid )
    {
    }

    ~DataPointAggregator() {}

    void put( const DataPoint& newPoint )
    {
        if ( !accumulator_.isValid() )
        {
            accumulator_ = newPoint;
        }
        else
        {
            accumulator_.value = 
                ( accumulator_.value * (
                        last_.time - accumulator_.time )
                  + ( newPoint.value + last_.value ) / (
                        ( newPoint.time - last_.time ) ) / 2
                ) / ( newPoint.time - accumulator_.time );
        }

        if ( !min_.isValid() || min_.value > newPoint.value )
        {
            min_ = newPonit;
        }

        if ( !max_.isValid() || max_.value < newPoint.value )
        {
            max_ = newPoint;
        }

        last_ = newPoint;
    }

    const DataPoint& get() const
    {
        return accumulator_;
    }

    const DataPoint& getMin() const
    {
        return min_;
    }

    const DataPoint& getMax() const
    {
        return max_;
    }

    void reset()
    {
        accumulator_ = last_;
    }

    DataPoint& last() const
    {
        return last_;
    }
private:
    DataPoint accumulator_;
    DataPoint min_;
    DataPoint max_;
    DataPoint last_;
};

} // namespace libecs

#endif /* __DATAPOINTAGGREGATOR_HPP */
