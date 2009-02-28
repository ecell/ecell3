//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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


#if !defined(__DATAPOINT_HPP)
#define __DATAPOINT_HPP

#include "libecs/Defs.hpp"
#include "libecs/Polymorph.hpp"

namespace libecs
{

class LongDataPoint;
class DataPoint;

class DataPoint
{
public:
    class EarlinessOrdering
    {
    public:
        bool operator()(const DataPoint& x, const DataPoint& y)
        {
            return x.getTime() < y.getTime();
        }
    };

    class LatenessOrdering
    {
    public:
        bool operator()(const DataPoint& x, const DataPoint& y)
        {
            return x.getTime() > y.getTime();
        }
    };

public:
    DataPoint()
        : theTime ( 0.0 ),
          theValue( 0.0 )
    {
        ; // do nothing
    }

    DataPoint( RealParam aTime, RealParam aValue )
        : theTime ( aTime ),
          theValue( aValue )     
    {
        ; //do nothing
    }

    ~DataPoint()
    {
        ; // do nothing
    }


    const Real getTime() const
    {
        return theTime;
    }


    const Real getValue() const
    {
        return theValue;
    }


    const Real getAvg() const
    {
        return theValue;
    }


    const Real getMin() const
    {
        return theValue;
    }


    const Real getMax() const
    {
        return theValue;
    }


    void setTime( RealParam aReal )
    {
        theTime = aReal;
    }


    void setValue( RealParam aReal )
    {
        theValue = aReal;
    }


    static const size_t getElementSize()
    {
        return sizeof( Real );
    }


    static const int getElementNumber()
    {
        return 2;
    }


    DataPointRef operator= ( LongDataPointCref aLongDataPoint );


    bool operator==( const DataPoint& that ) const
    {
        return theTime == that.theTime && theValue == that.theValue;
    }

protected:

    Real theTime;
    Real theValue;

};



class LongDataPoint: public DataPoint
{

public:
    LongDataPoint() //constructor with no arguments
        : theAvg( 0.0 ),
          theMin( 0.0 ),
          theMax( 0.0 )
    {
        ; // do nothing
    }


    LongDataPoint( RealParam aTime, RealParam aValue )//constructor with 2 args
        : DataPoint( aTime, aValue ),
          theAvg( aValue ),
          theMin( aValue ),
          theMax( aValue )
    {
        ; // do nothing
    }

    LongDataPoint( RealParam aTime, RealParam aValue, 
                   RealParam anAvg, RealParam aMax, 
                   RealParam aMin ) //constructor with 5 args
        : DataPoint( aTime, aValue ),
          theAvg( anAvg ),
          theMin( aMin ),
          theMax( aMax )
    {
        ; // do nothing
    }


    LongDataPoint( DataPointCref aDataPoint ) // constructor from DP2
        : DataPoint( aDataPoint ),
          theAvg( aDataPoint.getAvg() ),
          theMin( aDataPoint.getMin() ),
          theMax( aDataPoint.getMax() )
    {
        ; // do nothing
    }

    
    ~LongDataPoint()
    {
        ; // do nothing
    }

    
    const Real getTime() const
    {
        return theTime;
    }

    
    const Real getValue() const
    {
        return theValue;
    }

    
    const Real getAvg() const
    {
        return theAvg;
    }

    
    const Real getMin() const
    {
        return theMin;
    }


    const Real getMax() const
    {
        return theMax;
    }


    void setTime( RealParam aReal )
    {
        theTime = aReal;
    }


    void setValue( RealParam aReal )
    {
        theValue = aReal;
    }


    void setAvg( RealParam aReal )
    {
        theAvg = aReal;
    }


    void setMin( RealParam aReal )
    {
        theMin = aReal;
    }


    void setMax( RealParam aReal )
    {
        theMax = aReal;
    }


    static const size_t getElementSize()
    {
        return sizeof( Real );
    }


    static const int getElementNumber()
    {
        return 5;
    }

protected:
    Real theAvg;
    Real theMin;
    Real theMax;
};


inline DataPoint& DataPoint::operator=( LongDataPoint const& aLongDataPoint )
{
    setTime( aLongDataPoint.getTime() );
    setValue( aLongDataPoint.getValue() );
    return *this;
}


class DataPointAggregator
{
public:
    
    DataPointAggregator();
    
    DataPointAggregator( LongDataPointCref );
    
    
    ~DataPointAggregator();
    
    void aggregate( LongDataPointCref );
    
    LongDataPointCref getData();
    
    void beginNextPoint();
    
    LongDataPoint getLastPoint();
    
private:
    void store( LongDataPointCref );
    
    bool stockpile( LongDataPointRef, LongDataPointCref );
    void calculate( LongDataPointCref );
    void calculateMinMax( LongDataPointRef, LongDataPointCref );
    
    LongDataPoint theAccumulator;
    LongDataPoint theCollector;
    LongDataPoint thePreviousPoint;
};

} // namespace libecs

#endif /* __DATAPOINT_HPP */
