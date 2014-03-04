//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
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

    DataPoint( Real aTime, Real aValue )
        : theTime ( aTime ),
          theValue( aValue )     
    {
        ; //do nothing
    }

    ~DataPoint()
    {
        ; // do nothing
    }


    Real getTime() const
    {
        return theTime;
    }


    Real getValue() const
    {
        return theValue;
    }


    Real getAvg() const
    {
        return theValue;
    }


    Real getMin() const
    {
        return theValue;
    }


    Real getMax() const
    {
        return theValue;
    }


    void setTime( Real aReal )
    {
        theTime = aReal;
    }


    void setValue( Real aReal )
    {
        theValue = aReal;
    }


    static size_t getElementSize()
    {
        return sizeof( Real );
    }


    static int getElementNumber()
    {
        return 2;
    }


    DataPoint& operator=( LongDataPoint const& aLongDataPoint );


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


    LongDataPoint( Real aTime, Real aValue )//constructor with 2 args
        : DataPoint( aTime, aValue ),
          theAvg( aValue ),
          theMin( aValue ),
          theMax( aValue )
    {
        ; // do nothing
    }

    LongDataPoint( Real aTime, Real aValue, 
                   Real anAvg, Real aMax, 
                   Real aMin ) //constructor with 5 args
        : DataPoint( aTime, aValue ),
          theAvg( anAvg ),
          theMin( aMin ),
          theMax( aMax )
    {
        ; // do nothing
    }


    LongDataPoint( DataPoint const& aDataPoint ) // constructor from DP2
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

    
    Real getTime() const
    {
        return theTime;
    }

    
    Real getValue() const
    {
        return theValue;
    }

    
    Real getAvg() const
    {
        return theAvg;
    }

    
    Real getMin() const
    {
        return theMin;
    }


    Real getMax() const
    {
        return theMax;
    }


    void setTime( Real aReal )
    {
        theTime = aReal;
    }


    void setValue( Real aReal )
    {
        theValue = aReal;
    }


    void setAvg( Real aReal )
    {
        theAvg = aReal;
    }


    void setMin( Real aReal )
    {
        theMin = aReal;
    }


    void setMax( Real aReal )
    {
        theMax = aReal;
    }


    static size_t getElementSize()
    {
        return sizeof( Real );
    }


    static int getElementNumber()
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
    
    DataPointAggregator( LongDataPoint const& );
    
    
    ~DataPointAggregator();
    
    void aggregate( LongDataPoint const& );
    
    LongDataPoint const& getData();
    
    void beginNextPoint();
    
    LongDataPoint getLastPoint();
    
private:
    void store( LongDataPoint const& );
    
    bool stockpile( LongDataPoint&, LongDataPoint const& );
    void calculate( LongDataPoint const& );
    void calculateMinMax( LongDataPoint&, LongDataPoint const& );
    
    LongDataPoint theAccumulator;
    LongDataPoint theCollector;
    LongDataPoint thePreviousPoint;
};

} // namespace libecs

#endif /* __DATAPOINT_HPP */
