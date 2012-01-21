//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
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
// 24/03/2002

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "DataPoint.hpp"

namespace libecs
{
DataPointAggregator::DataPointAggregator()
    : theAccumulator( -1.0, 0.0 ),
      theCollector ( -1.0, 0.0 ),
      thePreviousPoint( 1.0, 0.0 )
{
    ; //do nothing
}


DataPointAggregator::DataPointAggregator( LongDataPoint const& aDataPoint )
{
    store( aDataPoint );
}


DataPointAggregator::~DataPointAggregator()
{
    ; // do nothing
}


void DataPointAggregator::store( LongDataPoint const& aDataPoint )
{
    theAccumulator = aDataPoint;
    thePreviousPoint = aDataPoint;
    theCollector.setTime( -1.0 );
}


bool DataPointAggregator::stockpile( LongDataPoint& aTarget, 
                                     LongDataPoint const& aNewPoint )
{
    //if target empty, simply store
    //return true
    if( aTarget.getTime() == -1.0 )
    {
        aTarget = aNewPoint;
        return true;
    }
    
    // if target not empty and time is the same
    // calculate MinMax, store Avg
    //return true
    if( aTarget.getTime() == aNewPoint.getTime() )
    {
        calculateMinMax( aTarget, aNewPoint );
        aTarget.setAvg( aNewPoint.getAvg() );
        aTarget.setValue( aNewPoint.getValue() );
        return true;
    }
    
    //if target time is below newtime
    //return false
    return false;
}


void DataPointAggregator::aggregate( LongDataPoint const& aNewPoint )
{
    // first try to put it into accumulator
    if ( ! stockpile( theAccumulator, aNewPoint ) )
    {
        // then try to put it into collector
        if (! stockpile( theCollector, aNewPoint ) )
        {
            // then calculate
            calculate( aNewPoint );
            theCollector = aNewPoint;
            calculateMinMax( theAccumulator, theCollector );
        }
        else
        {
            calculateMinMax( theAccumulator, theCollector );
        }
    }
}


LongDataPoint const& DataPointAggregator::getData()
{
    return theAccumulator;
}


inline void
DataPointAggregator::calculateMinMax( LongDataPoint& aTarget,
                                      LongDataPoint const& aNewPoint )
{
    // accu min
    if( aTarget.getMin() > aNewPoint.getMin() )
    {
        aTarget.setMin ( aNewPoint.getMin() );
    }
    
    // accu max
    if( aTarget.getMax() < aNewPoint.getMax() )
    {
        aTarget.setMax ( aNewPoint.getMax() );
    }
}


void DataPointAggregator::calculate( LongDataPoint const& aNewPoint )
{
    // accu avg
    theAccumulator.setAvg
        ( ( theCollector.getAvg() *
                ( aNewPoint.getTime() - theCollector.getTime() ) +
                theAccumulator.getAvg() * 
                ( theCollector.getTime() - theAccumulator.getTime() ) ) 
            / ( aNewPoint.getTime() - theAccumulator.getTime() ) );
}

void DataPointAggregator::beginNextPoint()
{
    store( theCollector );
}


LongDataPoint DataPointAggregator::getLastPoint()
{
    //if collector empty return Accu
    if (theCollector.getTime() == -1.0 )
    {
        return theAccumulator;
    }
    else
    {
        return theCollector;
    }
}

} // namespace libecs
