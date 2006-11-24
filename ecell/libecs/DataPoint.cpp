//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2001-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
//END_HEADER
//
// written by Gabor Bereczki <gabor.bereczki@talk21.com>
// 24/03/2002




#include <stdio.h>
#include "DataPoint.hpp"

namespace libecs
{
  

  DataPointRef DataPoint::operator= ( LongDataPointCref aLongDataPoint )
  {
    setTime( aLongDataPoint.getTime() );
    setValue ( aLongDataPoint.getValue() );
    return *this;
  }

  
  DataPointAggregator::DataPointAggregator()
    :
    theAccumulator( -1.0, 0.0 ),
    theCollector ( -1.0, 0.0 ),
    thePreviousPoint( 1.0, 0.0 )
  {
    ; //do nothing
  }
  

  DataPointAggregator::DataPointAggregator( LongDataPointCref aDataPoint )
  {
    store( aDataPoint );
  }
  

  DataPointAggregator::~DataPointAggregator()
  {
    ; // do nothing
  }
  
  
  void DataPointAggregator::store( LongDataPointCref aDataPoint )
  {
    theAccumulator = aDataPoint;
    thePreviousPoint = aDataPoint;
    theCollector.setTime( -1.0 );
  }
  
  
  bool DataPointAggregator::stockpile( LongDataPointRef aTarget, 
				       LongDataPointCref aNewPoint )
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
  
  
  void DataPointAggregator::aggregate( LongDataPointCref aNewPoint )
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
  
  
  LongDataPointCref DataPointAggregator::getData()
  {
    return theAccumulator;
  }
  
  
  void DataPointAggregator::calculateMinMax( LongDataPointRef aTarget,  
					     LongDataPointCref aNewPoint)
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
  
  
  void DataPointAggregator::calculate( LongDataPointCref aNewPoint )
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
    //	theAccumulator = thePreviousPoint;
    //	thePreviousPoint = theCollector;
    
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


#if defined(STANDALONE_TEST)

using namespace libecs;
void agr( Real aTime, Real aValue, DataPointAggregator* dpa)
{
  LongDataPoint pa;
  DataPoint dp;
  dp.setTime(aTime);
  dp.setValue(aValue);
  dpa->aggregate(dp);
  pa=dpa->getData();
  printf("aggregating time %f, value %f, results:  avg %f\n",  aTime,aValue, pa.getAvg() );

}

int main()
{
  DataPointAggregator dpa;

  agr(3,4,&dpa);
  agr(5,6,&dpa);
  agr(7,8,&dpa);
  agr(9,6,&dpa);
  agr(11,4,&dpa);
  agr(13,6,&dpa);
  
}

#endif /* End of test script*/
