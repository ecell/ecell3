//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2001-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
//
//END_HEADER
//
// written by Gabor Bereczki <gabor.bereczki@talk21.com>
// 24/03/2002





#include "DataPoint.hpp"

namespace libecs
{

  
    DataPoint& DataPoint::operator = ( const DataPointLong& dpt5 )
	{

		setTime( dpt5.getTime() );
		setValue ( dpt5.getValue() );
	}


	DataPointAggregator::DataPointAggregator()
	:
	Accumulator( -1.0, 0.0 ),
	Collector ( -1.0, 0.0 ),
	PreviousPoint( 1.0, 0.0 )

	{
		; //do nothing
	}

	DataPointAggregator::DataPointAggregator( const DataPointLong& dpt )

	{
	store( dpt );
	}

	DataPointAggregator::~DataPointAggregator(){}


	void DataPointAggregator::store( const DataPointLong& dpt )
	{
	Accumulator = dpt;
	PreviousPoint = dpt;
	Collector.setTime( -1.0 );
	}


	bool DataPointAggregator::stockpile( DataPointLong& Target, const DataPointLong& NewPoint )
	{
	//if target empty, simply store
	//return true
	if ( Target.getTime() == -1.0 )
	{
		Target = NewPoint;
		return true;
	}

	// if target not empty and time is the same
	// calculate MinMax, store Avg
	//return true
	if ( Target.getTime() == NewPoint.getTime() )
	{
		calculateMinMax( Target, NewPoint );
		Target.setAvg( NewPoint.getAvg() );
		Target.setValue( NewPoint.getValue() );
		return true;
	}

	//if target time is below newtime
	//return false
	else
	{
		return false;
	}

	}


	void DataPointAggregator::aggregate( const DataPointLong& NewPoint )
	{
	// first try to put it into accumulator
	if ( ! stockpile( Accumulator, NewPoint ) )
	{

		// then try to put it into collector
		if (! stockpile( Collector, NewPoint ) )
		{
			// then calculate
			calculate( NewPoint );
			Collector = NewPoint;
			calculateMinMax( Accumulator, Collector );

		
		}
		else
		{
			calculateMinMax( Accumulator, Collector );
		}
	}

	}


	const DataPointLong& DataPointAggregator::getData()
	{

	// return Accumulator

	return Accumulator;
	}


	void DataPointAggregator::calculateMinMax( DataPointLong& Target, const DataPointLong& NewPoint)
	{
	// accu min


	if ( Target.getMin() > NewPoint.getMin() )
		{
		Target.setMin ( NewPoint.getMin() );
		}

	// accu max
	if ( Target.getMax() < NewPoint.getMax() )
		{

		Target.setMax ( NewPoint.getMax() );
		}

	}


	void DataPointAggregator::calculate( const DataPointLong& NewPoint )
	{

	// accu avg
	Accumulator.setAvg ( ( Collector.getAvg() *
		( NewPoint.getTime() - Collector.getTime() ) +
		Accumulator.getAvg() * ( Collector.getTime() -
		Accumulator.getTime() ) ) /
		( NewPoint.getTime() - Accumulator.getTime() ) );
	}

	void DataPointAggregator::beginNextPoint()
	{
//	Accumulator = PreviousPoint;
//	PreviousPoint = Collector;
	
	store( Collector );
	}


	DataPointLong DataPointAggregator::getLastPoint()
	{
	//if collector empty return Accu
	if (Collector.getTime() == -1.0 )
	{
		return Accumulator;
	}
	else
	{
		return Collector;
	}
	

	}



} // namespace libecs


