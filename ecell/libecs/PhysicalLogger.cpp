//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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
//

#include "config.h" 
#include "PhysicalLogger.hpp"
//#include <stdio.h>

namespace libecs
{

  PhysicalLogger::PhysicalLogger() 
    : 
    theVector(),
    anEmptyDataPoint()
  {
    ; // do nothing
  }

  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time ) const
  {
    PhysicalLoggerIterator iterator( ( start + end ) / 2 );
    PhysicalLoggerIterator i_start( start );
    PhysicalLoggerIterator i_end( end );
    
    if ( start > end )
      {
	i_start=end;
	i_end=start;
      }

    while( ( i_start + 1 ) < i_end )
      {
	if ( theVector[ iterator ].getTime() <= time )
	  {
	    i_start = iterator;
	  }
	else 
	  {
	    i_end = iterator;
	  }

	iterator = ( i_start + i_end ) / 2;
	
      }
    
    if ( theVector[ i_end ].getTime() == time )
	{
	    i_start=i_end;
	}
    
    return i_start;
  }

  PhysicalLoggerIterator 
  PhysicalLogger::upper_bound( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time ) const

  {
    PhysicalLoggerIterator result( lower_bound( start, end, time ) );

    if ( ( result < size() - 1 ) && ( theVector [ result ].getTime() != time ) )
      {
	++result;
      }

    return result;
  }

  PhysicalLoggerIterator 
  PhysicalLogger::next_index( PhysicalLoggerIteratorCref start ) const

  {
    PhysicalLoggerIterator result( start );

    if ( result < size() - 1 ) 
      {
	++result;
      }

    return result;
  }

  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound_linear_backwards( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time ) const
    {
    PhysicalLoggerIterator i_start( start );
    PhysicalLoggerIterator i_end( end );
    Real aTime;

    if ( start > end )
      {
	i_start=end;
	i_end=start;
      }
    PhysicalLoggerIterator iterator( i_end );
    
    while ( (iterator>i_start) && ( theVector [ iterator ].getTime() > time ) )
	{
	iterator--;
	}
    return iterator;
    }

  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound_linear_estimate( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time,
			       RealCref time_per_step ) const
    {
    //if time_per_step is zero fall back to stepwise linear search
    if ( time_per_step == 0 )
    {
    return lower_bound_linear( start, end, time);
    }
    Real theStartTime( theVector[start].getTime() );
    PhysicalLoggerIterator iterator;

    iterator = static_cast<PhysicalLoggerIterator> ( (time - theStartTime ) 
					/ time_per_step ) + start;
    if ( iterator > end ) { iterator = end;}
    if ( theVector [iterator].getTime() < time )
	{
	    return lower_bound_linear( iterator, end, time);
	}
	
    if ( theVector [iterator].getTime() > time )
	{
	    return lower_bound_linear_backwards( start, iterator, time );
	}
    
    return iterator;
    }

  PhysicalLoggerIterator 
  PhysicalLogger::upper_bound_linear_estimate( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time,
			       RealCref time_per_step ) const
    {
    PhysicalLoggerIterator result( lower_bound_linear_estimate( start, end, 
							time, time_per_step ) );

    if ( ( result < size() - 1 ) && ( theVector [ result ].getTime() != time ) )
      {
	++result;
      }

    return result;
    
    }
    
  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound_linear( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time ) const
  {
    PhysicalLoggerIterator i_start( start );
    PhysicalLoggerIterator i_end( end );
    Real aTime;

    if ( start > end )
      {
	i_start=end;
	i_end=start;
      }
    PhysicalLoggerIterator iterator( i_start );
    PhysicalLoggerIterator return_value( i_start );

	
    do
      {
	  aTime =theVector[ iterator ].getTime();
	    if ( aTime <= time )
		{
		return_value=iterator;
		}
	    
	  ++iterator;
	
      }
	while ( ( iterator <= i_end ) && ( aTime < time ) );

    return return_value;
  }

  PhysicalLoggerIterator 
  PhysicalLogger::upper_bound_linear( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time ) const

  {
    PhysicalLoggerIterator result( lower_bound_linear( start, end, time ) );

    if ( ( result < size() - 1 ) && ( theVector [ result ].getTime() != time ) )
      {
	++result;
      }

    return result;
  }

  PhysicalLoggerIterator 
  PhysicalLogger::upper_bound_search( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time,
			       bool isLinearSearch ) const
    {
	if (isLinearSearch)
	    {
	    return upper_bound_linear( start, end, time);
	    }	
	else
	    {
	    return upper_bound ( start, end, time );
	    }
    }

  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound_search( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       RealCref time,
			       bool isLinearSearch ) const
    {
	if (isLinearSearch)
	    {
	    return lower_bound_linear( start, end, time);
	    }	
	else
	    {
	    return lower_bound ( start, end, time );
	    }
    }

  void PhysicalLogger::getItem( PhysicalLoggerIteratorCref where,
				DataPointPtr what ) const
  {
    PhysicalLoggerIterator awhere( where );
    if( where > theVector.size() )
      { 
	awhere = theVector.size(); 
      }
    
    *what = theVector[ awhere ];
  }

  DataPointVectorRCPtr 
  PhysicalLogger::getVector( PhysicalLoggerIteratorCref start,
			     PhysicalLoggerIteratorCref end ) const
  {
    PhysicalLoggerIterator i_start ( start );
    PhysicalLoggerIterator i_end ( end );

    if ( start > end )
      {
	i_start = end;
	i_end = start;
      }

    PhysicalLoggerIterator counter ( start );

    DataPointVectorPtr aVector;

    //	assert((start>=0)&&(end<=theVector.size()));
    if ( empty() )
      {
	aVector = new DataPointVector ( 0 );
      }
    else
      {
	aVector = new DataPointVector ( end - start + 1 );

	do 
	  {
	    ( *aVector )[ counter - start ] = theVector [ counter ];
	    ++counter;
	  }
	  while ( counter <= end );
      }

    return DataPointVectorRCPtr( aVector );
  }
    
/*  void PhysicalLogger::set_stats( PhysicalLoggerIteratorCref distance,
				PhysicalLoggerIteratorCref num_of_elements) const
  {
    theVector.set_direct_read_stats(distance,num_of_elements);
  }

  void PhysicalLogger::set_default_stats() const
  {
    theVector.set_direct_read_stats();
  }
*/
} // namespace libecs


