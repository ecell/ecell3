//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2001 Keio University
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
//

#include "config.h" 
#include "PhysicalLogger.hpp"
//#include <stdio.h>

namespace libecs
{


  PhysicalLogger::PhysicalLogger(int aPointSize) 
    : 
    PointSize(aPointSize),
    anEmptyDataPoint(),
    theVector(),
    theVectorLong()
  {

    if (PointSize==2)
	{
		theVector.clear ( );
	} 
     else
	{
		theVectorLong.clear ( );
	}
  }

	void PhysicalLogger::setMaxSize( iterator aMaxSize){

    if (PointSize==2)
	{
		theVector.setMaxSize ( aMaxSize );
	} 
     else
	{
		theVectorLong.setMaxSize ( aMaxSize );
	}

	}


	void PhysicalLogger::setEndPolicy( Integer anEndPolicy)
	{

    if (PointSize==2)
	{
		theVector.setEndPolicy ( anEndPolicy );
	} 
     else
	{
		theVectorLong.setEndPolicy ( anEndPolicy );
	}	}


  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       const Real time ) const
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
	if ( at( iterator ).getTime() <= time )
	  {
	    i_start = iterator;
	  }
	else 
	  {
	    i_end = iterator;
	  }

	iterator = ( i_start + i_end ) / 2;
	
      }
    
    if ( at( i_end ).getTime() == time )
	{
	    i_start=i_end;
	}
    
    return i_start;
  }

  PhysicalLoggerIterator 
  PhysicalLogger::upper_bound( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       const Real time ) const

  {
    PhysicalLoggerIterator result( lower_bound( start, end, time ) );

    if ( ( result < size() - 1 ) && ( at( result ).getTime() != time ) )
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
			       const Real time ) const
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
    
    while ( (iterator>i_start) && ( at ( iterator ).getTime() > time ) )
	{
	iterator--;
	}
    return iterator;
    }

  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound_linear_estimate( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       const Real time,
			       const Real time_per_step ) const
    {
    //if time_per_step is zero fall back to stepwise linear search
    if ( time_per_step == 0 )
    {
    return lower_bound_linear( start, end, time);
    }
    Real theStartTime( at( start ).getTime() );
    PhysicalLoggerIterator iterator;

    iterator = static_cast<PhysicalLoggerIterator> ( (time - theStartTime ) 
					/ time_per_step ) + start;
    if ( iterator > end ) { iterator = end;}
    if ( at (iterator).getTime() < time )
	{
	    return lower_bound_linear( iterator, end, time);
	}
	
    if ( at (iterator).getTime() > time )
	{
	    return lower_bound_linear_backwards( start, iterator, time );
	}
    
    return iterator;
    }

  PhysicalLoggerIterator 
  PhysicalLogger::upper_bound_linear_estimate( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       const Real time,
			       const Real time_per_step ) const
    {
    PhysicalLoggerIterator result( lower_bound_linear_estimate( start, end, 
							time, time_per_step ) );

    if ( ( result < size() - 1 ) && ( at ( result ).getTime() != time ) )
      {
	++result;
      }

    return result;
    
    }
    
  PhysicalLoggerIterator 
  PhysicalLogger::lower_bound_linear( PhysicalLoggerIteratorCref start,
			       PhysicalLoggerIteratorCref end,
			       const Real time ) const
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
	  aTime =at( iterator ).getTime();
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
			       const Real time ) const
  {
    PhysicalLoggerIterator result( lower_bound_linear( start, end, time ) );

    if ( ( result < size() - 1 ) && ( at ( result ).getTime() != time ) )
      {
	++result;
      }

    return result;
  }



  void PhysicalLogger::getItem( PhysicalLoggerIteratorCref where,
				DataPointPtr what ) const
  {
    PhysicalLoggerIterator awhere( where );
    if( where > size() )
      { 
	awhere = size(); 
      }
    
    *what = at( awhere );
  }


  DataPointVectorSharedPtr 
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

    //	assert((start>=0)&&(end<=size()));
    if ( empty() )
      {
	aVector = new DataPointVector ( 0, PointSize );
      }
    else
      {
	aVector = new DataPointVector ( end - start + 1, PointSize );

	do 
	  {
	if (PointSize==2)
		{
		( *aVector ).asShort( counter - start ) = at( counter );
		}
	else
		{
		( *aVector ).asLong( counter - start ) = at( counter );
		}
	    ++counter;
	  }
	  while ( counter <= end );
      }

    return DataPointVectorSharedPtr( aVector );
  }
    


    PhysicalLogger::size_type PhysicalLogger::size() const
    {
	if (PointSize == 2)
		{
		return theVector.size();
		}
	return theVectorLong.size();
    }



    bool PhysicalLogger::empty() const
    {
      return ( size() == 0 );
    }


    PhysicalLogger::iterator PhysicalLogger::begin() const
    {
      return 0;
    }



    PhysicalLogger::iterator PhysicalLogger::end() const
    {
       if ( size() > 0 )
        {
    	    return size() - 1;
	}
       else
        {
	    return 0;
	}
    }


    DataPointLong PhysicalLogger::at( const iterator& index) const
	{
	if (PointSize == 2)
		{
		return theVector[ index ];
		}
	return theVectorLong[ index ];

	}

    DataPointLong PhysicalLogger::front() const
	{
      if ( empty ( ) )
        {
    	    return anEmptyDataPoint;
	}
	return at( 0 );

	}

    DataPointLong PhysicalLogger::back() const
	{
      if ( empty ( ) )
        {
    	    return anEmptyDataPoint;
	}
	return at( end() );

	}

	
    void PhysicalLogger::push( DataPointCref aDataPoint )
    {
      if (PointSize == 2)
	{
	theVector.push_back( aDataPoint );
	}
      else
	{
	theVectorLong.push_back( aDataPoint );
	}
    }


    void PhysicalLogger::push( DataPointLongCref aDataPoint )
    {
      if (PointSize == 2)
	{
	theVector.push_back( aDataPoint );
	}
      else
	{
	theVectorLong.push_back( aDataPoint );
	}
    }
  

   Real PhysicalLogger::get_avg_interval() const
   {
   if (size()<2)
     {
       return 0.0;
     }
   else
     {
       return (back().getTime()-front().getTime())/(size()-1);
     }
   }




} // namespace libecs


