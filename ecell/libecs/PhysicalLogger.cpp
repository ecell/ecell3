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
//

#ifdef HAVE_CONFIG_H
#include "ecell_config.h"
#endif /* HAVE_CONFIG_H */

#include "VVector.h"

#include "PhysicalLogger.hpp"
#include "libecs.hpp"

namespace libecs
{

PhysicalLogger::PhysicalLogger()
    : theVector( new Vector() )
{
    setMaxSize( 0 ); // no limit
}

PhysicalLogger::~PhysicalLogger()
{
    delete theVector;
}
            
void PhysicalLogger::push( DataPoint const& aDataPoint )
{
    theVector->push_back( aDataPoint );
}

void PhysicalLogger::setEndPolicy( Integer anEndPolicy )
{
    theVector->setEndPolicy ( anEndPolicy );
}

int PhysicalLogger::getEndPolicy() const
{
    return theVector->getEndPolicy();
}

    /// set max storage size in Kbytes.

void PhysicalLogger::setMaxSize( size_type aMaxSize )
{
    theMaxSize = aMaxSize;
    theVector->setMaxSize( ( theMaxSize * 1024 ) / sizeof( DataPoint ) );
}

LongDataPoint PhysicalLogger::at( size_type index) const
{
    return (*theVector)[ index ];
}

PhysicalLogger::size_type 
PhysicalLogger::lower_bound( size_type start,
                             size_type end,
                             Real time ) const
{
    size_type iterator( ( start + end ) / 2 );
    size_type i_start( start );
    size_type i_end( end );
    
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
        i_start = i_end;
    }
    
    return i_start;
}

PhysicalLogger::size_type 
PhysicalLogger::upper_bound( size_type start,
                             size_type end,
                             Real time ) const
{
    size_type result( lower_bound( start, end, time ) );
    
    if ( ( result < size() - 1 ) && ( at( result ).getTime() != time ) )
    {
        ++result;
    }
    
    return result;
}


PhysicalLogger::size_type 
PhysicalLogger::lower_bound_linear_backwards( size_type start,
                                              size_type end,
                                              Real time ) const
{
    size_type i_start( start );
    size_type i_end( end );
    
    if ( start > end )
    {
        i_start=end;
        i_end=start;
    }
    size_type iterator( i_end );
    
    while ( (iterator>i_start) && ( at ( iterator ).getTime() > time ) )
    {
        iterator--;
    }
    return iterator;
}

PhysicalLogger::size_type 
PhysicalLogger::lower_bound_linear_estimate( size_type start,
                                             size_type end,
                                             Real time,
                                             Real time_per_step ) const
{
    //if time_per_step is zero fall back to stepwise linear search
    if ( time_per_step == 0 )
    {
        return lower_bound_linear( start, end, time);
    }
    Real theStartTime( at( start ).getTime() );
    size_type iterator;

    iterator = static_cast< size_type >( (time - theStartTime ) 
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

PhysicalLogger::size_type 
PhysicalLogger::upper_bound_linear_estimate( size_type start,
                                             size_type end,
                                             Real time,
                                             Real time_per_step ) const
{
    size_type result( lower_bound_linear_estimate( start, end, time, time_per_step ) );
    
    if ( ( result < size() - 1 ) && ( at ( result ).getTime() != time ) )
    {
        ++result;
    }
    
    return result;
    
}

PhysicalLogger::size_type 
PhysicalLogger::lower_bound_linear( size_type start,
                                    size_type end,
                                    Real time ) const
{
    size_type i_start( start );
    size_type i_end( end );
    Real aTime;
    
    if ( start > end )
    {
        i_start=end;
        i_end=start;
    }
    size_type iterator( i_start );
    size_type return_value( i_start );
    
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



PhysicalLogger::size_type 
PhysicalLogger::upper_bound_linear( size_type start,
                                    size_type end,
                                    Real time ) const
{
    size_type result( lower_bound_linear( start, end, time ) );
    
    if ( ( result < size() - 1 ) && ( at ( result ).getTime() != time ) )
    {
        ++result;
    }
    
    return result;
}

boost::shared_ptr< DataPointVector > 
PhysicalLogger::getVector( size_type start, size_type end ) const
{
    size_type i_start ( start );
    size_type i_end ( end );
    
    if ( start > end )
    {
        i_start = end;
        i_end = start;
    }
    
    size_type counter ( start );
    
    DataPointVector* aVector;
    
    if ( empty() )
    {
        aVector = new DataPointVector( 0, 2 );
    }
    else
    {
        aVector = new DataPointVector( end - start + 1, 2 );
        
        do 
        {
            ( *aVector ).asShort( counter - start ) = at( counter );
            ++counter;
        }
        while ( counter <= end );
    }
    
    return boost::shared_ptr< DataPointVector >( aVector );
}
    

PhysicalLogger::size_type PhysicalLogger::size() const
{
    return theVector->size();
}


bool PhysicalLogger::empty() const
{
    return size() == 0;
}



Real PhysicalLogger::getAverageInterval() const
{
    if( size() < 2 )
    {
        return 0.0;
    }
    else
    {
        return ( back().getTime() - front().getTime() ) / ( size() - 1 );
    }
}
    
} // namespace libecs


