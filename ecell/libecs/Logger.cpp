//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::://
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
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>
// 14/04/2002


#include "Logger.hpp"
#include <cmath>
#include <assert.h>

#include <stdio.h>
namespace libecs
{

  // Constructor
  Logger::Logger( PropertySlotRef aPropertySlot ) 
    :
    thePropertySlot( aPropertySlot ),
    theMinimumInterval( 0.0 ),
    theLastTime( 0.0 ) // theStepper.getCurrentTime() - theMinimumInterval )
  {
    ; // do nothing
  }


  DataPointVectorRCPtr Logger::getData( void ) const
  {
    if (thePhysicalLogger.empty())
	{
	return anEmptyVector();
	}

    return thePhysicalLogger.getVector( thePhysicalLogger.begin(),
					thePhysicalLogger.end() );
  }

  //

  DataPointVectorRCPtr Logger::getData( RealCref aStartTime,
					RealCref anEndTime ) const
  {
    if (thePhysicalLogger.empty())
	{
	return anEmptyVector();
	}

    PhysicalLoggerIterator 
      top( thePhysicalLogger.upper_bound( thePhysicalLogger.begin(),
					  thePhysicalLogger.end(), 
					  anEndTime ) );

    PhysicalLoggerIterator 
      bottom( thePhysicalLogger.lower_bound( thePhysicalLogger.begin(),
					     top,
					     aStartTime ) );
    return thePhysicalLogger.getVector( bottom, top );
  }


  //

  DataPointVectorRCPtr Logger::anEmptyVector(void) const
  
  {
  DataPointVectorRCPtr aDataPointVector( new DataPointVector (0) );
  return aDataPointVector;
  }
  
  
  DataPointVectorRCPtr Logger::getData( RealCref aStartTime,
					RealCref anEndTime,
					RealCref anInterval ) const
  {
    if (thePhysicalLogger.empty())
	{
	return anEmptyVector();
	}
    Real theStartTime ( thePhysicalLogger.front().getTime() );
    Real theEndTime ( thePhysicalLogger.back().getTime() );
    if ( theStartTime < aStartTime )
      { 
	theStartTime = aStartTime;
      }
    if ( theEndTime > anEndTime )
      { 
	theEndTime = anEndTime;
      }
  
//    PhysicalLoggerIterator theMaxSize ( thePhysicalLogger.end() );  
// set up output vector
    DataPointVectorIterator 
      range( static_cast<DataPointVectorIterator>
	     ( ( theEndTime - theStartTime ) / anInterval ) );
    //this is a technical adjustment, because I realized that sometimes
    //conversion from real is flawed, maybe because after some divisions
    //the real value is not exactly an integer value
    Real range_pre( ( theEndTime - theStartTime ) / anInterval );

    if ( ( static_cast<Real>(range) ) + 0.9999 < range_pre ) 
	{
	    range++;
	}
	
    range++;

    DataPointVectorPtr aDataPointVector( new DataPointVector( range ) );
    DataPointVectorIterator counter( 0 );

//set uo iterators
    PhysicalLoggerIterator 
      vectorslice_end( thePhysicalLogger.upper_bound( thePhysicalLogger.begin(),
					  thePhysicalLogger.end(),
					  theEndTime ) );
    
    PhysicalLoggerIterator 
      vectorslice_start( thePhysicalLogger.lower_bound( thePhysicalLogger.begin(),
						   vectorslice_end,
						   theStartTime ) );
//decide on applied method
    
    Real vectorslice_length( 
		    ( vectorslice_end - vectorslice_start ) );

    Real linear_step_estimate( vectorslice_length / range );
    Real logarythmic_step_estimate ( log2( vectorslice_length ) );
    
//    bool isLinearSearch( (logarythmic_step_estimate > linear_step_estimate) );
    bool isLinearSearch ( true );
//initialize iterator indexes    
    PhysicalLoggerIterator
	lowerbound_index (vectorslice_start);
	
    PhysicalLoggerIterator
	upperbound_index ( thePhysicalLogger.next_index( lowerbound_index ) ) ;
				
//initializa DP-s
    DataPoint current_DP;
    DataPoint lower_DP;
    DataPoint upper_DP;
    Real currentTime( theStartTime );
    thePhysicalLogger.getItem( lowerbound_index, &lower_DP );
    thePhysicalLogger.getItem( upperbound_index, &upper_DP );
    Real lowerbound_time( lower_DP.getTime() );
    Real upperbound_time( upper_DP.getTime() );
    Real lowerbound_value( lower_DP.getValue() );
    Real upperbound_value( upper_DP.getValue() );
    
    
    do 
	{

	    assert ( lowerbound_time <= currentTime );//this would be a serious algorythmical error
	    //if upperbound.time>=currenttime, then it is within interval
	    //value can be calculated and stored
	    if ( upperbound_time>= currentTime )
		{
		//decide whether there is need to interpolate
		if ( lowerbound_time == currentTime )
		    {
		    current_DP.setValue ( lowerbound_value );
		    }
		else
		    {
		    if (upperbound_time == currentTime )
			{
			current_DP.setValue ( upperbound_value );
			}
		    else //interpolate
			{
			current_DP.setValue( ( 
			    ( upperbound_value - lowerbound_value ) /
			    ( upperbound_time - lowerbound_time ) ) *
			    ( currentTime - lowerbound_time ) +
			    lowerbound_value );
			}
		    }
		// store currentDP
	    	current_DP.setTime( currentTime );
    		( *aDataPointVector )[counter] = current_DP;
		counter++;
		currentTime+=anInterval;
		if ( currentTime > theEndTime ) 
		    {
		    currentTime = theEndTime;
		    }		
		}
	    else //upperbound_time<current_time
	    //else get new value
		{
		lowerbound_index=thePhysicalLogger.lower_bound_search(
		    upperbound_index, vectorslice_end, currentTime,
		    isLinearSearch );

		upperbound_index=thePhysicalLogger.next_index( lowerbound_index );
		thePhysicalLogger.getItem( lowerbound_index, &lower_DP );
		thePhysicalLogger.getItem( upperbound_index, &upper_DP );
		lowerbound_time=lower_DP.getTime();
		upperbound_time=upper_DP.getTime();
		lowerbound_value=lower_DP.getValue();
		upperbound_value=upper_DP.getValue();
		}
	
	}
    while ( counter < range );
    

    return DataPointVectorRCPtr( aDataPointVector );
  }

  void Logger::setMinimumInterval( RealCref anInterval )
  {
    if( anInterval < 0 )
      {
	THROW_EXCEPTION( ValueError, 
			 "Negative value given to Logger::MinimumInterval" );
      }
    
    theMinimumInterval = anInterval;
  }


  void Logger::appendData( RealCref aTime, RealCref aValue )
  {
    const Real aCurrentInterval( aTime - theLastTime );

    if( theMinimumInterval <= aCurrentInterval )
      {
        theDataInterval.addPoint( aTime, aValue );
        thePhysicalLogger.push( theDataInterval.getFinalDataPoint() );
	theLastTime = aTime;
	theDataInterval.beginNewInterval();
      }
    
    theDataInterval.addPoint( aTime, aValue );
  }


  void Logger::flush()
  {
    // prevent flushing it twice
    // if min ingterval is zero there is no point in flushing

    if ( ( theDataInterval.getInterval() != -1.0 ) && 
	 ( theMinimumInterval > 0 ) )
      {  
	thePhysicalLogger.push( theDataInterval.getFinalDataPoint() );
	theDataInterval.beginNewInterval();
      }
  }

  //

  Real Logger::getStartTime( void ) const
  {
    return thePhysicalLogger.front().getTime();
  }


  //

  Real Logger::getEndTime( void ) const
  {
    return thePhysicalLogger.back().getTime();
  }



} // namespace libecs


#ifdef LOGGER_TEST


#include <stdio.h>
#include <iostream>
#include "DataPoint.cpp"
#include "StlDataPointVector.cpp"

using namespace libecs;

const Real& func(void)
{
  const Real* fp = new Real(3.14);
  return *fp;
}

main()
{


  Logger<Real,Real> lg = Logger<Real,Real>(op);
  /*
    lg.update(d1);
    lg.update(d2);
    lg.update(d3);
    lg.update(d4);
    lg.update(d5);
  */


  Logger<Real,Real> lg_clone = Logger<Real,Real>(lg);

  //  printf("%p %p\n",&(lg.getDataPointVector()),&(lg_clone.getDataPointVector()));

  lg.getData(0.0,5.0,0.5);
  printf("%f\n",lg.getStartTime());
  printf("%f\n",lg.getEndTime());
}



#endif /* LOGGER_TEST */
