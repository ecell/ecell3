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


#include <cmath>
#include <assert.h>
#include "PropertySlotProxy.hpp"


#include "Logger.hpp"

namespace libecs
{

  ///////////////////////////// LoggerAdapter

  LoggerAdapter::LoggerAdapter()
  {
    ; // do nothing
  }

  LoggerAdapter::~LoggerAdapter()
  {
    ; // do nothing
  }



  ////////////////////////////// Logger


  static const Int _LOGGER_MAX_PHYSICAL_LOGGERS = 5;
  static const Int _LOGGER_DIVIDE_STEP = 100;

  // Constructor
  Logger::Logger( LoggerAdapterPtr aLoggerAdapter )
    :
    theLoggerAdapter( aLoggerAdapter ),
    theMinimumInterval( 0.0 ),
    theLastTime( 0.0 ) // theStepper.getCurrentTime() - theMinimumInterval )
  {
    thePhysicalLoggers=new PhysicalLogger[_LOGGER_MAX_PHYSICAL_LOGGERS]; // do nothing
  }

  Logger::~Logger()
  {
    delete[] thePhysicalLoggers;
    delete theLoggerAdapter;
  }
  

  void Logger::log( const Real aTime )
  {
    appendData( aTime, theLoggerAdapter->getValue() );
  }


  DataPointVectorRCPtr Logger::getData( void ) const
  {
    if (thePhysicalLoggers[0].empty())
	{
	  return anEmptyVector();
	}

    return thePhysicalLoggers[0].getVector( thePhysicalLoggers[0].begin(),
					thePhysicalLoggers[0].end() );
  }

  //

  DataPointVectorRCPtr Logger::getData( RealCref aStartTime,
					RealCref anEndTime ) const
  {
    if (thePhysicalLoggers[0].empty())
	{
	return anEmptyVector();
	}

    PhysicalLoggerIterator 
      top( thePhysicalLoggers[0].upper_bound( thePhysicalLoggers[0].begin(),
					  thePhysicalLoggers[0].end(), 
					  anEndTime ) );

    PhysicalLoggerIterator 
      bottom( thePhysicalLoggers[0].lower_bound( thePhysicalLoggers[0].begin(),
					     top,
					     aStartTime ) );
    return thePhysicalLoggers[0].getVector( bottom, top );
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
    if (thePhysicalLoggers[0].empty())
	{
	return anEmptyVector();
	}
	
    //choose appropriate physlogger
    int log_no(_LOGGER_MAX_PHYSICAL_LOGGERS+1);
    Real avg_interval;
    do{
    --log_no;
    avg_interval=thePhysicalLoggers[log_no].get_avg_interval();
    }
    while (((avg_interval>anInterval)||(avg_interval==-1.0))&&
	    (log_no>0));
	
    Real theStartTime ( thePhysicalLoggers[log_no].front().getTime() );
    Real theEndTime ( thePhysicalLoggers[log_no].back().getTime() );
    Real time_per_step( ( theEndTime - theStartTime ) /
		    ( thePhysicalLoggers[log_no].end() - thePhysicalLoggers[log_no].begin() ) );

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
      vectorslice_end( thePhysicalLoggers[log_no].upper_bound_linear_estimate
    					( thePhysicalLoggers[log_no].begin(),
					  thePhysicalLoggers[log_no].end(),
					  theEndTime,
					  time_per_step ) );
    
    PhysicalLoggerIterator 
      vectorslice_start( thePhysicalLoggers[log_no].lower_bound_linear_estimate
    						( thePhysicalLoggers[log_no].begin(),
						   vectorslice_end,
						   theStartTime,
						   time_per_step ) );
//decide on applied method
    //refine time_per_step
    Real vectorslice_length( 
		    ( vectorslice_end - vectorslice_start ) );
    time_per_step = ( ( theEndTime - theStartTime ) / vectorslice_length );

    Real linear_step_estimate( vectorslice_length / range );
    
    //if estimated linear step is 3 or smaller use the simple lianear search
    if ( linear_step_estimate < 4 )
	{
	time_per_step = 0;
	}    

//initialize iterator indexes    
    PhysicalLoggerIterator
	lowerbound_index (vectorslice_start);
	
    PhysicalLoggerIterator
	upperbound_index ( thePhysicalLoggers[log_no].next_index( lowerbound_index ) ) ;
				
//initializa DP-s
    DataPoint current_DP;
    DataPoint lower_DP;
    DataPoint upper_DP;
    Real currentTime( theStartTime );
    thePhysicalLoggers[log_no].getItem( lowerbound_index, &lower_DP );
    thePhysicalLoggers[log_no].getItem( upperbound_index, &upper_DP );
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
		lowerbound_index=thePhysicalLoggers[log_no].lower_bound_linear_estimate(
		    upperbound_index, vectorslice_end, currentTime,
		    time_per_step );
		upperbound_index=thePhysicalLoggers[log_no].next_index( lowerbound_index );
		thePhysicalLoggers[log_no].getItem( lowerbound_index, &lower_DP );
		thePhysicalLoggers[log_no].getItem( upperbound_index, &upper_DP );
		lowerbound_time=lower_DP.getTime();
		upperbound_time=upper_DP.getTime();
		lowerbound_value=lower_DP.getValue();
		upperbound_value=upper_DP.getValue();
		time_per_step=(theEndTime-upperbound_time)/
		    (vectorslice_end-upperbound_index);
		}
	
	}
    while ( counter < range );
    
//    thePhysicalLogger.set_default_stats();
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
    int log_no(0);
    PhysicalLoggerIterator psize;
    if( theMinimumInterval <= aCurrentInterval )
      {
        theDataInterval.addPoint( aTime, aValue );
        
	do{
	
	thePhysicalLoggers[log_no].push( theDataInterval.getFinalDataPoint() );
	psize=thePhysicalLoggers[log_no].size();
	++log_no;
	}
	while ((log_no<=_LOGGER_MAX_PHYSICAL_LOGGERS) &&
		(((psize%_LOGGER_DIVIDE_STEP)==0)));
	
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
	thePhysicalLoggers[0].push( theDataInterval.getFinalDataPoint() );
	theDataInterval.beginNewInterval();
      }
  }

  //

  Real Logger::getStartTime( void ) const
  {
    return thePhysicalLoggers[0].front().getTime();
  }


  //

  Real Logger::getEndTime( void ) const
  {
    return thePhysicalLoggers[0].back().getTime();
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
