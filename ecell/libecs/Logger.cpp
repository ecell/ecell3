//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::://
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>
// 14/04/2002


#include <cmath>
#include <assert.h>

#include <stdio.h>

#include "PropertySlotProxy.hpp"

#include "Polymorph.hpp"
#include "Logger.hpp"

namespace libecs
{


  // Constructor
  Logger::Logger( LoggerAdapterPtr aLoggerAdapter)
    :
    theLoggerAdapter( aLoggerAdapter ),
    theMinimumInterval( 0.0 ),
    theLastTime( 0.0 ) ,
    theStepCounter( 0 ),
    theMinimumStep ( 1 )
  {

    // init physicallogers, the first 2 element, the others five element ones
    thePhysicalLoggers[ 0 ] = new PhysicalLogger( 2 );
    theSizeArray[ 0 ] = 0;
    theDataAggregators = new DataPointAggregator[ _LOGGER_MAX_PHYSICAL_LOGGERS ];

    for ( int i=1; i<_LOGGER_MAX_PHYSICAL_LOGGERS; i++ )
      {
	thePhysicalLoggers[ i ] = new PhysicalLogger( 5 );
	theSizeArray[ i ] = 0;
      }

  }


  //Destructor
  Logger::~Logger()
  {

    for ( int i=0; i<_LOGGER_MAX_PHYSICAL_LOGGERS; i++ )
      {
	delete thePhysicalLoggers[ i ];
      }
    
    
    delete[] theDataAggregators;
    delete theLoggerAdapter;
  }
  


  void Logger::setLoggerPolicy( PolymorphCref aParamList )
  {

    phys_iterator theMaxSize( 0 );
    Integer userSpecifiedLimit( 0 );

    if ( aParamList.asPolymorphVector().size() != 4)
      {
	THROW_EXCEPTION( libecs::Exception, "Logger policy array should be 4 element long.\n" );
      }

    theLoggingPolicy = aParamList;
    theMinimumStep = theLoggingPolicy.asPolymorphVector()[ _STEP_SIZE ].asInteger();
    theMinimumInterval = theLoggingPolicy.asPolymorphVector()[ _TIME_INTERVAL ].asReal();
    userSpecifiedLimit = theLoggingPolicy.asPolymorphVector()[ _MAX_SPACE ].asInteger();

    //calculate maximum size of logger from user specified limit in Kbytes
    if ( userSpecifiedLimit > 0 )
      {
	Real theLoggerRatio( 1.0 / _LOGGER_DIVIDE_STEP );

	// calculating sum for 1/x + 1/x^2 + 1/x^3 ...to estimate how many additional logs are performed for one ordinary log
	Real estimatedSecondaryLoggerAbundance( theLoggerRatio * 
						( static_cast<Real>( powf( 
									  static_cast<float>( theLoggerRatio ), 
									  static_cast<float> ( _LOGGER_MAX_PHYSICAL_LOGGERS ) - 1.0 
									  )  
								     ) - 1.0 
						) / ( theLoggerRatio - 1.0 ) );

	Real theAverageDataPointSize( static_cast<Real>( sizeof( DataPoint ) ) + 
				 static_cast<Real>( sizeof( DataPointLong ) ) * estimatedSecondaryLoggerAbundance );

	// make our estimate a bit conservative
	theAverageDataPointSize *= 1.02;
	
	theMaxSize = static_cast<phys_iterator>( static_cast<Real>( userSpecifiedLimit * 1024 )
						 / theAverageDataPointSize );
      }
    
    for ( int i = 0; i < _LOGGER_MAX_PHYSICAL_LOGGERS; i++ )
      {
	thePhysicalLoggers[ i ] -> setMaxSize( theMaxSize );
	thePhysicalLoggers[ i ] -> setEndPolicy( theLoggingPolicy.asPolymorphVector()[ _END_POLICY ].asInteger() );
	theMaxSize /= _LOGGER_DIVIDE_STEP;
      }
    
  }




  DataPointVectorSharedPtr Logger::getData( void ) const
  {
    if (thePhysicalLoggers[0]->empty())
      {
	return anEmptyVector();
      }
    
    return thePhysicalLoggers[0]->getVector( thePhysicalLoggers[0]->begin(),
					     thePhysicalLoggers[0]->end() );
  }

  

  DataPointVectorSharedPtr Logger::getData( RealParam aStartTime,
					RealParam anEndTime ) const
  {
    if (thePhysicalLoggers[0]->empty())
      {
	return anEmptyVector();
      }
    
    PhysicalLoggerIterator 
      top( thePhysicalLoggers[0]->upper_bound( thePhysicalLoggers[0]->begin(),
					       thePhysicalLoggers[0]->end(), 
					       anEndTime ) );
    
    PhysicalLoggerIterator 
      bottom( thePhysicalLoggers[0]->lower_bound( thePhysicalLoggers[0]->begin(),
						  top,
						  aStartTime ) );

    return thePhysicalLoggers[0]->getVector( bottom, top );
  }


 

  DataPointVectorSharedPtr Logger::anEmptyVector(void) const
  
  {

    DataPointVectorSharedPtr aDataPointVector( new DataPointVector (0,2) );
    return aDataPointVector;

  }
  
  

  void Logger::setMinimumInterval( RealParam anInterval )
  {
    if( anInterval < 0 )
      {
	THROW_EXCEPTION( ValueError, 
			 "Negative value given to Logger::MinimumInterval" );
      }
    
    theMinimumInterval = anInterval;
  }


  void Logger::appendData( RealParam aTime, RealParam aValue )
  {
    
    const Real       aCurrentInterval( aTime - theLastTime );
    DataPoint        aDataPoint;
    DataPointLong    aDataPointLong;
    bool             theLogCondition( true );
    bool             theStepCondition( false );
    bool             theTimeCondition( false );


    aDataPoint.setTime( aTime );
    aDataPoint.setValue( aValue );
    theDataAggregators[ 0 ].aggregate( aDataPoint ); 

	if ( ( theMinimumStep > 0 ) || ( theMinimumInterval >= 0 ) ) 
	  {
	    
	    if (theMinimumStep>0)
	      {
		theStepCounter++;
		theStepCondition = ( theStepCounter >= static_cast<const_iterator>( theMinimumStep ) );
	      }
	    else
	      {
		; //do nothing
	      }
	    
	    if (theMinimumInterval>0)
	      {
		theTimeCondition = ( theMinimumInterval <= aCurrentInterval );
	      }
	    else
	      {
		; // do nothing
	      }

	    theLogCondition = theTimeCondition || theStepCondition;
	  }
	else
	  {
	    ; //do nothing
	  }


    if ( theLogCondition )
      {
	
	//getdata
	aDataPointLong = theDataAggregators[ 0 ].getData();
	
	//store
	thePhysicalLoggers[ 0 ]->push( aDataPointLong );
	theSizeArray[ 0 ]++;
	
	//aggregate highlevel
	aggregate( aDataPointLong, 1 );
	
	//beginnextpoint
	
	theDataAggregators[0].beginNextPoint();
	
	theLastTime = aTime;
	theStepCounter = 0;
	
      }
    else
      {
	; // do nothing
      }
  
  }



  void Logger::aggregate( DataPointLong aDataPointLong, int aLoggerIndex )
  {
    
    DataPointLong aDataPointLongAggregator;
    
    if ( aLoggerIndex == _LOGGER_MAX_PHYSICAL_LOGGERS ) 
      { 
	return;
      }
    
    
    //aggregate
    
    theDataAggregators[ aLoggerIndex ].aggregate( aDataPointLong );
    
    // if psize is turning point
    if (theSizeArray[ aLoggerIndex-1 ]==_LOGGER_DIVIDE_STEP)
      {
	
	//getdata
	aDataPointLongAggregator = theDataAggregators[ aLoggerIndex ].getData();
	
	//store
	thePhysicalLoggers[ aLoggerIndex ]->push( aDataPointLongAggregator );
	theSizeArray[ aLoggerIndex ]++;
	
	//aggregate highlevel
	aggregate( aDataPointLongAggregator, aLoggerIndex + 1 );
	
	//beginnextpoint
	theDataAggregators[ aLoggerIndex ].beginNextPoint();
	theSizeArray[ aLoggerIndex - 1 ] = 0;
	
	
      }
    
    
  }


  void Logger::flush()
  {
    // prevent flushing it twice
    // if min ingterval is zero there is no point in flushing
    
    if ( ( theDataAggregators[0].getData().getTime() != -1.0 ) && 
	 ( theMinimumInterval > 0 ) )
      {  
	thePhysicalLoggers[0]->push( theDataAggregators[0].getData() );
	theDataAggregators[0].beginNextPoint();
      }
  }
  
  //
  
  Real Logger::getStartTime( void ) const
  {
    Real retval(thePhysicalLoggers[0]->front().getTime());
    
    return retval;
  }
  

  //

  Real Logger::getEndTime( void ) const
  {
    
    return thePhysicalLoggers[0]->back().getTime();
  }
  
  



  DataPointVectorSharedPtr Logger::getData( RealParam aStartTime,
					RealParam anEndTime,
					RealParam anInterval ) const
  {
    if (thePhysicalLoggers[0]->empty())
	{
	return anEmptyVector();
	}

	
    //choose appropriate physlogger
    int aLoggerIndex(_LOGGER_MAX_PHYSICAL_LOGGERS);

    // set up output vector
    DataPointVectorIterator 
      thePhysicalRange( static_cast<DataPointVectorIterator>
	     ( ( anEndTime - aStartTime ) / anInterval ) );

    //this is a technical adjustment, because I realized that sometimes
    //conversion from real is flawed: rounding error
    Real theEstimatedRange( ( anEndTime - aStartTime ) / anInterval );

    if ( ( static_cast<Real>(thePhysicalRange) ) + 0.9999 < theEstimatedRange ) 
      {
	thePhysicalRange++;
      }
    
    thePhysicalRange++;
    
    Real theAverageTimeInterval;

    do
      {
	--aLoggerIndex;
	theAverageTimeInterval=thePhysicalLoggers[ aLoggerIndex ]->get_avg_interval();

      } while ( ( ( theAverageTimeInterval > ( anInterval / 3 ) ) || 
		  ( theAverageTimeInterval == 0.0 ) || 
		  ( thePhysicalLoggers[aLoggerIndex]->size() < thePhysicalRange ) ) &&
		  ( aLoggerIndex > 0 ) );


    Real theStartTime ( thePhysicalLoggers[aLoggerIndex]->front().getTime() );
    Real theEndTime ( thePhysicalLoggers[aLoggerIndex]->back().getTime() );
    Real theRealTimeGap( ( theEndTime - theStartTime ) /
		    ( thePhysicalLoggers[aLoggerIndex]->end() - thePhysicalLoggers[aLoggerIndex]->begin() ) );


	theStartTime = aStartTime;
	theEndTime = anEndTime;




    DataPointVectorPtr aDataPointVector( new DataPointVector( thePhysicalRange, 5 ) );


//set uo iterators
    PhysicalLoggerIterator 
      theIterationEnd( thePhysicalLoggers[aLoggerIndex]->upper_bound_linear_estimate
    					( thePhysicalLoggers[aLoggerIndex]->begin(),
					  thePhysicalLoggers[aLoggerIndex]->end(),
					  theEndTime,
					  theRealTimeGap ) );


    
    PhysicalLoggerIterator 
      theIterationStart( thePhysicalLoggers[aLoggerIndex]->lower_bound_linear_estimate( 
							    thePhysicalLoggers[aLoggerIndex]->begin(),
							    theIterationEnd,
							    theStartTime,
							    theRealTimeGap ) );

	// start from vectorslice start to vectorslice end, scan through all datapoints
	
	PhysicalLoggerIterator loggerCounter( theIterationStart );	
	Real targetTime( theStartTime + anInterval );
	DataPointLong readDpl( thePhysicalLoggers[aLoggerIndex]->at( loggerCounter ) );
	//DataPointLong dp1;

	DataPointAggregator theAggregator;
	theAggregator.aggregate( readDpl );
	for (DataPointVectorIterator elementCount = 0;elementCount < thePhysicalRange; elementCount++)
	  {
	    do 
	      {
		
		if ((loggerCounter < theIterationEnd)&&(readDpl.getTime() < targetTime))
		  { 
		    loggerCounter++;
		    readDpl = thePhysicalLoggers[aLoggerIndex]->at( loggerCounter );
		    
		    
		  }
		theAggregator.aggregate( readDpl );
		
	      }
	    while( ( readDpl.getTime() < targetTime ) && (loggerCounter < theIterationEnd ) );


	    aDataPointVector->asLong(elementCount) = theAggregator.getData();
 
	    theAggregator.beginNextPoint();
	    
	    targetTime += anInterval;
	  }
	
	
	return DataPointVectorSharedPtr( aDataPointVector );

  }


} // namespace libecs

