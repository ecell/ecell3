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
    theMinimumStep ( 1 ),
    thePrimaryPhysicalLogger( 2 ),
    thePrimaryMaxSize( 0 )
  {
    PolymorphVector aPolymorphVector;
    aPolymorphVector.push_back( static_cast<Integer> ( 1 ) );
    aPolymorphVector.push_back( static_cast<Real> ( 0.0 ) );
    aPolymorphVector.push_back( static_cast<Integer> ( 0 ) );
    aPolymorphVector.push_back( static_cast<Integer> ( 0 ) );
    theLoggingPolicy = aPolymorphVector;
  }


  //Destructor
  Logger::~Logger()
  {

    for ( int i=0; i < theSubPhysicalLoggerArray.size(); i++ )
      {
	delete theSubPhysicalLoggerArray[ i ];
      }
    
    delete theLoggerAdapter;
  }
  


  void Logger::setLoggerPolicy( PolymorphCref aParamList )
  {

    Integer userSpecifiedLimit( 0 );
    if ( aParamList.asPolymorphVector().size() != 4)
      {
	THROW_EXCEPTION( libecs::Exception, "Logger policy array should be 4 element long.\n" );
      }

    theLoggingPolicy = aParamList;
    theMinimumStep = theLoggingPolicy.asPolymorphVector()[ STEP_SIZE ].asInteger();
    theMinimumInterval = theLoggingPolicy.asPolymorphVector()[ TIME_INTERVAL ].asReal();
    userSpecifiedLimit = theLoggingPolicy.asPolymorphVector()[ MAX_SPACE ].asInteger();

    //calculate maximum size of logger from user specified limit in Kbytes
    if ( userSpecifiedLimit > 0 )
      {
	Real theLoggerRatio( 1.0 / LOGGER_DIVIDE_STEP );

	// calculating sum for 1/(1-x) to estimate how many additional logs are performed for one ordinary log
	Real estimatedSecondaryLoggerAbundance( 1.0 / ( 1.0 - theLoggerRatio ) );

	Real theAverageDataPointSize( static_cast<Real>( sizeof( DataPoint ) ) + 
				 static_cast<Real>( sizeof( DataPointLong ) ) * estimatedSecondaryLoggerAbundance );

	// make our estimate a bit conservative
	theAverageDataPointSize *= 1.02;
	
	thePrimaryMaxSize = static_cast<PhysicalLoggerIterator>( static_cast<Real>( userSpecifiedLimit * 1024 )
						 / theAverageDataPointSize );
      }
    thePrimaryPhysicalLogger.setEndPolicy( theLoggingPolicy.asPolymorphVector()[ END_POLICY ].asInteger() );
    thePrimaryPhysicalLogger.setMaxSize( thePrimaryMaxSize );

    for ( int i = 0; i < theSubPhysicalLoggerArray.size(); ++i )
      {
	setSubLoggerPolicy( i );
      }
    
  }

  void Logger::setSubLoggerPolicy ( int anIndex )
  {
    PhysicalLoggerPtr aPhysicalLoggerPtr = theSubPhysicalLoggerArray[ anIndex ];
    aPhysicalLoggerPtr -> setEndPolicy( theLoggingPolicy.asPolymorphVector()[ END_POLICY ].asInteger() );


    if ( thePrimaryMaxSize > 0 )
      {
	PhysicalLoggerIterator aMaxSize =  thePrimaryMaxSize; 
	for ( int i = 0; i < anIndex; i++ )
	  {
	    aMaxSize /= LOGGER_DIVIDE_STEP;
	  }
	
	aPhysicalLoggerPtr -> setMaxSize( aMaxSize );
      }
    else
      {
	aPhysicalLoggerPtr -> setMaxSize( 0 );
	
      }
    
  }


  DataPointVectorSharedPtr Logger::getData( void ) const
  {
    if (thePrimaryPhysicalLogger.empty())
      {
	return anEmptyVector();
      }
    
    return thePrimaryPhysicalLogger.getVector( thePrimaryPhysicalLogger.begin(),
					       thePrimaryPhysicalLogger.end() );
  }

  

  DataPointVectorSharedPtr Logger::getData( RealParam aStartTime,
					    RealParam anEndTime ) const
  {
    if (thePrimaryPhysicalLogger.empty())
      {
	return anEmptyVector();
      }
    
    PhysicalLoggerIterator 
      topIterator( thePrimaryPhysicalLogger.upper_bound( thePrimaryPhysicalLogger.begin(),
							 thePrimaryPhysicalLogger.end(), 
							 anEndTime ) );
    
    PhysicalLoggerIterator 
      bottomIterator( thePrimaryPhysicalLogger.lower_bound( thePrimaryPhysicalLogger.begin(),
							    topIterator,
							    aStartTime ) );

    return thePrimaryPhysicalLogger.getVector( bottomIterator, topIterator );
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
    thePrimaryPhysicalLogger.aggregate( aDataPoint ); 

	if ( ( theMinimumStep > 0 ) || ( theMinimumInterval >= 0 ) ) 
	  {
	    
	    if ( theMinimumStep > 0 )
	      {
		theStepCounter++;
		theStepCondition = ( theStepCounter >= static_cast<PhysicalLoggerIterator>( theMinimumStep ) );
	      }
	    else
	      {
		; //do nothing
	      }
	    
	    if ( theMinimumInterval > 0 )
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

	flush();
	
	
	theLastTime = aTime;
	theStepCounter = 0;
	
      }
    else
      {
	; // do nothing
      }
  
  }



  void Logger::aggregate( DataPointLongCref aDataPointLong, int aPhysicalLoggerIndex )
  {
    
    DataPointLong aDataPointLongAggregator;
    
    if ( theSubPhysicalLoggerArray.size() == MAX_SUBLOGGER_NUMBER ) 
      { 
	return;
      }
    if ( aPhysicalLoggerIndex == theSubPhysicalLoggerArray.size() )
      {
	// create new logger
	theSubPhysicalLoggerArray.push_back( new PhysicalLogger( 5 ) );

	// set max size, endpolicy for new logger
	setSubLoggerPolicy( aPhysicalLoggerIndex );

      }
    
    //aggregate
    PhysicalLoggerPtr aPhysicalLoggerPtr = theSubPhysicalLoggerArray[ aPhysicalLoggerIndex ];
    aPhysicalLoggerPtr -> aggregate( aDataPointLong );
    
    // if psize is turning point

    if ( aPhysicalLoggerPtr -> getElementCount() == LOGGER_DIVIDE_STEP )
      {
	aPhysicalLoggerPtr -> flushAggregate();

	//aggregate highlevel
	aggregate( aDataPointLongAggregator, aPhysicalLoggerIndex + 1 );

      }
    
  }


  void Logger::flush()
  {
    // preventaDataPointLong flushing it twice
    // if min ingterval is zero there is no point in flushing
    
    //aggregate highlevel
    DataPointLongCref aDataPointLong = thePrimaryPhysicalLogger.getAggregate();
    if ( aDataPointLong.getTime() >= 0.0 )
      {
	aggregate( aDataPointLong , 0 );
    
	thePrimaryPhysicalLogger.flushAggregate();
      }

  }
  
  //
  
  Real Logger::getStartTime( void ) const
  {
    return  thePrimaryPhysicalLogger.front().getTime();
    
  }
  

  //

  Real Logger::getEndTime( void ) const
  {
    
    return thePrimaryPhysicalLogger.back().getTime();
  }
  
  



  DataPointVectorSharedPtr Logger::getData( RealParam aStartTime,
					RealParam anEndTime,
					RealParam anInterval ) const
  {
    if ( thePrimaryPhysicalLogger.empty() )
	{
	return anEmptyVector();
	}

	
    //choose appropriate physlogger
    int aPhysicalLoggerVectorIndex = theSubPhysicalLoggerArray.size();

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
    
    Real theAverageTimeInterval( 0.0 );
    PhysicalLoggerPtr aPhysicalLoggerPtr = NULL;
    bool isFitForExtraction = false;
    do
      {
	--aPhysicalLoggerVectorIndex;
	aPhysicalLoggerPtr = theSubPhysicalLoggerArray[ aPhysicalLoggerVectorIndex ];
	theAverageTimeInterval = aPhysicalLoggerPtr ->getAverageInterval();
	isFitForExtraction = ( ( theAverageTimeInterval < ( anInterval / 3 ) ) && 
			       ( theAverageTimeInterval != 0.0 ) && 
			       ( aPhysicalLoggerPtr -> size() > thePhysicalRange ) );
	
      } 
    while (  ( !isFitForExtraction ) && ( aPhysicalLoggerVectorIndex > 0 ) ) ;
      
    if ( !isFitForExtraction )
      {
	aPhysicalLoggerPtr = const_cast<PhysicalLoggerPtr>( &thePrimaryPhysicalLogger );
      }
    else
      {
	;// do nothing
      }
    
    
    Real theStartTime ( aPhysicalLoggerPtr ->front().getTime() );
    Real theEndTime ( aPhysicalLoggerPtr->back().getTime() );
    Real theRealTimeGap( ( theEndTime - theStartTime ) /
			 ( aPhysicalLoggerPtr->end() - aPhysicalLoggerPtr->begin() ) );

    theStartTime = aStartTime;
    theEndTime = anEndTime;




    DataPointVectorPtr aDataPointVector( new DataPointVector( thePhysicalRange, 5 ) );

    //set uo iterators
    
    PhysicalLoggerIterator 
      theIterationEnd( aPhysicalLoggerPtr->upper_bound_linear_estimate
		       ( aPhysicalLoggerPtr->begin(),
			 aPhysicalLoggerPtr->end(),
			 theEndTime,
			 theRealTimeGap ) );
    
    PhysicalLoggerIterator 
      theIterationStart( aPhysicalLoggerPtr->lower_bound_linear_estimate( 
									 aPhysicalLoggerPtr->begin(),
									 theIterationEnd,
									 theStartTime,
									 theRealTimeGap ) );
    // start from vectorslice start to vectorslice end, scan through all datapoints
	
    PhysicalLoggerIterator loggerCounter( theIterationStart );	
    Real targetTime( theStartTime + anInterval );
    DataPointLong readDpl( aPhysicalLoggerPtr->at( loggerCounter ) );
    //DataPointLong dp1;
    
    DataPointAggregator anAggregator;
    anAggregator.aggregate( readDpl );
    for ( DataPointVectorIterator elementCount = 0; elementCount < thePhysicalRange; elementCount++ )
      {

	do 
	  {
	    
	    if ( ( loggerCounter < theIterationEnd ) && ( readDpl.getTime() < targetTime ) )
	      { 
		loggerCounter++;
		readDpl = aPhysicalLoggerPtr->at( loggerCounter );
		
		
	      }
	    anAggregator.aggregate( readDpl );
	    
	  }
	while( ( readDpl.getTime() < targetTime ) && (loggerCounter < theIterationEnd ) );
	
	
	aDataPointVector->asLong(elementCount) = anAggregator.getData();
	
	anAggregator.beginNextPoint();
	
	targetTime += anInterval;
      }
    
    
    return DataPointVectorSharedPtr( aDataPointVector );
    
  }
  

} // namespace libecs

