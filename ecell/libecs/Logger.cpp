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


  ////////////////////////////// Logger
//#   define _LOGGER_MAX_PHYSICAL_LOGGERS  5;
//#   define int _LOGGER_DIVIDE_STEP  100;



  // Constructor
  Logger::Logger( LoggerAdapterPtr aLoggerAdapter)
    :
    theLoggerAdapter( aLoggerAdapter ),
    theMinimumInterval( 0.0 ),
    theLastTime( 0.0 ) ,
	theStepCounter( 0 ),
	theMinimumStep (1 )
  {
	// init physicallogers, the first 2 element, the others five element ones
	thePhysicalLoggers[0] = new PhysicalLogger(2);
	sizeArray[0]=0;
	theDataAggregators = new DataPointAggregator[_LOGGER_MAX_PHYSICAL_LOGGERS];
	for (int i=1;i<_LOGGER_MAX_PHYSICAL_LOGGERS;i++)
	{
	   thePhysicalLoggers[i] = new PhysicalLogger(5);
		sizeArray[i]=0;
	}
  }

  Logger::~Logger()
  {
	for (int i=0;i<_LOGGER_MAX_PHYSICAL_LOGGERS;i++)
	{
	   delete thePhysicalLoggers[i];
	}


    delete[] theDataAggregators;
    delete theLoggerAdapter;
  }
  
  void Logger::setLoggerPolicy( PolymorphCref aParamList )
  {
	if (aParamList.asPolymorphVector().size()!=4){
		THROW_EXCEPTION( libecs::Exception, "Logger policy array should be 4 element long.\n" );
}
	loggingPolicy = aParamList;
	theMinimumStep = loggingPolicy.asPolymorphVector()[0].asInteger();
	theMinimumInterval = loggingPolicy.asPolymorphVector()[1].asReal();
	phys_iterator theMaxSize(0);
	//calculate maxsize from available Kbytes
	if (loggingPolicy.asPolymorphVector()[3].asInteger()>0){
		Real lds_inv( 1.0/_LOGGER_DIVIDE_STEP );
		Real longdp_part( lds_inv * ( static_cast<Real>(powf(static_cast<float>(lds_inv), static_cast<float>
		(_LOGGER_MAX_PHYSICAL_LOGGERS)-1.0))-1.0)/(lds_inv-1.0));
		Real avg_datapoint_size( static_cast<Real>(sizeof(DataPoint)) + 
			static_cast<Real>(sizeof(DataPointLong))*longdp_part );
		avg_datapoint_size*=1.02;

		theMaxSize = static_cast<phys_iterator>(static_cast<Real>(loggingPolicy.asPolymorphVector()[3].asInteger())*1024.0/avg_datapoint_size );
	}

	for (int i=0;i<_LOGGER_MAX_PHYSICAL_LOGGERS;i++)
	{
	    thePhysicalLoggers[i]->setMaxSize( theMaxSize);
	    thePhysicalLoggers[i]->setEndPolicy( loggingPolicy.asPolymorphVector()[2].asInteger());
		theMaxSize/=_LOGGER_DIVIDE_STEP;
	}

  }




  DataPointVectorRCPtr Logger::getData( void ) const
  {
    if (thePhysicalLoggers[0]->empty())
	{
	  return anEmptyVector();
	}

    return thePhysicalLoggers[0]->getVector( thePhysicalLoggers[0]->begin(),
					thePhysicalLoggers[0]->end() );
  }

  //

  DataPointVectorRCPtr Logger::getData( RealParam aStartTime,
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


  //

  DataPointVectorRCPtr Logger::anEmptyVector(void) const
  
  {
  DataPointVectorRCPtr aDataPointVector( new DataPointVector (0,2) );
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

    const Real aCurrentInterval( aTime - theLastTime );
    DataPoint dp;
    DataPointLong dpl;
	bool logcondition(true);
	bool stepcondition(false);
	bool timecondition(false);
    dp.setTime( aTime);
    dp.setValue( aValue);
    theDataAggregators[0].aggregate( dp ); 

	if ((theMinimumStep>0)||(theMinimumInterval>=0)){

	if (theMinimumStep>0){
		theStepCounter++;

		stepcondition = ( theStepCounter >= static_cast<const_iterator>(theMinimumStep) );

		}
	if (theMinimumInterval>0){

		timecondition = ( theMinimumInterval <= aCurrentInterval );

		}
	logcondition=timecondition||stepcondition;


	}
    if ( logcondition )
      {

		//getdata
		dpl=theDataAggregators[0].getData();

		//store
		thePhysicalLoggers[0]->push( dpl );
		sizeArray[0]++;

		//aggregate highlevel
		aggregate( dpl, 1);
		
		//beginnextpoint

		theDataAggregators[0].beginNextPoint();

		theLastTime = aTime;
		theStepCounter = 0;

	}


  
  }


   void Logger::aggregate( DataPointLong dpl, int log_no )
	{

	DataPointLong dpl_aggd;

	if (log_no == _LOGGER_MAX_PHYSICAL_LOGGERS) { return;}


	//aggregate
	
	theDataAggregators[log_no].aggregate( dpl );

	// if psize is turning point
	if (sizeArray[log_no-1]==_LOGGER_DIVIDE_STEP)
	{

		//getdata
		dpl_aggd = theDataAggregators[log_no].getData();

		//store
		thePhysicalLoggers[log_no]->push( dpl_aggd );
		sizeArray[log_no]++;

		//aggregate highlevel
		aggregate( dpl_aggd, log_no + 1 );

		//beginnextpoint
		theDataAggregators[log_no].beginNextPoint();
		sizeArray[log_no-1]=0;


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





  DataPointVectorRCPtr Logger::getData( RealParam aStartTime,
					RealParam anEndTime,
					RealParam anInterval ) const
  {
    if (thePhysicalLoggers[0]->empty())
	{
	return anEmptyVector();
	}

	
    //choose appropriate physlogger
    int log_no(_LOGGER_MAX_PHYSICAL_LOGGERS);

// set up output vector
    DataPointVectorIterator 
      range( static_cast<DataPointVectorIterator>
	     ( ( anEndTime - aStartTime ) / anInterval ) );
    //this is a technical adjustment, because I realized that sometimes
    //conversion from real is flawed: rounding error
    Real range_pre( ( anEndTime - aStartTime ) / anInterval );

    if ( ( static_cast<Real>(range) ) + 0.9999 < range_pre ) 
	{
	    range++;
	}
	
    range++;

    Real avg_interval;
    do{
    --log_no;
    avg_interval=thePhysicalLoggers[log_no]->get_avg_interval();


	}
    while (((avg_interval>(anInterval/3))||(avg_interval==0.0)||(thePhysicalLoggers[log_no]->size()<range))&&
	    (log_no>0));


    Real theStartTime ( thePhysicalLoggers[log_no]->front().getTime() );
    Real theEndTime ( thePhysicalLoggers[log_no]->back().getTime() );
    Real time_per_step( ( theEndTime - theStartTime ) /
		    ( thePhysicalLoggers[log_no]->end() - thePhysicalLoggers[log_no]->begin() ) );


	theStartTime = aStartTime;
	theEndTime = anEndTime;




    DataPointVectorPtr aDataPointVector( new DataPointVector( range, 5 ) );


//set uo iterators
    PhysicalLoggerIterator 
      vectorslice_end( thePhysicalLoggers[log_no]->upper_bound_linear_estimate
    					( thePhysicalLoggers[log_no]->begin(),
					  thePhysicalLoggers[log_no]->end(),
					  theEndTime,
					  time_per_step ) );


    
    PhysicalLoggerIterator 
      vectorslice_start( thePhysicalLoggers[log_no]->lower_bound_linear_estimate
    						( thePhysicalLoggers[log_no]->begin(),
						   vectorslice_end,
						   theStartTime,
						   time_per_step ) );

	// start from vectorslice start to vectorslice end, scan through all datapoints
	
	PhysicalLoggerIterator loggerCounter(vectorslice_start);	
	Real targetTime( theStartTime + anInterval );
	DataPointLong readDpl(thePhysicalLoggers[log_no]->at(loggerCounter));
	DataPointLong dp1;

	DataPointAggregator theAggregator;
	theAggregator.aggregate( readDpl );
	for (DataPointVectorIterator elementCount=0;elementCount<range;elementCount++)
		{
		do 
			{

			if ((loggerCounter < vectorslice_end)&&(readDpl.getTime() < targetTime))
				{ 
				loggerCounter++;
				readDpl = thePhysicalLoggers[log_no]->at(loggerCounter);


				}
			theAggregator.aggregate( readDpl );

			}
		while((readDpl.getTime() < targetTime ) && (loggerCounter < vectorslice_end));
		aDataPointVector->asLong(elementCount) = theAggregator.getData();


		theAggregator.beginNextPoint();

		targetTime += anInterval;
		}


    return DataPointVectorRCPtr( aDataPointVector );
  }


} // namespace libecs

