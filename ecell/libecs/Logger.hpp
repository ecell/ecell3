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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>



#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include "libecs.hpp"
#include "LoggerAdapter.hpp"
#include "PhysicalLogger.hpp"
#include "DataPointVector.hpp"


namespace libecs
{


  
  const Integer   _LOGGER_MAX_PHYSICAL_LOGGERS = 5;
  const Integer   _LOGGER_DIVIDE_STEP = 200;
  
  // enumeration for logging policy
  enum  
    {
      _STEP_SIZE,
      _TIME_INTERVAL,
      _END_POLICY,
      _MAX_SPACE
    };



  /** @addtogroup logging The Data Logging Module.
      The Data Logging Module.

      @ingroup libecs
      
      @{ 
   */ 

  /** @file */

  /**

  Logger module for logging and retrieving data runtime.
   
  */

  class Logger
  {


  public:

    typedef DataPointVectorIterator const_iterator;
    typedef PhysicalLogger::iterator phys_iterator;


  public:

    /**
       Constructor

    */
  
    //    explicit Logger( ModelCref aModel, PropertySlotRef aPropertySlot );
    //    explicit Logger( PropertySlotRef aPropertySlot );
    explicit Logger( LoggerAdapterPtr aLoggerAdapter );
  
    /// Destructor

    ~Logger( void );


    /**
    
    Sets logging policy that is a vector of 4 numerical values. 
    0 - minimum step size between logs
    1 - minimum time interval between logs
    2 - action to be taken when disk space runs out
    3 - user set max disk space, if 0 nothing 
    

    */

    void setLoggerPolicy( PolymorphCref aParamList );


    /**

    Returns logging policy vector.

    */

    PolymorphCref getLoggerPolicy( void )
    {

      return theLoggingPolicy;
    }


    /**

      Log current value of logger FullPN

    */


    void log( RealParam aTime )

    {
      appendData( aTime, theLoggerAdapter->getValue() );
    }


    /**

    Returns contents of the whole logger.

    */

    DataPointVectorSharedPtr getData( void ) const;

    /**


    @note It is assumed for both following getData methods that the
       Time values returned by GetCurrentTime method are monotonously
       increasing, therefore
       - a newer theTime is always greater than previous
       - no 2 theTime values are the same 
    */

    DataPointVectorSharedPtr getData( RealParam aStartTine,
				  RealParam anEndTime ) const;

    /**
    Returns a summary of the data from aStartTime to anEndTime with at least 
    intervals anInterval between data elements
    */

    DataPointVectorSharedPtr getData( RealParam aStartTime,
				  RealParam anEndTime, 
				  RealParam anInterval ) const;
    


    /**
       Returns time of the first element  in Logger.
    */

    Real getStartTime( void ) const;

    /**
       Returns time of the last element in Logger
    */

    Real getEndTime( void ) const;

    /**
      Returns size of logger
    */

    const int getSize() const
    {
      return thePhysicalLoggers[0]->size();
    }

    /**
       DEPRECATED - Use setLoggerPolicy 
    */

    void setMinimumInterval( RealParam anInterval );

    /**
       DEPRECATED - Use getLoggerPolicy

    */

    const Real getMinimumInterval( void ) const
    {
      return theMinimumInterval;
    }


    /**
       Writes data (aTime, aValue ) onto the logger
    */

    void appendData( RealParam aTime, RealParam aValue );


    /**
       Forces logger to write data even if mimimuminterval or
       step count has not been exceeded.
    */

    void flush();


  protected:

    /**

    @internal

    */

    const_iterator binary_search( const_iterator begin,
				  const_iterator end,
				  RealParam t ) 
    {
      return thePhysicalLoggers[0]->lower_bound( thePhysicalLoggers[0]->begin(), 
						 thePhysicalLoggers[0]->end(), 
						 t );
    }
    

  private:
    
    DataPointVectorSharedPtr anEmptyVector(void) const;

    // no copy constructor
  
    Logger( LoggerCref );

    /// Assignment operator is hidden
  
    Logger& operator=( const Logger& );

    /// no default constructor

    Logger( void );
  



  private:

    /// Data members

    //    PropertySlotRef      thePropertySlot;
    void aggregate( DataPointLong , int );

    LoggerAdapterPtr     theLoggerAdapter;
    PhysicalLogger*	 thePhysicalLoggers[_LOGGER_MAX_PHYSICAL_LOGGERS];
    DataPointAggregator* theDataAggregators;
    Real                 theLastTime;
    const_iterator       theStepCounter;
    Integer              theMinimumStep; //0-minimum step, 1 minimum time 3 end policy 4 max space available in kbytes
    Real                 theMinimumInterval;
    Integer              theSizeArray[_LOGGER_MAX_PHYSICAL_LOGGERS];
    Polymorph	         theLoggingPolicy;
  };


  /** @} */ // logging module

} // namespace libecs


#endif /* __LOGGER_HPP */

