//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2000-2004 Keio University
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
// E-Cell Project.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>
// modified by Koichi Takahashi <shafi@e-cell.org>



#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include <vector>

#include <boost/utility.hpp>

#include "libecs.hpp"
#include "LoggerAdapter.hpp"
#include "PhysicalLogger.hpp"
#include "DataPointVector.hpp"


namespace libecs
{


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
    :
    private boost::noncopyable
  {

  public:

    DECLARE_TYPE( PhysicalLogger::size_type, size_type );
    
    // enumeration for logging policy
    enum Policy
      {
	STEP_SIZE = 0,
	TIME_INTERVAL,
	END_POLICY,
	MAX_SPACE
      };


  public:

    /**
       Constructor.

       Takes up the ownership of the given LoggerAdapter.

    */
  
    Logger( LoggerAdapterPtr aLoggerAdapter );
  
    /// Destructor

    ~Logger( void );


    /**
    
    Sets logging policy that is a vector of 4 numerical values. 
    0 (int)  - minimum step size between logs
    1 (real) - minimum time interval between logs
    2 (int) - action to be taken when disk space runs out
    3 (int) - user set max disk space, if 0 nothing 
    
    */

    void setLoggerPolicy( IntegerParam aMinimumStep,
			  RealParam    aMinimumTimeInterval,
			  IntegerParam anEndPolicy,
			  IntegerParam aMaxSpace );

    /**
    
    Sets logging policy as a PolymorphVector of 4 numerical values. 
    
    */

    ECELL_API void setLoggerPolicy( PolymorphCref aParamList );


    /**

    Returns logging policy vector.

    */

    ECELL_API const Polymorph getLoggerPolicy( void );

    /**

      Log current value that theLoggerAdapter gives with aTime.

    */

    void log( RealParam aTime );


    /**
       Returns contents of the whole logger.

    */

    ECELL_API DataPointVectorSharedPtr getData( void ) const;

    /**
       Returns a slice of the data from aStartTime to anEndTime.

    */

    ECELL_API DataPointVectorSharedPtr getData( RealParam aStartTime,
				      RealParam anEndTime ) const;

    /**
       Returns a summary of the data from aStartTime to anEndTime with
       intervals anInterval between data elements.
    */

    ECELL_API DataPointVectorSharedPtr getData( RealParam aStartTime,
				      RealParam anEndTime, 
				      RealParam anInterval ) const;
    


    /**
       Returns time of the first element  in Logger.
    */

    ECELL_API const Real getStartTime( void ) const;

    /**
       Returns time of the last element in Logger
    */

    ECELL_API const Real getEndTime( void ) const;

    /**
      Returns size of logger
    */

    const size_type getSize() const
    {
      return thePhysicalLogger.size();
    }

    /**
       DEPRECATED - Use setLoggerPolicy 
    */

    ECELL_API void setMinimumInterval( RealParam anInterval );

    /**
       DEPRECATED - Use getLoggerPolicy

    */

    const Real getMinimumInterval( void ) const
    {
      return theMinimumInterval;
    }



    /**
       This method does nothing as of version 3.1.103.
    */

    void flush()
    {
      ; // do nothing
    }


  protected:

    /**

    @internal

    */

    DataPointVectorIterator binary_search( DataPointVectorIterator begin,
					   DataPointVectorIterator end,
					   RealParam t ) 
    {
      return thePhysicalLogger.lower_bound( thePhysicalLogger.begin(), 
					    thePhysicalLogger.end(), 
					    t );
    }
    
  protected:

    /**
       Writes data (aTime, aValue ) onto the logger
    */

    void pushData( RealParam aTime, RealParam aValue )
    {
      thePhysicalLogger.push( DataPoint( aTime, aValue ) );
    }

    static DataPointVectorSharedPtr createEmptyVector();

  private:

    /// no default constructor
    Logger( void );


  private:

    /// Data members

    PhysicalLogger              thePhysicalLogger;

    LoggerAdapterPtr                     theLoggerAdapter;

    PhysicalLogger::size_type   theStepCounter;
    Integer                              theMinimumStep; 

    Real                                 theLastTime;
    Real                                 theMinimumInterval;

  };


  /** @} */ // logging module

} // namespace libecs


#endif /* __LOGGER_HPP */

