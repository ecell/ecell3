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
// written by Masayuki Okayama <smash@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>



#if !defined(__LOGGER_HPP)
#define __LOGGER_HPP

#include "libecs.hpp"
#include "Model.hpp"
#include "Polymorph.hpp"
#include "PhysicalLogger.hpp"
#include "DataPointVector.hpp"
#define _LOGGER_MAX_PHYSICAL_LOGGERS 5
#define _LOGGER_DIVIDE_STEP 200

namespace libecs
{


  /** @addtogroup logging The Data Logging Module.
      The Data Logging Module.

      @ingroup libecs
      
      @{ 
   */ 

  /** @file */


  class LoggerAdapter
  {

  public:

    virtual ~LoggerAdapter();

    virtual const Real getValue() const = 0;

  protected:

    LoggerAdapter();

  };


  /**
   
  */

  class Logger
  {
  public:

    typedef DataPointVectorIterator const_iterator;


  public:

    /**
       Constructor

    */
  
    //    explicit Logger( ModelCref aModel, PropertySlotRef aPropertySlot );
    //    explicit Logger( PropertySlotRef aPropertySlot );
    explicit Logger( LoggerAdapterPtr aLoggerAdapter );
  
    /// Destructor

    ~Logger( void );


    void log( const Real aTime );


    /**

    */

    DataPointVectorRCPtr getData( void ) const;

    /**


    @note It is assumed for both following getData methods that the
       Time values returned by GetCurrentTime method are monotonously
       increasing, therefore
       - a newer theTime is always greater than previous
       - no 2 theTime values are the same 
    */

    DataPointVectorRCPtr getData( RealCref aStartTine,
				  RealCref anEndTime ) const;

    /**
    
    */

    DataPointVectorRCPtr getData( RealCref aStartTime,
				  RealCref anEndTime, 
				  RealCref anInterval ) const;
    


    /**

    */

    Real getStartTime( void ) const;

    /**

    */

    Real getEndTime( void ) const;


    const int getSize() const
    {
      return thePhysicalLoggers[0]->size();
    }

    /**

    */

    void setMinimumInterval( RealCref anInterval );

    /**

    */

    RealCref getMinimumInterval( void ) const
    {
      return theMinimumInterval;
    }


    /**

    */

    void appendData( RealCref aTime, RealCref aValue );


    /**

    */

    void flush();


  protected:

    /**

    @internal

    */

    const_iterator binary_search( const_iterator begin,
				  const_iterator end,
				  RealCref t ) 
    {
      return thePhysicalLoggers[0]->lower_bound( thePhysicalLoggers[0]->begin(), 
    					    thePhysicalLoggers[0]->end(), 
					    t );
    }
    

  private:
    
    DataPointVectorRCPtr anEmptyVector(void) const;

    // no copy constructor
  
    Logger( LoggerCref );

    /// Assignment operator is hidden
  
    Logger& operator=( const Logger& );

    /// no default constructor

    Logger( void );
  



  private:

    /// Data members

    //    PropertySlotRef      thePropertySlot;
    LoggerAdapterPtr     theLoggerAdapter;

    PhysicalLogger*	 thePhysicalLoggers[_LOGGER_MAX_PHYSICAL_LOGGERS];

    void aggregate( DataPointLong , int );

    DataPointAggregator* theDataAggregators;
    Real                 theLastTime;

    Real                 theMinimumInterval;

  };


  /** @} */ // logging module

} // namespace libecs


#endif /* __LOGGER_HPP */

