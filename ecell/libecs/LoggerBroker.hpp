//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2000-2002 Keio University
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


#if !defined(__LOGGER_BROKER_HPP)
#define __LOGGER_BROKER_HPP

#include <map>

#include "libecs.hpp"
#include "FullID.hpp"
#include "Logger.hpp"

namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  class LoggerBroker
  {

  public:

    DECLARE_MAP( const FullPN, 
		 LoggerPtr, std::less<const FullPN>, LoggerMap );

    LoggerBroker( ModelRef aModel );

    
    ~LoggerBroker();


    /**
       Get or create a Logger for a PropertySlot.

       This method first look for a Logger object which is logging
       the specified PropertySlot, and if it is found, returns the
       Logger.  If there is no Logger connected to the PropertySlot yet,
       it creates and returns a new Logger.  

       FIXME: doc for interval needed

       \param aFullPN     a FullPN of the requested FullPN
       \param anInterval  a logging interval
       \return a borrowed pointer to the Logger
       
    */

    LoggerPtr getLogger( FullPNCref aFullPN, RealCref anInterval = 0.0 );


    /**
       Flush the data in all the Loggers immediately.

       Usually Loggers record data with logging intervals.  This method
       orders every Logger to write the data immediately ignoring the
       logging interval.
    
    */

    void flush();


    //FIXME: should be private
    LoggerMapCref getLoggerMap() const
    {
      return theLoggerMap;
    }

    //FIXME: should be private
    ModelRef getModel() const
    {
      return theModel;
    }

  protected:
    
    LoggerPtr createLogger( FullPNCref fpn );
    
  private:

    // prevent copy
    LoggerBroker( LoggerBrokerCref );
    LoggerBrokerRef operator=( const LoggerBroker& );

  private:

    LoggerMap     theLoggerMap;
    ModelRef      theModel;

  };

  /** @} */ //end of libecs_module
  
} // namespace libecs

#endif



