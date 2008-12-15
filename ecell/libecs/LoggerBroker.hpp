//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
// written by Masayuki Okayama <smash@e-cell.org>,
// E-Cell Project.
//


#if !defined(__LOGGER_BROKER_HPP)
#define __LOGGER_BROKER_HPP

#include <map>

#include "libecs.hpp"
#include "FullID.hpp"
#include "Logger.hpp"

namespace libecs
{
  // forward declaration
  class Model;

  /** @addtogroup logging
   *@{
   */

  /** @file */

  /**
     LoggerBroker creates and administrates Loggers in a model.

     This class creates, holds in a map which associates FullPN with a Logger,
     and responds to requests to Loggers.

     @see FullPN
     @see Logger

  */

  class LIBECS_API LoggerBroker
  {

  public:

    DECLARE_MAP( const FullPN, LoggerPtr, std::less<const FullPN>, LoggerMap );

    LoggerBroker();

    ~LoggerBroker();

    void setModel( Model* model )
    {
      theModel = model;
    }

    Model* getModel()
    {
      return theModel;
    }

    /**
       Get or create a Logger for a PropertySlot.

       This method first look for a Logger object which is logging
       the specified PropertySlot, and if it is found, returns the
       Logger.  If there is no Logger connected to the PropertySlot yet,
       it creates and returns a new Logger.  

       FIXME: doc for interval needed

       @param aFullPN     a FullPN of the requested FullPN
       @param anInterval  a logging interval
       @return a borrowed pointer to the Logger
       
    */

    LoggerPtr getLogger( FullPNCref aFullPN ) const;

    LoggerPtr createLogger( FullPNCref aFullPN, PolymorphVectorCref aParamList );

    /**
       Flush the data in all the Loggers immediately.

       Usually Loggers record data with logging intervals.  This method
       orders every Logger to write the data immediately ignoring the
       logging interval.
    
    */

    void flush();


    /**
       Get a const reference to the LoggerMap.

       Use this method for const operations such as LoggerMap::size() 
       and LoggerMap::begin().

       @return a const reference to the LoggerMap.
    */

    LoggerMapCref getLoggerMap() const
    {
      return theLoggerMap;
    }

  private:
    
    Model* getModel() const
    {
      return theModel;
    }


    // prevent copy
    LoggerBroker( LoggerBrokerCref );
    LoggerBrokerRef operator=( const LoggerBroker& );

  private:

    LoggerMap     theLoggerMap;
    Model*        theModel;

  };

  //@}
  
} // namespace libecs

#endif



