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


#if !defined(__LOGGER_BROKER_HPP)
#define __LOGGER_BROKER_HPP

#include <map>

#include "libecs.hpp"
#include "FullID.hpp"
#include "Logger.hpp"

namespace libecs
{

  class LoggerBroker
  {

  public:

    DECLARE_MAP( const FullPN, 
		 LoggerPtr, std::less<const FullPN>, LoggerMap );

    LoggerBroker( RootSystemRef aRootSystem )
      :
      theRootSystem( aRootSystem )
    {
      ; // do nothing
    }
    
    ~LoggerBroker();


    LoggerPtr getLogger( FullPNCref fpn );

    LoggerMapCref getLoggerMap() const
    {
      return theLoggerMap;
    }

  protected:
    
    LoggerPtr createLogger( FullPNCref fpn );
    //    void appendLogger( LoggerPtr );
    
  private:

    // prevent copy
    LoggerBroker( LoggerBrokerCref );
    LoggerBrokerRef operator=( const LoggerBroker& );

  private:

    LoggerMap     theLoggerMap;
    RootSystemRef theRootSystem;
    
  };
  
} // namespace libecs

#endif



