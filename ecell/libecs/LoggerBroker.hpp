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
#include "Logger.hpp"

namespace libecs
{

  class LoggerBroker
  {
  public:
    LoggerBroker( RootSystemPtr aRootSystemPtr )
      :
      theRootSystem( aRootSystemPtr )
    {
      ; // do nothing
    }
    
    LoggerPtr getLogger( StringCref, StringCref );
    //    LoggerPtr getLogger( FQPICref );

    class PairOfStrings
    {
    public:
      PairOfStrings( StringCref first , StringCref second )
	:
	thePair( first, second )
      {
	;
      }
      
      const std::pair<String, String>& getPair( void ) const
      {
	return thePair;
      }

      bool operator < ( const PairOfStrings& rhs ) const
      {
	if( rhs.getPair().first > this->getPair().first )
	  {
	    return true;
	  }
	return false;
      }


    private:
      const std::pair<String, String> thePair;
    };
    

    DECLARE_MAP( const PairOfStrings, LoggerPtr, std::less<const PairOfStrings>, LoggerMap );
    typedef std::pair<const PairOfStrings, LoggerPtr> PairInLoggerMap;
  protected:
    
    void appendLogger( StringCref, StringCref );
    //    void appendLogger( FQPICref );
    
    
    
  private:
    LoggerBroker( LoggerBrokerCref );
    LoggerBrokerRef operator=( const LoggerBroker& );

    LoggerMap     theLoggerMap;
    RootSystemPtr theRootSystem;
    
  };
  
} // namespace libecs

#endif



