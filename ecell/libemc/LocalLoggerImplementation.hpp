//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
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


#if !defined( __LOCAL_LOGGER_IMPLEMENTATION_HPP )
#define       __LOCAL_LOGGER_IMPLEMENTATION_HPP

#include "libecs/libecs.hpp"
#include "libecs/Logger.hpp"

#include "LoggerImplementation.hpp"

namespace libemc
{

  typedef libecs::LoggerCptr LoggerCptr;

  class LocalLoggerImplementation
    :
    public LoggerImplementation
  {

  public:

    LocalLoggerImplementation( void );

    LocalLoggerImplementation( LoggerCptr );

    virtual ~LocalLoggerImplementation( );

    libecs::Logger::DataPointVectorCref
    getData( void ) const;

    libecs::Logger::DataPointVectorCref
    getData( libecs::RealCref start,
	     libecs::RealCref end ) const;

    libecs::Logger::DataPointVectorCref
    getData( libecs::RealCref start,
	     libecs::RealCref end,
	     libecs::RealCref interval ) const;

    libecs::RealCref getStartTime() const
    {
      return theLogger.getStartTime();
    }

    libecs::RealCref getEndTime() const
    {
      return theLogger.getEndTime();
    }

  private:

    libecs::LoggerCref theLogger;
    
  };


} // namespace libemc

#endif
