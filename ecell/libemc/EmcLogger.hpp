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


#if !defined( __EMC_LOGGER_HPP )
#define __EMC_LOGGER_HPP

#include "libecs/libecs.hpp"
#include "libecs/Logger.hpp"
#include "LoggerImplementation.hpp"
#include "LocalLoggerImplementation.hpp"

namespace libemc
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  class EmcLogger
  {

  public:

    EmcLogger( void )
      :
      theLoggerImplementation( NULLPTR )
    {
      ; // do nothing
    }

    virtual ~EmcLogger( ) { }

    const libecs::Logger::DataPointVector
    getData( void ) const
    {
      return theLoggerImplementation->getData();
    }

    const libecs::Logger::DataPointVector
    getData( libecs::RealCref start, libecs::RealCref end ) const
    {
      return theLoggerImplementation->getData( start, end );
    }

    const libecs::Logger::DataPointVector
    getData( libecs::RealCref start, libecs::RealCref end, libecs::RealCref interval ) const
    {
      return theLoggerImplementation->getData( start, end, interval );
    }

    void setLogger( LoggerCptr lptr )
    {
      delete theLoggerImplementation;
      theLoggerImplementation = new LocalLoggerImplementation( lptr );
    }

    libecs::RealCref getStartTime() const
    {
      return theLoggerImplementation->getStartTime();
    }

    libecs::RealCref getEndTime() const
    {
      return theLoggerImplementation->getEndTime();
    }

  private:

    LoggerImplementation* theLoggerImplementation;

  };

  /** @} */ //end of libecs_module 

} // namespace libemc

#endif
