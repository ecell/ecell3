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
// modified by Gabor Bereczki <gabor.bereczki@talk21.com> (14/04/2002)

#if !defined( __EMC_LOGGER_HPP )
#define __EMC_LOGGER_HPP

#include "libecs/libecs.hpp"
#include "libecs/Logger.hpp"
#include "LoggerImplementation.hpp"
#include "LocalLoggerImplementation.hpp"

namespace libemc
{

  class EmcLogger
  {

  public:

    EmcLogger( libecs::LoggerPtr aLoggerPtr )
      :
      theLoggerImplementation( NULLPTR )
    {
      setLogger( aLoggerPtr );
    }

    virtual ~EmcLogger( ) { }

    const libecs::DataPointVectorRCPtr getData( void ) 
    {
      return theLoggerImplementation->getData();
    }

    const libecs::DataPointVectorRCPtr
    getData( libecs::RealCref start, libecs::RealCref end ) 
    {
      return theLoggerImplementation->getData( start, end );
    }

    const libecs::DataPointVectorRCPtr
    getData( libecs::RealCref start, libecs::RealCref end, 
	     libecs::RealCref interval ) 
    {
      return theLoggerImplementation->getData( start, end, interval );
    }

    void setLogger( LoggerPtr lptr )
    {
      delete theLoggerImplementation;
      theLoggerImplementation = new LocalLoggerImplementation( lptr );
    }

    const libecs::String getName() const
    {
      return theLoggerImplementation->getName();
    }

    const libecs::Real getStartTime() 
    {
      return theLoggerImplementation->getStartTime();
    }

    const libecs::Real getEndTime() 
    {
      return theLoggerImplementation->getEndTime();
    }

    const libecs::Real getMinimumInterval( void ) const
    {
      return theLoggerImplementation->getMinimumInterval();
    }

    const libecs::Real getCurrentInterval( void ) const
    {
      return theLoggerImplementation->getCurrentInterval();
    }

    const libecs::Int getSize( void ) const
    {
      return theLoggerImplementation->getSize();
    }

    // for testing only
    void appendData(libecs::RealCref aValue)
    {
      theLoggerImplementation->appendData(aValue);
    }

  private:

    // hidden default constructor
    EmcLogger();

  private:

    LoggerImplementation* theLoggerImplementation;
  };


} // namespace libemc

#endif
