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
// modified by Gabor Bereczki <gabor.bereczki@talk21.com>
// 14/04/2002

#ifndef ___LOGGER_IMPLEMENTATION_H___
#define ___LOGGER_IMPLEMENTATION_H___

#include "libecs/libecs.hpp"
#include "libecs/Logger.hpp" 

namespace libemc
{

  /**
     Pure virtual base class (interface definition) of logger
     implementation.
  */

  class LoggerImplementation
  {

  public:

    LoggerImplementation(){} 
    virtual ~LoggerImplementation(){} 

    virtual const libecs::DataPointVectorRCPtr
    getData( void ) = 0;

    virtual const libecs::DataPointVectorRCPtr
    getData( libecs::RealCref start,
	     libecs::RealCref end ) = 0;

    virtual const libecs::DataPointVectorRCPtr
    getData( libecs::RealCref start,
	     libecs::RealCref end,
	     libecs::RealCref interval ) = 0;

    virtual const libecs::String getName() const = 0;
    virtual const libecs::Real   getStartTime() = 0;
    virtual const libecs::Real   getEndTime() = 0;
    virtual const libecs::Int    getSize() = 0;
    virtual const libecs::Real   getMinimumInterval( void ) const = 0;
    virtual const libecs::Real   getCurrentInterval( void ) const = 0;

    virtual const void appendData(libecs::RealCref aValue) = 0;

  };


} // namespace libecs

#endif


