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
// modified by Gabor Bereczki <gabor.bereczki@talk21.com> (14/04/2002)

#include "libecs/PropertyInterface.hpp"

#include "LocalLoggerImplementation.hpp"
namespace libemc
{

  using namespace libecs;

  //  LocalLoggerImplementation::
  //  LocalLoggerImplementation( void )
  //    :
  //    theLogger( Logger() ) 
  //  {
  //    ; // do nothing
  //  }

  LocalLoggerImplementation::
  LocalLoggerImplementation( LoggerPtr lptr )
    :
    theLogger( *lptr ) 
  {
    ; // do nothing
  }


  LocalLoggerImplementation::~LocalLoggerImplementation( )
  {
    ;
  }

  const libecs::DataPointVectorRCPtr
  LocalLoggerImplementation::getData( libecs::RealCref start,
				      libecs::RealCref end,
				      libecs::RealCref interval ) 
  {
    return theLogger.getData( start, end, interval );
  }

  const libecs::DataPointVectorRCPtr
  LocalLoggerImplementation::getData( libecs::RealCref start,
				      libecs::RealCref end ) 				      
  {
    return theLogger.getData( start, end );
  }

  const libecs::DataPointVectorRCPtr
  LocalLoggerImplementation::getData( void ) 
				      
  {
    return theLogger.getData();
  }



} // namespace libemc
