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


#include "LocalLoggerImplementation.hpp"

namespace libemc
{

  LocalLoggerImplementation::LocalLoggerImplementation( ObjectPtr optr )
    :
    theRealLogger( RealLogger( optr ) ) 
  {
    ; // do nothing
  }

  LocalLoggerImplementation::~LocalLoggerImplementation( )
  {
    ;
  }

  void LocalLoggerImplementation::update( )
  {
    theRealLogger.update( );
  }

  void
  LocalLoggerImplementation::update( RealLogger::containee_type& datapoint )
  {
    theRealLogger.update( datapoint );
  }


  /*
    void LocalLoggerImplementation::push( const RealLogger::containee_type& x )
    {
    theRealLogger.push( x );
    }

    void LocalLoggerImplementation::push( const Real& t, const Real& v )
    {
    theRealLogger.push( t, v );
    } 
  */

} // namespace libemc
