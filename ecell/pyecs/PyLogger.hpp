//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio University
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
// E-CELL Project, Institute for Advanced Biosciences, Keio University.
//
// modified by Gabor Bereczki <gabor.bereczki@talk21.com> (14/04/2002)

#if !defined( __PY_LOGGER_HPP )
#define __PY_LOGGER_HPP

#include "libemc/EmcLogger.hpp"
#include "CXX/Extensions.hxx"
using Py::Object;
using Py::Tuple;
using Py::PythonExtension;

using libemc::EmcLogger;



class PyLogger
  :
  public PythonExtension< PyLogger >,
  public EmcLogger
{

public:

  PyLogger()
  {
    ; // do nothing
  }

  PyLogger( libecs::Logger* lptr )
  {
    EmcLogger::setLogger( lptr );
  }

  void setLogger( libecs::Logger* lptr )
  {
    EmcLogger::setLogger( lptr );
  }

  virtual ~PyLogger()
  {
    ; // do nothing
  }

  static void init_type();

  Object getData( const Tuple& args );
  Object getStartTime( const Tuple& args );
  Object getEndTime( const Tuple& args );
  Object appendData( const Tuple& args );
  
private:

  Object convertVector( libecs::DataPointVectorRCPtr );

};


#endif
