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
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Institute for Advanced Biosciences, Keio University.
//


#ifndef __PYSIMULATOR_HPP
#define __PYSIMULATOR_HPP

#include "libemc/Simulator.hpp"

#include "CXX/Extensions.hxx"

using Py::Object;
using Py::Callable;
using Py::Tuple;
using Py::PythonExtension;

class PySimulator 
  :
  public PythonExtension< PySimulator >,
  public libemc::Simulator
{
public:
  
  PySimulator();
  virtual ~PySimulator(){};

  static void init_type();

  Object createEntity           ( const Tuple& args );
  Object setProperty            ( const Tuple& args );
  Object getProperty            ( const Tuple& args );
  Object step                   ( const Tuple& args );
  Object initialize             ( const Tuple& args );
  Object getLogger              ( const Tuple& args );
  Object getLoggerList          ( const Tuple& args );
  Object run                    ( const Tuple& args );
  Object stop                   ( const Tuple& args );
  Object setPendingEventChecker ( const Tuple& args );
  Object setEventHandler        ( const Tuple& args );

private:

  static void callPendingEventChecker();
  static void callEventHandler();  
  static Callable* thePendingEventChecker;
  static Callable* theEventHandler;
  Object theTmpPendingEventChecker;
  Object theTmpEventHandler;

};

#endif   /* __PYSIMULATOR_HPP */








