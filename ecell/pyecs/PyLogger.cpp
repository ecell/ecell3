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


#include "PyLogger.hpp"

void PyLogger::init_type()
{
  behaviors().name("Logger");
  behaviors().doc("Logger Python class");

  add_varargs_method( "getData", &PyLogger::getData );
}

Object PyLogger::getData( const Py::Tuple& args )
{
  args.verify_length( 0, 3 );
  libecs::Logger::DataPointVectorCptr aDataPointVector( NULLPTR );
  switch( args.length() )
    {
    case 0:
      aDataPointVector = &EmcLogger::getData();
      break;
    case 2:
      aDataPointVector = &EmcLogger::getData( double( static_cast<Py::Float>(args[0]) ),
					      double( static_cast<Py::Float>(args[1]) )
					      );
      break;
    case 3:
      aDataPointVector = &EmcLogger::getData( double( static_cast<Py::Float>(args[0]) ),
					      double( static_cast<Py::Float>(args[1]) ),
					      double( static_cast<Py::Float>(args[2]) )
					      );
      break;
    }

  Py::Tuple* aPyTupleReturned = new Py::Tuple( aDataPointVector->size() );
  for(int i = 0; i < aDataPointVector->size(); i++ )
    {
      Py::Tuple aPyTuple( 2 );
      aPyTuple[0] = Py::Float( (*aDataPointVector)[i]->getTime() );
      aPyTuple[1] = Py::Float( (*aDataPointVector)[i]->getValue().asReal() );
      (*aPyTupleReturned)[i] = aPyTuple;
    }
  
  return *aPyTupleReturned;
}

