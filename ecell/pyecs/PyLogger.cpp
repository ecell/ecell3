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

void PyDataPoint::init_type()
{
  behaviors().name("DataPoint");
  behaviors().doc("DataPoint Python class");

  add_varargs_method( "getTime", &PyDataPoint::getTime );
  add_varargs_method( "getValue", &PyDataPoint::getValue );
}

Object PyDataPoint::getTime( const Tuple& args )
{
  return Py::Float( EmcDataPoint::getTime() );
}
  
Object PyDataPoint::getValue( const Tuple& args )
{
  return Py::Float( EmcDataPoint::getValue().asReal() );
}
  

void PyLogger::init_type()
{
  behaviors().name("Logger");
  behaviors().doc("Logger Python class");

  add_varargs_method( "getData", &PyLogger::getData );
}

Object PyLogger::getData( const Py::Tuple& args )
{
  args.verify_length( 0, 3 );
  Py::List aPyList;
  libecs::Logger::DataPointVectorCptr aDataPointVector( NULLPTR );
  switch( args.length() )
    {
    case 0:
      aDataPointVector = &EmcLogger::getData();
      for(int i = 0; i < aDataPointVector->size(); i++ )
	{
	  aPyList.
	    append( asObject( new PyDataPoint( *(*aDataPointVector)[i] )
			      )
		    );
	}
      break;
    case 2:
      aDataPointVector = &EmcLogger::getData();
      for(int i = 0; i < aDataPointVector->size(); i++ )
	{
	  aPyList.
	    append( asObject( new PyDataPoint( *(*aDataPointVector)[i] )
			      )
		    );
	}
      break;
    case 3:
      aDataPointVector = &EmcLogger::getData();
      for(int i = 0; i < aDataPointVector->size(); i++ )
	{
	  aPyList.
	    append( asObject( new PyDataPoint( *(*aDataPointVector)[i] )
			      )
		    );
	}
      break;
    }

  return Py::Tuple( aPyList );
}

