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
#include "PyEcs.hpp"

using libecs::Real;


void PyLogger::init_type()
{
  behaviors().name("Logger");
  behaviors().doc("Logger Python class");

  add_varargs_method( "getData", &PyLogger::getData );
  add_varargs_method( "getStartTime", &PyLogger::getStartTime );
  add_varargs_method( "getEndTime", &PyLogger::getEndTime );
}

Object PyLogger::getData( const Py::Tuple& args )
{
  //ECS_TRY;
  args.verify_length( 0, 3 );
  switch( args.length() )
    {

    case 0:
      return convertVector( EmcLogger::getData() );

    case 2:
      return convertVector( EmcLogger
			    ::getData( Real( static_cast<Py::Float>
						      ( args[0] ) ),
						Real( static_cast<Py::Float>
						      ( args[1] ) )  ) );

    case 3:
      return convertVector( EmcLogger
			    ::getData( Real( static_cast<Py::Float>
					     ( args[0] ) ),
				       Real( static_cast<Py::Float>
					     ( args[1] ) ),
				       Real( static_cast<Py::Float>
					     ( args[2] ) ) ) );
      break;

    default:
      throw Py::IndexError ( "Unexpected number of arguments." );

    }


  //ECS_CATCH;
}

Object PyLogger::getStartTime( const Py::Tuple& args )
{
  //ECS_TRY;
  args.verify_length( 0 );

  Py::Float aReturnedPyFloat( EmcLogger::getStartTime() );

  return aReturnedPyFloat;

  //ECS_CATCH;
}

Object PyLogger::getEndTime( const Py::Tuple& args )
{
  //ECS_TRY;
  args.verify_length( 0 );

  Py::Float aReturnedPyFloat( EmcLogger::getEndTime() );

  return aReturnedPyFloat;

  //ECS_CATCH;
}

Object PyLogger::convertVector( libecs::Logger::DataPointVectorCref aVector )
{
  //ECS_TRY;
  // FIXME: should return Numeric::array
  Py::Tuple aReturnedPyTuple( aVector.size() );
  int i(0);
  for(libecs::Logger::const_iterator anItr(aVector.begin());
      anItr < aVector.end();
      ++anItr)
    {
      Py::Tuple aPyTuple( 2 );
      aPyTuple[0] = Py::Float( (*anItr)->getTime() );
      aPyTuple[1] = Py::Float( (*anItr)->getValue() );
      cerr << libecs::UVariable( (*anItr)->getTime() ).asString() << "\t"
	   << libecs::UVariable( (*anItr)->getValue() ).asString() << endl;
      aReturnedPyTuple[i] = aPyTuple;
      ++i;
    }
  
  return aReturnedPyTuple;
  //ECS_CATCH;
}  


