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

#ifndef ___PYECS_H___
#define ___PYECS_H___

#include <exception>

#include "CXX/Extensions.hxx"

/** @defgroup pyecs_module The Pyecs Module 
 * This is the pyecs module 
 * @{ 
 */ 


class PyEcs
  : 
  public Py::ExtensionModule< PyEcs >
{

public:  

  PyEcs();
  ~PyEcs(){};
 
  Py::Object createSimulator( const Py::Tuple& args );
  Py::Object createLogger( const Py::Tuple& args );

private:

};   //end of class PyEcs

extern "C" void initecs();



#define ECS_TRY try {

#define ECS_CATCH\
    }\
  catch( libecs::ExceptionCref e )\
    {\
      throw Py::Exception( e.message() );\
    }\
  catch( const std::exception& e)\
    {\
      throw Py::SystemError( std::string( "E-CELL internal error: " )\
			     + std::string( e.what() ) );\
    }\
  catch( Py::Exception& )\
    {\
      throw;\
    }\
  catch( ... ) \
    {\
      throw Py::SystemError( "E-CELL internal error (unexpected)." );\
    }

/** @} */ //end of pyecs_module 

#endif   /* ___PYECS_H___ */













