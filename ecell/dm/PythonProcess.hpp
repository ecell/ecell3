//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2003 Keio University
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
// authors:
// Kouichi Takahashi <shafi@e-cell.org>
// Nayuta Iwata
//
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#ifndef __PYTHONPROCESS_HPP
#define __PYTHONPROCESS_HPP

#include "PythonProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( PythonProcess, PythonProcessBase )
{

public:

  LIBECS_DM_OBJECT( PythonProcess, Process )
    {
      INHERIT_PROPERTIES( PythonProcessBase );

      PROPERTYSLOT_SET_GET( String, ProcessMethod );
      PROPERTYSLOT_SET_GET( String, InitializeMethod );
    }

  PythonProcess()
    :
    theIsContinuous( false )
  {

    setInitializeMethod( "" );
    setProcessMethod( "" );

    //FIXME: additional properties:
    // Unidirectional   -> call declareUnidirectional() in initialize()
    //                     if this is set
  }

  ~PythonProcess()
  {
    ; // do nothing
  }

  SET_METHOD( Int, IsContinuous )
  {
    theIsContinuous = value;
  }

  virtual const bool isContinuous() const
  {
    return theIsContinuous;
  }

  //  GET_METHOD( Int, IsContinuous )
  //  {
  //    return theIsContinuous;
  //  }


  SET_METHOD( String, ProcessMethod )
  {
    theProcessMethod = value;

    theCompiledProcessMethod = compilePythonCode( theProcessMethod,
						  getFullID().getString() +
						  ":ProcessMethod",
						  Py_file_input );

    // error check
  }

  GET_METHOD( String, ProcessMethod )
  {
    return theProcessMethod;
  }


  SET_METHOD( String, InitializeMethod )
  {
    theInitializeMethod = value;

    theCompiledInitializeMethod = compilePythonCode( theInitializeMethod,
						     getFullID().getString() +
						     ":InitializeMethod",
						     Py_file_input );
  }

  GET_METHOD( String, InitializeMethod )
  {
    return theInitializeMethod;
  }

  virtual void initialize();
  virtual void process();


protected:

  String    theProcessMethod;
  String    theInitializeMethod;

  python::object theCompiledProcessMethod;
  python::object theCompiledInitializeMethod;

  bool theIsContinuous;
};




#endif /* __PYTHONPROCESS_HPP */
