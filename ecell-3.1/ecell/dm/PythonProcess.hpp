//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// authors:
// Koichi Takahashi <shafi@e-cell.org>
// Nayuta Iwata
//
// E-Cell Project.
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

      PROPERTYSLOT_SET_GET( Integer, IsContinuous );
      PROPERTYSLOT_SET_GET( String, FireMethod );
      PROPERTYSLOT_SET_GET( String, InitializeMethod );
    }

  PythonProcess()
    :
    theIsContinuous( false )
  {

    setInitializeMethod( "" );
    setFireMethod( "" );

    //FIXME: additional properties:
    // Unidirectional   -> call declareUnidirectional() in initialize()
    //                     if this is set
  }

  virtual ~PythonProcess()
  {
    ; // do nothing
  }

  virtual const bool isContinuous() const
  {
    return theIsContinuous;
  }

  SET_METHOD( Integer, IsContinuous )
  {
    theIsContinuous = value;
  }

  SET_METHOD( String, FireMethod )
  {
    theFireMethod = value;

    theCompiledFireMethod = compilePythonCode( theFireMethod,
						  getFullID().getString() +
						  ":FireMethod",
						  Py_file_input );

    // error check
  }

  GET_METHOD( String, FireMethod )
  {
    return theFireMethod;
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
  virtual void fire();


protected:

  String    theFireMethod;
  String    theInitializeMethod;

  python::object theCompiledFireMethod;
  python::object theCompiledInitializeMethod;

  bool theIsContinuous;
};




#endif /* __PYTHONPROCESS_HPP */
