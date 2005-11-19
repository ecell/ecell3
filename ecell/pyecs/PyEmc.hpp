//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org> for
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#if ! defined( __PYEMC_HPP )
#define __PYEMC_HPP


#include <boost/cast.hpp>
#include <boost/python.hpp>

#include "libemc/libemc.hpp"

#include "pyecs/PyEcs.hpp"

namespace python = boost::python;

DECLARE_CLASS( PythonCallable );
DECLARE_CLASS( PythonEventChecker );
DECLARE_CLASS( PythonEventHandler );

class PythonCallable
{
public:

  PythonCallable( PyObject* aPyObjectPtr )
    :
    thePyObject( python::handle<>( aPyObjectPtr ) )
  {
    // this check isn't needed actually, because BPL does this automatically
    if( ! PyCallable_Check( thePyObject.ptr() ) )
      {
	PyErr_SetString( PyExc_TypeError, "Callable object must be given" );
	python::throw_error_already_set();
      }
  }

  virtual ~PythonCallable()
  {
    ; // do nothing
  }

protected:

  python::object thePyObject;
};


class PythonEventChecker
  : 
  public PythonCallable,
  public libemc::EventChecker
{
public:

  PythonEventChecker( PyObject* aPyObjectPtr )
    :
    PythonCallable( aPyObjectPtr )
  {
    ; // do nothing
  }
    
  virtual ~PythonEventChecker() {}

  virtual bool operator()( void ) const
  {
    // check signal
    //    PyErr_CheckSignals();

    // check event.
    // this is faster than just 'return thePyObject()', unfortunately..
    PyObject* aPyObjectPtr( PyObject_CallFunction( thePyObject.ptr(), NULL ) );
    const bool aResult( PyObject_IsTrue( aPyObjectPtr ) );
    Py_DECREF( aPyObjectPtr );

    return aResult;
  }

};

class PythonEventHandler
  : 
  public PythonCallable,
  public libemc::EventHandler
{
public:

  PythonEventHandler( PyObject* aPyObjectPtr )
    :
    PythonCallable( aPyObjectPtr )
  {
    ; // do nothing
  }
    
  virtual ~PythonEventHandler() {}

  virtual void operator()( void ) const
  {
    PyObject_CallFunction( thePyObject.ptr(), NULL );

    // faster than just thePyObject() ....
  }

};


class register_EventCheckerSharedPtr_from_python
{
public:

  register_EventCheckerSharedPtr_from_python()
  {
    python::converter::
      registry::insert( &convertible, &construct,
			python::type_id<libemc::EventCheckerSharedPtr>() );
  }

  static void* convertible( PyObject* aPyObjectPtr )
  {
    if( PyCallable_Check( aPyObjectPtr ) )
      {
	return aPyObjectPtr;
      }
    else
      {
	return 0;
      }
  }

  static void 
  construct( PyObject* aPyObjectPtr, 
	     python::converter::rvalue_from_python_stage1_data* data )
  {
    void* storage( ( ( python::converter::
		       rvalue_from_python_storage<libecs::Polymorph>*) 
		     data )->storage.bytes );

    new (storage) 
      libemc::EventCheckerSharedPtr( new PythonEventChecker( aPyObjectPtr ) );

    data->convertible = storage;
  }

};



class register_EventHandlerSharedPtr_from_python
{
public:

  register_EventHandlerSharedPtr_from_python()
  {
    python::converter::
      registry::insert( &convertible, &construct,
			python::type_id<libemc::EventHandlerSharedPtr>() );
  }

  static void* convertible( PyObject* aPyObjectPtr )
  {
    if( PyCallable_Check( aPyObjectPtr ) )
      {
	return aPyObjectPtr;
      }
    else
      {
	return 0;
      }
  }

  static void construct( PyObject* aPyObjectPtr, 
			 python::converter::
			 rvalue_from_python_stage1_data* data )
  {
    void* storage( ( ( python::converter::
		       rvalue_from_python_storage<libecs::Polymorph>*) 
		     data )->storage.bytes );

    new (storage) 
      libemc::EventHandlerSharedPtr( new PythonEventHandler( aPyObjectPtr ) );

    data->convertible = storage;
  }

};


#endif // __PYEMC_HPP
