//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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
// written by Kouichi Takahashi <shafi@e-cell.org> for
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#if ! defined( __PYECS_HPP )
#define __PYECS_HPP

// for memcpy()
#include <string.h>

#include <boost/cast.hpp>
#include <boost/python.hpp>

#include <Numeric/arrayobject.h>

#include "libecs/libecs.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/DataPointVector.hpp"

#include "libemc/libemc.hpp"

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

class Polymorph_to_python
{
public:

  static PyObject* convert( libecs::PolymorphCref aPolymorph )
  {
    switch( aPolymorph.getType() )
      {
      case libecs::Polymorph::REAL :
	return PyFloat_FromDouble( aPolymorph.asReal() );
      case libecs::Polymorph::INTEGER :
	return PyInt_FromLong( aPolymorph.asInteger() );
      case libecs::Polymorph::POLYMORPH_VECTOR :
	return PolymorphVector_to_PyTuple( aPolymorph.asPolymorphVector() );
      case libecs::Polymorph::STRING :
      case libecs::Polymorph::NONE :
      default: // should this default be an error?
	return PyString_FromString( aPolymorph.asString().c_str() );
      }
  }

  static PyObject* 
  PolymorphVector_to_PyTuple( libecs::PolymorphVectorCref aVector )
  {
    libecs::PolymorphVector::size_type aSize( aVector.size() );
    
    PyObject* aPyTuple( PyTuple_New( aSize ) );
    
    for( size_t i( 0 ) ; i < aSize ; ++i )
      {
	PyTuple_SetItem( aPyTuple, i, 
			 Polymorph_to_python::convert( aVector[i] ) );
      }
    
    return aPyTuple;
  }


};


class DataPointVectorRCPtr_to_python
{
public:

  static PyObject* 
  convert( const libecs::DataPointVectorRCPtr& aVectorRCPtr )
  {
    // here starts an ugly C hack :-/

    libecs::DataPointVectorCref aVector( *aVectorRCPtr );

    int aDimensions[2] = { aVector.getSize(),
			   aVector.getElementSize() / sizeof( double ) };


    PyArrayObject* anArrayObject( reinterpret_cast<PyArrayObject*>
				  ( PyArray_FromDims( 2, aDimensions, 
						      PyArray_DOUBLE ) ) );

    memcpy( anArrayObject->data, aVector.getRawArray(),   
	    aVector.getSize() * aVector.getElementSize() );
    
    return PyArray_Return( anArrayObject );
  }

};


class register_Polymorph_from_python
{
public:

  register_Polymorph_from_python()
  {
    python::converter::
      registry::insert( &convertible, &construct,
			python::type_id<libecs::Polymorph>() );
  }

  static void* convertible( PyObject* aPyObject )
  {
    // always passes the test for efficiency.  overload won't work.
    return aPyObject;
  }

  static void construct( PyObject* aPyObjectPtr, 
			 python::converter::
			 rvalue_from_python_stage1_data* data )
  {
    void* storage( ( ( python::converter::
		       rvalue_from_python_storage<libecs::Polymorph>*) 
		     data )->storage.bytes );

    new (storage) libecs::Polymorph( Polymorph_from_python( aPyObjectPtr ) );
    data->convertible = storage;
  }


  static const libecs::Polymorph 
  Polymorph_from_python( PyObject* aPyObjectPtr )
  {
    if( PyFloat_Check( aPyObjectPtr ) )
      {
	return PyFloat_AS_DOUBLE( aPyObjectPtr );
      }
    else if( PyInt_Check( aPyObjectPtr ) )
      {
	return PyInt_AS_LONG( aPyObjectPtr );
      }
    else if ( PyTuple_Check( aPyObjectPtr ) )
      {
	return to_PolymorphVector( aPyObjectPtr );
      }
    else if( PyList_Check( aPyObjectPtr ) )
      {
	return to_PolymorphVector( PyList_AsTuple( aPyObjectPtr ) );
      }      
    else if( PyString_Check( aPyObjectPtr ) )
      {
	return libecs::Polymorph( PyString_AsString( aPyObjectPtr ) );
      }
    else
      {
	// conversion is failed. ( convert with repr() ? )
	PyErr_SetString( PyExc_TypeError, 
			 "Unacceptable type of an object in the tuple." );
	python::throw_error_already_set();
      }
  }


  static const libecs::PolymorphVector 
  to_PolymorphVector( PyObject* aPyObjectPtr )
  {
    std::size_t aSize( PyTuple_GET_SIZE( aPyObjectPtr ) );
      
    libecs::PolymorphVector aVector;
    aVector.reserve( aSize );
      
    for ( std::size_t i( 0 ); i < aSize; ++i )
      {
	aVector.
	  push_back( Polymorph_from_python( PyTuple_GET_ITEM( aPyObjectPtr, 
							      i ) ) );
      }
      
    return aVector;
  }
    
};


class register_EventCheckerRCPtr_from_python
{
public:

  register_EventCheckerRCPtr_from_python()
  {
    python::converter::
      registry::insert( &convertible, &construct,
			python::type_id<libemc::EventCheckerRCPtr>() );
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
      libemc::EventCheckerRCPtr( new PythonEventChecker( aPyObjectPtr ) );

    data->convertible = storage;
  }

};



class register_EventHandlerRCPtr_from_python
{
public:

  register_EventHandlerRCPtr_from_python()
  {
    python::converter::
      registry::insert( &convertible, &construct,
			python::type_id<libemc::EventHandlerRCPtr>() );
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
      libemc::EventHandlerRCPtr( new PythonEventHandler( aPyObjectPtr ) );

    data->convertible = storage;
  }

};


#endif // __PYECS_HPP
