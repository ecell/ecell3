#if ! defined( __PYECS_HPP )
#define __PYECS_HPP

// for ::memcpy()
#include <string.h>

#include <boost/cast.hpp>

#include <Numeric/arrayobject.h>

#include "libecs/libecs.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/DataPointVector.hpp"


namespace python = boost::python;

class PythonCallable
{
public:

  PythonCallable( PyObject* aPyObjectPtr )
    :
    thePyObjectPtr( aPyObjectPtr )
  {
    if( ! PyCallable_Check( thePyObjectPtr ) )
      {
	PyErr_SetString( PyExc_TypeError, 
			 "Callable object must be given" );
	python::throw_argument_error();
      }

    Py_INCREF( thePyObjectPtr );
  }
    
  virtual ~PythonCallable()
  {
    Py_DECREF( thePyObjectPtr );
  }

protected:

  PyObject* thePyObjectPtr;
};


class PythonPendingEventChecker
  : 
  public PythonCallable,
  public libemc::PendingEventChecker
{
public:

  PythonPendingEventChecker( PyObject* aPyObjectPtr )
    :
    PythonCallable( aPyObjectPtr )
  {
    ; // do nothing
  }
    
  virtual ~PythonPendingEventChecker() {}

  virtual bool operator()( void ) const
  {
    PyObject* aPyObjectPtr( PyObject_CallFunction( thePyObjectPtr, NULL ) );

    const bool aResult( PyObject_IsTrue( aPyObjectPtr ) );
    Py_DECREF( aPyObjectPtr );

    return aResult;

    // probably faster than this
    //    from_python<bool>( PyObject_CallFunction( thePyObjectPtr, NULL ) );
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
    PyObject_CallFunction( thePyObjectPtr, NULL );
  }

};



BOOST_PYTHON_BEGIN_CONVERSION_NAMESPACE

//
// type conversions between python <-> libecs
//


//
// Polymorph
//

libecs::Polymorph from_python( PyObject* aPyObjectPtr,
			       type<libecs::Polymorph> )
{
  libecs::Polymorph aPolymorph;

  if( PyFloat_Check( aPyObjectPtr ) )
    {
      libecs::Real aReal( BOOST_PYTHON_CONVERSION::
			  from_python( aPyObjectPtr,
				       type<libecs::Real>() ) );
      aPolymorph = aReal;
    }
  else if( PyInt_Check( aPyObjectPtr ) )
    {
      libecs::Int anInt( BOOST_PYTHON_CONVERSION::
			 from_python( aPyObjectPtr,
				      type<long int>()) );
      //					  type<libecs::Int>()) );
      aPolymorph = anInt;
    }
  else if( PyString_Check( aPyObjectPtr ) )
    {
      libecs::String 
	aString( BOOST_PYTHON_CONVERSION::
		 from_python( aPyObjectPtr,
			      type<libecs::String>() ) );
      aPolymorph = aString;
    }
  else
    {
      // convert with repr() ?
      
      PyErr_SetString( PyExc_TypeError, 
		       "Unacceptable type of an object in the tuple." );
      throw_argument_error();
      
    }

  // here I expect named return value optimization
  return aPolymorph;
}

PyObject* to_python( libecs::PolymorphCref aPolymorph )
{
  PyObject* aPyObjectPtr;

  switch( aPolymorph.getType() )
    {
    case libecs::Polymorph::REAL :
      aPyObjectPtr = BOOST_PYTHON_CONVERSION::to_python( aPolymorph.asReal() );
      break;
    case libecs::Polymorph::INT :
      // FIXME: ugly cast... determine the type by autoconf?
      aPyObjectPtr = BOOST_PYTHON_CONVERSION::
	to_python( boost::numeric_cast<long int>( aPolymorph.asInt() ) );
      break;
    case libecs::Polymorph::STRING :
    case libecs::Polymorph::NONE :
    default: // should this default be an error?
      aPyObjectPtr = BOOST_PYTHON_CONVERSION::
	to_python( aPolymorph.asString() );
      break;
    }

  // named return optimization
  return aPyObjectPtr;
}

PyObject* to_python( libecs::PolymorphCptr aPolymorphPtr )
{
  to_python( aPolymorphPtr );
}



//
// PolymorphVector
//

libecs::PolymorphVector from_python( PyObject* aPyObjectPtr, 
				     type<libecs::PolymorphVector> )
{
  ref aRef;

  if ( PyTuple_Check( aPyObjectPtr ) )
    {
      aRef = make_ref( aPyObjectPtr );
    }
  else if( PyList_Check( aPyObjectPtr ) )
    {
      aRef = make_ref( PyList_AsTuple( aPyObjectPtr ) );
    }
  else
    {
      PyErr_SetString( PyExc_TypeError, 
		       "This method only takes tuple or list." );
      throw_argument_error();
    }
  
  tuple aPyTuple( aRef );

  std::size_t aSize( aPyTuple.size() );

  libecs::PolymorphVector aVector;
  aVector.reserve( aSize );

  for ( std::size_t i( 0 ); i < aSize; ++i )
    {
      ref anItemRef( aPyTuple[i] );
      PyObject* aPyObjectPtr( anItemRef.get() ); 

      aVector.push_back( from_python( aPyObjectPtr, 
				      type<libecs::Polymorph>() ) );
    }

  return aVector;
}

libecs::PolymorphVector from_python( PyObject* aPyObjectPtr, 
				     type<libecs::PolymorphVectorCref> )
{
  return from_python( aPyObjectPtr, type<libecs::PolymorphVector>() );
}

PyObject* to_python( libecs::PolymorphVectorCref aVector )
{
  libecs::PolymorphVector::size_type aSize( aVector.size() );
  
  tuple aPyTuple( aSize );

  for( size_t i( 0 ) ; i < aSize ; ++i )
    {
      aPyTuple.set_item( i, BOOST_PYTHON_CONVERSION::to_python( aVector[i] ) );
    }

  return to_python( aPyTuple.get() );
}

PyObject* to_python( libecs::PolymorphVectorCptr aVectorCptr )
{
  return to_python( *aVectorCptr );
}

PyObject* to_python( libecs::PolymorphVectorRCPtr aVectorRCPtr )
{
  return to_python( *aVectorRCPtr );
}



//
// DataPointVector
//

// currently to_python only

PyObject* to_python( libecs::DataPointVectorCref aVector )
{
  // here starts an ugly C hack :-/

  int aDimensions[2] = { aVector.getSize(),
			 aVector.getElementSize() / sizeof( double ) };

  PyArrayObject* anArrayObject( reinterpret_cast<PyArrayObject*>
				( PyArray_FromDims( 2, aDimensions, 
						    PyArray_DOUBLE ) ) );

  ::memcpy( anArrayObject->data, 
	    aVector.getRawArray(),   
	    aVector.getSize() * aVector.getElementSize() );

  return PyArray_Return( anArrayObject );
}

PyObject* to_python( libecs::DataPointVectorRCPtr aVectorRCPtr )
{
  return to_python( *aVectorRCPtr );
}


//
// libemc::PendingEventChecker and libemc::EventHandler
//

// NOTE: these functions return pointers to newly allocated objects

libemc::PendingEventCheckerPtr 
from_python( PyObject* aPyObjectPtr, 
	     type<libemc::PendingEventCheckerPtr> )
{
  return new PythonPendingEventChecker( aPyObjectPtr );
}

libemc::EventHandlerPtr
from_python( PyObject* aPyObjectPtr, 
	     type<libemc::EventHandlerPtr> )
{
  return new PythonEventHandler( aPyObjectPtr );
}


//
// StringVector  
//

// to_python only -- is this really needed? (sha)

PyObject* to_python( libecs::StringVectorCref aVector )
{
  libecs::StringVector::size_type aSize( aVector.size() );
  
  tuple aPyTuple( aSize );
  
  for( std::size_t i( 0 ) ; i < aSize ; ++i )
    {
      aPyTuple.set_item( i, BOOST_PYTHON_CONVERSION::to_python( aVector[i] ) );
    }

  return to_python( aPyTuple.get() );
}

PyObject* to_python( libecs::StringVectorCptr aVectorCptr )
{
  return to_python( *aVectorCptr );
}

PyObject* to_python( libecs::StringVectorRCPtr aVectorRCPtr )
{
  return to_python( *aVectorRCPtr );
}


BOOST_PYTHON_END_CONVERSION_NAMESPACE


#endif // __PYECS_HPP
