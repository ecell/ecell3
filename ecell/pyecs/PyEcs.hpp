#if ! defined( __PYECS_HPP )
#define __PYECS_HPP

// for ::memcpy()
#include <string.h>

#include <boost/cast.hpp>

#include <Numeric/arrayobject.h>

#include "libecs/libecs.hpp"
#include "libecs/UVariable.hpp"
#include "libecs/DataPointVector.hpp"


namespace python = boost::python;

class PythonCallable
{
public:

  PythonCallable( PyObject* aPyObjectPtr )
    :
    thePyObjectPtr( aPyObjectPtr )
  {
    Py_INCREF( thePyObjectPtr );
    //Py_CallableCheck( thePyObjectPtr );
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

PyObject* to_python( libecs::UVariableVectorCref aVector )
{
  libecs::UVariableVector::size_type aSize( aVector.size() );
  
  python::tuple aPyTuple( aSize );
  
  for( size_t i( 0 ) ; i < aSize ; ++i )
    {
      switch( aVector[i].getType() )
	{
	case libecs::UVariable::REAL :
	  aPyTuple.set_item( i, BOOST_PYTHON_CONVERSION::
			     to_python( aVector[i].asReal() ) );
	  break;
	case libecs::UVariable::INT :
	  // FIXME: ugly cast... determine the type by autoconf?

	  aPyTuple.set_item( i,BOOST_PYTHON_CONVERSION::
			  to_python( boost::numeric_cast<long>
				     ( aVector[i].asInt() ) ) );
	  break;
	case libecs::UVariable::STRING :
	case libecs::UVariable::NONE :
	  aPyTuple.set_item( i,BOOST_PYTHON_CONVERSION::
			  to_python( aVector[i].asString() ) );
	  break;
	}
    }

  return to_python( aPyTuple.get() );
}

PyObject* to_python( libecs::UVariableVectorRCPtr aVectorRCPtr )
{
  return to_python( *aVectorRCPtr );
}

PyObject* to_python( libecs::StringVectorCref aVector )
{
  libecs::StringVector::size_type aSize( aVector.size() );
  
  python::tuple aPyTuple( aSize);
  
  for( std::size_t i( 0 ) ; i < aSize ; ++i )
    {
      aPyTuple.set_item( i,BOOST_PYTHON_CONVERSION::to_python( aVector[i] ) );
    }

  return to_python( aPyTuple.get() );
}

PyObject* to_python( libecs::StringVectorRCPtr aVectorRCPtr )
{
  return to_python( *aVectorRCPtr );
}


libecs::UVariableVector from_python( PyObject* aPyObjectPtr, 
				     python::type<libecs::UVariableVector> )
{
  python::ref aRef;

  if( PyList_Check( aPyObjectPtr ) )
    {
      aRef = make_ref( PyList_AsTuple( aPyObjectPtr ) );
    }
  else
    {
      aRef = make_ref( aPyObjectPtr );
    }
  
  python::tuple aPyTuple( aRef );

  std::size_t aSize( aPyTuple.size() );

  libecs::UVariableVector aVector;
  aVector.reserve( aSize );

  for ( std::size_t i( 0 ); i < aSize; ++i )
    {
      libecs::UVariable aUVariable;

      python::ref anItemRef( aPyTuple[i] );
      PyObject* aPyObjectPtr( anItemRef.get() ); 

      if( PyFloat_Check( aPyObjectPtr ) )
	{
	  libecs::Real aReal( BOOST_PYTHON_CONVERSION::
			      from_python( aPyObjectPtr,
					   python::type<libecs::Real>() ) );
	  aUVariable = aReal;
	}
      else if( PyInt_Check( aPyObjectPtr ) )
	{
	  libecs::Int anInt( BOOST_PYTHON_CONVERSION::
			     from_python( aPyObjectPtr,
					  python::type<long int>()) );
  //					  python::type<libecs::Int>()) );
	  aUVariable = anInt;
	}
      else if( PyString_Check( aPyObjectPtr ) )
	{
	  libecs::String aString( BOOST_PYTHON_CONVERSION::
				  from_python( aPyObjectPtr,
					       python::type<libecs::String>() ) );
	  aUVariable = aString;
	}
      else
	{
	  // convert with repr() ?

	  PyErr_SetString( PyExc_TypeError, 
			   "unacceptable type of object given" );
	  throw_argument_error();

	}

      aVector.push_back( aUVariable );
    }

  return aVector;
}

libecs::UVariableVector from_python( PyObject* aPyObjectPtr, 
				     python::type<libecs::UVariableVectorCref> )
{
  return from_python( aPyObjectPtr, python::type<libecs::UVariableVector>() );
}


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


libemc::PendingEventCheckerPtr 
from_python( PyObject* aPyObjectPtr, 
	     python::type<libemc::PendingEventCheckerPtr> )
{
  return new PythonPendingEventChecker( aPyObjectPtr );
}

libemc::EventHandlerPtr
from_python( PyObject* aPyObjectPtr, 
	     python::type<libemc::EventHandlerPtr> )
{
  return new PythonEventHandler( aPyObjectPtr );
}


BOOST_PYTHON_END_CONVERSION_NAMESPACE


#endif // __PYECS_HPP
