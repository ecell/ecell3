//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
// written by Koichi Takahashi <shafi@e-cell.org> for
// E-Cell Project.
//


#include <signal.h>
#include <string.h>

#include <boost/cast.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <numpy/arrayobject.h>

#include "libecs/libecs.hpp"
#include "libecs/Process.hpp"
#include "libecs/Exceptions.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/DataPointVector.hpp"
#include "libecs/VariableReference.hpp"
#include "libemc/libemc.hpp"
#include "libemc/Simulator.hpp"

using namespace libecs;
using namespace libemc;
namespace python = boost::python;

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


class DataPointVectorSharedPtr_to_python
{
public:

  static PyObject* 
  convert( const libecs::DataPointVectorSharedPtr& aVectorSharedPtr )
  {
    // here starts an ugly C hack :-/

    libecs::DataPointVectorCref aVector( *aVectorSharedPtr );

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
	// conversion is failed. ( convert with repr() ? )
	PyErr_SetString( PyExc_TypeError, 
			 "Unacceptable type of an object in the tuple." );
	python::throw_error_already_set();
    // never get here: the following is for suppressing warnings
    return libecs::Polymorph();
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

class PolymorphMap_to_python
{
public:
static PyObject* convert(const libecs::PolymorphMap& aPolymorphMapCref )
{
	//Polymorph_to_python aPolymorphConverter;
	PyObject * aPyDict(PyDict_New());
	libecs::PolymorphMap aPolymorphMap( aPolymorphMapCref );
	for (libecs::PolymorphMap::iterator i=aPolymorphMap.begin();
			i!=aPolymorphMap.end();++i)
	{
	PyDict_SetItem( aPyDict, PyString_FromString( i->first.c_str() ),
			Polymorph_to_python::convert( i->second ) );
			
	}
	return aPyDict;
}

};


// exception translators

//void translateException( libecs::ExceptionCref anException )
//{
//  PyErr_SetString( PyExc_RuntimeError, anException.what() );
//}

void translateException( const std::exception& anException )
{
  PyErr_SetString( PyExc_RuntimeError, anException.what() );
}


static PyObject* getLibECSVersionInfo()
{
  PyObject* aPyTuple( PyTuple_New( 3 ) );
    
  PyTuple_SetItem( aPyTuple, 0, PyInt_FromLong( libecs::getMajorVersion() ) );
  PyTuple_SetItem( aPyTuple, 1, PyInt_FromLong( libecs::getMinorVersion() ) );
  PyTuple_SetItem( aPyTuple, 2, PyInt_FromLong( libecs::getMicroVersion() ) );
  
  return aPyTuple;
}

// module initializer / finalizer
static struct _
{
  inline _()
  {
    if (!libecs::initialize())
      {
	throw std::runtime_error( "Failed to initialize libecs" );
      }
  }

  inline ~_()
  {
    libecs::finalize();
  }
} _;

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

BOOST_PYTHON_MODULE( _ecs )
{
  using namespace boost::python;

  if (!libecs::initialize())
    {
      throw std::runtime_error( "Failed to initialize libecs" );
    }

  // without this it crashes when Logger::getData() is called. why?
  import_array();

  // functions

  to_python_converter< Polymorph, Polymorph_to_python >();
  to_python_converter< DataPointVectorSharedPtr, 
    DataPointVectorSharedPtr_to_python >();
  to_python_converter< PolymorphMap, PolymorphMap_to_python>();
  
  register_Polymorph_from_python();

  register_exception_translator<Exception>     ( &translateException );
  register_exception_translator<std::exception>( &translateException );

  register_EventCheckerSharedPtr_from_python();
  register_EventHandlerSharedPtr_from_python();

  def( "getLibECSVersionInfo", &getLibECSVersionInfo );
  def( "getLibECSVersion",     &libecs::getVersion );

  def( "setDMSearchPath", &libecs::setDMSearchPath );
  def( "getDMSearchPath", &libecs::getDMSearchPath );

  class_<VariableReference>( "VariableReference", no_init )

    // properties
    .add_property( "Coefficient",
		   &VariableReference::getCoefficient )   // read-only
    .add_property( "MolarConc", 
		   &VariableReference::getMolarConc ) // read-only
    .add_property( "Name", 
		   &VariableReference::getName ) // read-only
    .add_property( "NumberConc", 
		   &VariableReference::getNumberConc ) // read-only
    .add_property( "IsFixed",
    		   &VariableReference::isFixed )      // read-only
    .add_property( "IsAccessor",
    		   &VariableReference::isAccessor )       // read-only
    .add_property( "Value", 
		   &VariableReference::getValue, 
		   &VariableReference::setValue )
    .add_property( "Velocity",
		   &VariableReference::getVelocity )

    // methods
    .def( "addValue",    &VariableReference::addValue )
    .def( "getSuperSystem",  // this should be a property, but not supported
	  &VariableReference::getSuperSystem,
	  python::return_value_policy<python::reference_existing_object>() )
    ;

  class_<Process, bases<>, Process, boost::noncopyable>
    ( "Process", no_init )

    // properties
    .add_property( "Activity",
		   &Process::getActivity,
		   &Process::setActivity )
    .add_property( "Priority",
		   &Process::getPriority )
    .add_property( "StepperID",
		   &Process::getStepperID )

    // methods
    .def( "addValue",    &Process::addValue )
    .def( "getPositiveVariableReferenceOffset",     
	  &Process::getPositiveVariableReferenceOffset )
    .def( "getSuperSystem",   // this can be a property, but not supported
	  &Process::getSuperSystem,
	  python::return_value_policy<python::reference_existing_object>() )
    .def( "getVariableReference",       // this should be a property
	  &Process::getVariableReference,
	  python::return_internal_reference<>() )
    .def( "getVariableReferenceVector",       // this should be a property
	  &Process::getVariableReferenceVector,
	  python::return_value_policy<python::reference_existing_object>() )
    .def( "getZeroVariableReferenceOffset",     
	  &Process::getZeroVariableReferenceOffset )
    .def( "setFlux",     &Process::setFlux )
    ;


  class_<System, bases<>, System, boost::noncopyable>( "System", no_init )

    // properties
    .add_property( "Size",
		   &System::getSize )
    .add_property( "SizeN_A",
		   &System::getSizeN_A )
    .add_property( "StepperID",
		   &System::getStepperID )
    // methods
    .def( "getSuperSystem",   // this should be a property, but not supported
	  &System::getSuperSystem,
	  python::return_value_policy<python::reference_existing_object>() )
    ;



  class_<VariableReferenceVector>( "VariableReferenceVector" )
    //, bases<>, VariableReferenceVector>

    .def( vector_indexing_suite<VariableReferenceVector>() )
    ;


  // Simulator class
  class_<Simulator>( "Simulator" )
    .def( init<>() )
    .def( "getClassInfo",
	  ( const libecs::PolymorphMap
	    ( libemc::Simulator::* )( libecs::StringCref, libecs::StringCref ) )
	  &libemc::Simulator::getClassInfo )
    .def( "getClassInfo",
	  ( const libecs::PolymorphMap
	    ( libemc::Simulator::* )( libecs::StringCref, libecs::StringCref,
                                  const libecs::Integer ) )
	  &libemc::Simulator::getClassInfo )
    // Stepper-related methods
    .def( "createStepper",
	  &libemc::Simulator::createStepper )
    .def( "deleteStepper",
	  &libemc::Simulator::deleteStepper )
    .def( "getStepperList",
	  &libemc::Simulator::getStepperList )
    .def( "getStepperPropertyList",
	  &libemc::Simulator::getStepperPropertyList )
    .def( "getStepperPropertyAttributes", 
	  &libemc::Simulator::getStepperPropertyAttributes )
    .def( "setStepperProperty",
	  &libemc::Simulator::setStepperProperty )
    .def( "getStepperProperty",
	  &libemc::Simulator::getStepperProperty )
    .def( "loadStepperProperty",
	  &libemc::Simulator::loadStepperProperty )
    .def( "saveStepperProperty",
	  &libemc::Simulator::saveStepperProperty )
    .def( "getStepperClassName",
	  &libemc::Simulator::getStepperClassName )

    // Entity-related methods
    .def( "createEntity",
	  &libemc::Simulator::createEntity )
    .def( "deleteEntity",
	  &libemc::Simulator::deleteEntity )
    .def( "getEntityList",
	  &libemc::Simulator::getEntityList )
    .def( "isEntityExist",
	  &libemc::Simulator::isEntityExist )
    .def( "getEntityPropertyList",
	  &libemc::Simulator::getEntityPropertyList )
    .def( "setEntityProperty",
	  &libemc::Simulator::setEntityProperty )
    .def( "getEntityProperty",
	  &libemc::Simulator::getEntityProperty )
    .def( "loadEntityProperty",
	  &libemc::Simulator::loadEntityProperty )
    .def( "saveEntityProperty",
	  &libemc::Simulator::saveEntityProperty )
    .def( "getEntityPropertyAttributes", 
	  &libemc::Simulator::getEntityPropertyAttributes )
    .def( "getEntityClassName",
	  &libemc::Simulator::getEntityClassName )

    // Logger-related methods
    .def( "getLoggerList",
	  &libemc::Simulator::getLoggerList )  
    .def( "createLogger",
	  ( void ( libemc::Simulator::* )( libecs::StringCref ) )
	  &libemc::Simulator::createLogger )  
    .def( "createLogger",		 
	  ( void ( libemc::Simulator::* )( libecs::StringCref,
					   libecs::Polymorph ) )
	  &libemc::Simulator::createLogger )  
    .def( "getLoggerData", 
	  ( const libecs::DataPointVectorSharedPtr
	    ( libemc::Simulator::* )( libecs::StringCref ) const )
	  &libemc::Simulator::getLoggerData )
    .def( "getLoggerData", 
	  ( const libecs::DataPointVectorSharedPtr 
	    ( libemc::Simulator::* )( libecs::StringCref,
				      libecs::RealCref,
				      libecs::RealCref ) const )
	  &libemc::Simulator::getLoggerData )
    .def( "getLoggerData",
	  ( const libecs::DataPointVectorSharedPtr
	    ( libemc::Simulator::* )( libecs::StringCref,
				      libecs::RealCref, 
				      libecs::RealCref,
				      libecs::RealCref ) const )
	  &libemc::Simulator::getLoggerData )
    .def( "getLoggerStartTime",
	  &libemc::Simulator::getLoggerStartTime )  
    .def( "getLoggerEndTime",
	  &libemc::Simulator::getLoggerEndTime )    
    .def( "getLoggerMinimumInterval",
          &libemc::Simulator::getLoggerMinimumInterval )
    .def( "setLoggerMinimumInterval",
          &libemc::Simulator::setLoggerMinimumInterval )
    .def( "getLoggerPolicy",
	  &libemc::Simulator::getLoggerPolicy )
    .def( "setLoggerPolicy",
	  &libemc::Simulator::setLoggerPolicy )
    .def( "getLoggerSize",
	  &libemc::Simulator::getLoggerSize )

    // Simulation-related methods
    .def( "getCurrentTime",
	  &libemc::Simulator::getCurrentTime )
    .def( "getNextEvent",
	  &libemc::Simulator::getNextEvent )
    .def( "stop",
	  &libemc::Simulator::stop )
    .def( "step",
	  ( void ( libemc::Simulator::* )( void ) )
	  &libemc::Simulator::step )
    .def( "step",
	  ( void ( libemc::Simulator::* )( const libecs::Integer ) )
	  &libemc::Simulator::step )
    .def( "run",
	  ( void ( libemc::Simulator::* )() )
	  &libemc::Simulator::run )
    .def( "run",
	  ( void ( libemc::Simulator::* )( const libecs::Real ) ) 
	  &libemc::Simulator::run )
    .def( "setEventChecker",
	  &libemc::Simulator::setEventChecker )
    .def( "setEventHandler",
	  &libemc::Simulator::setEventHandler )
    ;  

}
