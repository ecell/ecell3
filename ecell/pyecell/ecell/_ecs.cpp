//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//             This file is part of the E-Cell System
//
//             Copyright (C) 1996-2008 Keio University
//             Copyright (C) 2005-2008 The Molecular Sciences Institute
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

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/size_type.hpp>
#include <boost/range/const_iterator.hpp>
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

    static PyObject* convert( PolymorphCref aPolymorph )
    {
        switch( aPolymorph.getType() )
        {
        case PolymorphValue::REAL :
            return PyFloat_FromDouble( aPolymorph.as<Real>() );
        case PolymorphValue::INTEGER :
            return PyInt_FromLong( aPolymorph.as<Integer>() );
        case PolymorphValue::TUPLE :
            return rangeToPyTuple( aPolymorph.as<PolymorphVector>() );
        case PolymorphValue::STRING :
            return PyString_FromStringAndSize(
                static_cast< const char * >(
                    aPolymorph.as< PolymorphValue::RawString const& >() ),
                aPolymorph.as< PolymorphValue::RawString const& >().size() );
        case PolymorphValue::NONE :
            return 0;
        }
        NEVER_GET_HERE;
    }

    template< typename Trange_ >
    static PyObject* 
    rangeToPyTuple( Trange_ const& aRange )
    {
        typename boost::range_size< Trange_ >::type
                aSize( boost::size( aRange ) );
        
        PyObject* aPyTuple( PyTuple_New( aSize ) );
       
        typename boost::range_const_iterator< Trange_ >::type j( boost::begin( aRange ) );
        for( std::size_t i( 0 ) ; i < aSize ; ++i, ++j )
        {
            PyTuple_SetItem( aPyTuple, i,
                Polymorph_to_python::convert( *j ) );
        }
        
        return aPyTuple;
    }


};


class DataPointVectorSharedPtr_to_python
{
public:

    static PyObject* 
    convert( const DataPointVectorSharedPtr& aVectorSharedPtr )
    {
        // here starts an ugly C hack :-/

        DataPointVectorCref aVector( *aVectorSharedPtr );

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
                                                python::type_id<Polymorph>() );
    }

    static void* convertible( PyObject* aPyObject )
    {
        // always passes the test for efficiency.    overload won't work.
        return aPyObject;
    }

    static void construct( PyObject* aPyObjectPtr, 
                                                 python::converter::
                                                 rvalue_from_python_stage1_data* data )
    {
        void* storage( ( ( python::converter::
                                             rvalue_from_python_storage<Polymorph>*) 
                                         data )->storage.bytes );

        new (storage) Polymorph( Polymorph_from_python( aPyObjectPtr ) );
        data->convertible = storage;
    }


    static const Polymorph 
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
                return Polymorph( PyString_AsString( aPyObjectPtr ) );
            }
                // conversion is failed. ( convert with repr() ? )
                PyErr_SetString( PyExc_TypeError, 
                                                 "Unacceptable type of an object in the tuple." );
                python::throw_error_already_set();
        // never get here: the following is for suppressing warnings
        return Polymorph();
    }


    static const PolymorphVector 
    to_PolymorphVector( PyObject* aPyObjectPtr )
    {
        std::size_t aSize( PyTuple_GET_SIZE( aPyObjectPtr ) );
            
        PolymorphVector aVector;
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
static PyObject* convert(const PolymorphMap& aPolymorphMapCref )
{
        //Polymorph_to_python aPolymorphConverter;
        PyObject * aPyDict(PyDict_New());
        PolymorphMap aPolymorphMap( aPolymorphMapCref );
        for (PolymorphMap::iterator i=aPolymorphMap.begin();
                                        i!=aPolymorphMap.end();++i)
        {
        PyDict_SetItem( aPyDict, PyString_FromString( i->first.c_str() ),
                                        Polymorph_to_python::convert( i->second ) );
                                        
        }
        return aPyDict;
}

};


// exception translators

//void translateException( ExceptionCref anException )
//{
//    PyErr_SetString( PyExc_RuntimeError, anException.what() );
//}

void translateException( const std::exception& anException )
{
    PyErr_SetString( PyExc_RuntimeError, anException.what() );
}


static PyObject* getLibECSVersionInfo()
{
    PyObject* aPyTuple( PyTuple_New( 3 ) );
        
    PyTuple_SetItem( aPyTuple, 0, PyInt_FromLong( getMajorVersion() ) );
    PyTuple_SetItem( aPyTuple, 1, PyInt_FromLong( getMinorVersion() ) );
    PyTuple_SetItem( aPyTuple, 2, PyInt_FromLong( getMicroVersion() ) );
    
    return aPyTuple;
}

// module initializer / finalizer
static struct _
{
    inline _()
    {
        if (!initialize())
            {
                throw std::runtime_error( "Failed to initialize libecs" );
            }
    }

    inline ~_()
    {
        finalize();
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
    public EventChecker
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
        //        PyErr_CheckSignals();

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
    public EventHandler
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
                                                python::type_id<EventCheckerSharedPtr>() );
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
                                             rvalue_from_python_storage<Polymorph>*) 
                                         data )->storage.bytes );

        new (storage) 
            EventCheckerSharedPtr( new PythonEventChecker( aPyObjectPtr ) );

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
                                                python::type_id<EventHandlerSharedPtr>() );
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
                                             rvalue_from_python_storage<Polymorph>*) 
                                         data )->storage.bytes );

        new (storage) 
            EventHandlerSharedPtr( new PythonEventHandler( aPyObjectPtr ) );

        data->convertible = storage;
    }
};

BOOST_PYTHON_MODULE( _ecs )
{
    using namespace boost::python;

    if (!initialize())
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

    register_exception_translator<Exception>         ( &translateException );
    register_exception_translator<std::exception>( &translateException );

    register_EventCheckerSharedPtr_from_python();
    register_EventHandlerSharedPtr_from_python();

    def( "getLibECSVersionInfo", &getLibECSVersionInfo );
    def( "getLibECSVersion",         &getVersion );

    class_<VariableReference>( "VariableReference", no_init )

        // properties
        .add_property( "Coefficient", &VariableReference::getCoefficient )
        .add_property( "MolarConc",   &VariableReference::getMolarConc )
        .add_property( "Name",        &VariableReference::getName )
        .add_property( "NumberConc",  &VariableReference::getNumberConc )
        .add_property( "IsFixed",     &VariableReference::isFixed )
        .add_property( "IsAccessor",  &VariableReference::isAccessor )
        .add_property( "Value",       &VariableReference::getValue, 
                                      &VariableReference::setValue )
        .add_property( "Velocity", &VariableReference::getVelocity )

        // methods
        .def( "addValue",        &VariableReference::addValue )
        .def( "getSuperSystem",    // this should be a property, but not supported
              &VariableReference::getSuperSystem,
              python::return_value_policy<python::reference_existing_object>() )
        ;

    class_<Process, bases<>, Process, boost::noncopyable>
        ( "Process", no_init )

        // properties
        .add_property( "Activity",  &Process::getActivity,
                                    &Process::setActivity )
        .add_property( "Priority",  &Process::getPriority )
        .add_property( "StepperID", &Process::getStepperID )

        // methods
        .def( "addValue",        &Process::addValue )
        .def( "getPositiveVariableReferenceOffset",         
              &Process::getPositiveVariableReferenceOffset )
        .def( "getSuperSystem",     // this can be a property, but not supported
              &Process::getSuperSystem,
              python::return_value_policy<python::reference_existing_object>() )
        .def( "getVariableReference",             // this should be a property
              &Process::getVariableReference,
              python::return_internal_reference<>() )
        .def( "getVariableReferenceVector",             // this should be a property
              &Process::getVariableReferenceVector,
              python::return_value_policy<python::reference_existing_object>() )
        .def( "getZeroVariableReferenceOffset",         
              &Process::getZeroVariableReferenceOffset )
        .def( "setFlux",         &Process::setFlux )
        ;


    class_<System, bases<>, System, boost::noncopyable>( "System", no_init )

        // properties
        .add_property( "Size",        &System::getSize )
        .add_property( "SizeN_A",     &System::getSizeN_A )
        .add_property( "StepperID",   &System::getStepperID )
        // methods
        .def( "getSuperSystem",     // this should be a property, but not supported
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
              &Simulator::getClassInfo )
        // Stepper-related methods
        .def( "createStepper",
              &Simulator::createStepper )
        .def( "deleteStepper",
              &Simulator::deleteStepper )
        .def( "getStepperList",
              &Simulator::getStepperList )
        .def( "getStepperPropertyList",
              &Simulator::getStepperPropertyList )
        .def( "getStepperPropertyAttributes", 
              &Simulator::getStepperPropertyAttributes )
        .def( "setStepperProperty",
              &Simulator::setStepperProperty )
        .def( "getStepperProperty",
              &Simulator::getStepperProperty )
        .def( "loadStepperProperty",
              &Simulator::loadStepperProperty )
        .def( "saveStepperProperty",
              &Simulator::saveStepperProperty )
        .def( "getStepperClassName",
              &Simulator::getStepperClassName )

        // Entity-related methods
        .def( "createEntity",
              &Simulator::createEntity )
        .def( "deleteEntity",
              &Simulator::deleteEntity )
        .def( "getEntityList",
              &Simulator::getEntityList )
        .def( "entityExists",
              &Simulator::entityExists )
        .def( "getEntityPropertyList",
              &Simulator::getEntityPropertyList )
        .def( "setEntityProperty",
              &Simulator::setEntityProperty )
        .def( "getEntityProperty",
              &Simulator::getEntityProperty )
        .def( "loadEntityProperty",
              &Simulator::loadEntityProperty )
        .def( "saveEntityProperty",
              &Simulator::saveEntityProperty )
        .def( "getEntityPropertyAttributes", 
              &Simulator::getEntityPropertyAttributes )
        .def( "getEntityClassName",
              &Simulator::getEntityClassName )

        // Logger-related methods
        .def( "getLoggerList",
                    &Simulator::getLoggerList )    
        .def( "createLogger",
              ( void ( Simulator::* )( StringCref ) )
                    &Simulator::createLogger )    
        .def( "createLogger",                                 
              ( void ( Simulator::* )( StringCref,
                                                                                     Polymorph ) )
                    &Simulator::createLogger )    
        .def( "getLoggerData", 
              ( const DataPointVectorSharedPtr( Simulator::* )(
                    StringCref ) const )
              &Simulator::getLoggerData )
        .def( "getLoggerData", 
              ( const DataPointVectorSharedPtr( Simulator::* )(
                    StringCref, RealCref,
                    RealCref ) const )
              &Simulator::getLoggerData )
        .def( "getLoggerData",
              ( const DataPointVectorSharedPtr( Simulator::* )(
                     StringCref, RealCref, 
                     RealCref, RealCref ) const )
              &Simulator::getLoggerData )
        .def( "getLoggerStartTime",
              &Simulator::getLoggerStartTime )    
        .def( "getLoggerEndTime",
              &Simulator::getLoggerEndTime )        
        .def( "getLoggerPolicy",
              &Simulator::getLoggerPolicy )
        .def( "setLoggerPolicy",
              &Simulator::setLoggerPolicy )
        .def( "getLoggerSize",
              &Simulator::getLoggerSize )

        // Simulation-related methods
        .def( "getCurrentTime",
              &Simulator::getCurrentTime )
        .def( "getNextEvent",
              &Simulator::getNextEvent )
        .def( "stop",
              &Simulator::stop )
        .def( "step",
              ( void ( Simulator::* )( void ) )
              &Simulator::step )
        .def( "step",
              ( void ( Simulator::* )( const Integer ) )
              &Simulator::step )
        .def( "run",
              ( void ( Simulator::* )() )
              &Simulator::run )
        .def( "run",
              ( void ( Simulator::* )( const Real ) ) 
              &Simulator::run )
        .def( "getPropertyInfo",
              &Simulator::getPropertyInfo )
        .def( "getDMInfo",
              &Simulator::getDMInfo )
        .def( "setEventChecker",
              &Simulator::setEventChecker )
        .def( "setEventHandler",
              &Simulator::setEventHandler )
        .add_property( "DMSearchPathSeparator",
                       &Simulator::getDMSearchPathSeparator )
        .def( "setDMSearchPath", &Simulator::setDMSearchPath )
        .def( "getDMSearchPath", &Simulator::getDMSearchPath )

        ;    

}
