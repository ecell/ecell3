//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

#include <cstring>
#include <cstddef>
#include <boost/cast.hpp>
#include <boost/python.hpp>
#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_const.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/arrayobject.h>

#include "libecs/Exceptions.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/Process.hpp"
#include "libecs/VariableReference.hpp"
#include "libecs/libecs.hpp"

namespace python = boost::python;

class Polymorph_to_python
{
public:

    static PyObject* convert( const libecs::Polymorph& aPolymorph )
    {
        switch ( aPolymorph.getType() )
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
    PolymorphVector_to_PyTuple( const libecs::PolymorphVector& aVector )
    {
        libecs::PolymorphVector::size_type aSize( aVector.size() );

        PyObject* aPyTuple( PyTuple_New( aSize ) );

        for ( size_t i( 0 ) ; i < aSize ; ++i )
        {
            PyTuple_SetItem( aPyTuple, i,
                             Polymorph_to_python::convert( aVector[i] ) );
        }

        return aPyTuple;
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
                           rvalue_from_python_storage<libecs::Polymorph>* )
                         data )->storage.bytes );

        new ( storage ) libecs::Polymorph( Polymorph_from_python( aPyObjectPtr ) );
        data->convertible = storage;
    }


    static const libecs::Polymorph
    Polymorph_from_python( PyObject* aPyObjectPtr )
    {
        if ( PyFloat_Check( aPyObjectPtr ) )
        {
            return PyFloat_AS_DOUBLE( aPyObjectPtr );
        }
        else if ( PyInt_Check( aPyObjectPtr ) )
        {
            return PyInt_AS_LONG( aPyObjectPtr );
        }
        else if ( PyTuple_Check( aPyObjectPtr ) )
        {
            return to_PolymorphVector( aPyObjectPtr );
        }
        else if ( PyList_Check( aPyObjectPtr ) )
        {
            return to_PolymorphVector( PyList_AsTuple( aPyObjectPtr ) );
        }
        else if ( PyString_Check( aPyObjectPtr ) )
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
        return libecs::Polymorph(); // never get here
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

using namespace libecs;

template< typename T_, typename Tvtype_, std::size_t offset >
Tvtype_& memberRef( T_* x)
{
    return *reinterpret_cast< Tvtype_* >(
        reinterpret_cast<
            typename boost::mpl::if_<
                typename boost::is_const< Tvtype_ >,
                typename boost::add_const< unsigned char >::type*,
                unsigned char*>::type >( x ) + offset );
}


// exception translators
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
        if ( !libecs::initialize() )
        {
            throw std::runtime_error( "Failed to initialize libecs" );
        }
    }

    inline ~_()
    {
        libecs::finalize();
    }
} _;

BOOST_PYTHON_MODULE( _ecs )
{
    using namespace boost::python;

    // without this it crashes when Logger::getData() is called. why?
    import_array();

    // functions

    def( "getLibECSVersionInfo", &getLibECSVersionInfo );
    def( "getLibECSVersion",     &libecs::getVersion );

    def( "setDMSearchPath", &libecs::setDMSearchPath );
    def( "getDMSearchPath", &libecs::getDMSearchPath );
    //  def( "getDMInfoList",   &libemc::getDMInfoList );
    //  def( "getDMInfo",       &libemc::getDMInfo );


    to_python_converter< Polymorph, Polymorph_to_python >();
    register_Polymorph_from_python();

    register_exception_translator<Exception>     ( &translateException );
    register_exception_translator<std::exception>( &translateException );

    class_<VariableReference>( "VariableReference", no_init )
        // properties
        .add_property( "coefficient",
                       &VariableReference::getCoefficient )   // read-only
        .add_property( "name",
                       &VariableReference::getName ) // read-only
        .add_property( "isAccessor",
                       &VariableReference::isAccessor )       // read-only
        ;

    class_<Process, bases<>, Process, boost::noncopyable>( "Process", no_init )
        // properties
        .add_property( "activity",
                       &Process::getActivity,
                       &Process::setActivity )
        .add_property( "priority",
                       &Process::getPriority )
        .add_property( "stepperID",
                       &Process::getStepperID )
        .add_property( "enclosingSystem",
                       make_function(
                           &Process::getEnclosingSystem,
                           return_value_policy<
                               python::reference_existing_object>() ) )
        // methods
        .def( "addValue",    &Process::addValue )
        .def( "setFlux",     &Process::setFlux )
        ;


    class_<System, bases<>, System, boost::noncopyable>( "System", no_init )
        // properties
        .add_property( "size",
                       &System::getSize )
        .add_property( "sizeN_A",
                       &System::getSizeN_A )
        .add_property( "stepperID",
                       &System::getStepperID )
         // this should be a property, but not supported
        .add_property( "enclosingSystem",
                       make_function(
                           &System::getEnclosingSystem,
                           return_value_policy<reference_existing_object>() ) )
        ;

    class_<EntityType>( "EntityType", no_init )
        .add_property( "code", make_getter( &EntityType::code ) )
        .add_property( "name", make_getter( &EntityType::name ) )
        // properties
        .def_readonly( "NONE",      &EntityType::NONE )
        .def_readonly( "ENTITY",    &EntityType::ENTITY )
        .def_readonly( "VARIABLE",  &EntityType::VARIABLE )
        .def_readonly( "PROCESS",   &EntityType::PROCESS )
        .def_readonly( "SYSTEM",    &EntityType::SYSTEM )
        ;
}

