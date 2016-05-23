//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2016 Keio University
//       Copyright (C) 2008-2016 RIKEN
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
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
#include <cstdlib>
#include <utility>
#include <cctype>
#include <functional>

#include <boost/bind.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/size_type.hpp>
#include <boost/range/const_iterator.hpp>
#if BOOST_VERSION >= 103200 // for boost-1.32.0 or later.
#   include <boost/numeric/conversion/cast.hpp>
#else // use this instead for boost-1.31 or earlier.
#   include <boost/cast.hpp>
#endif
#include <boost/format.hpp>
#include <boost/format/group.hpp>
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/object.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/object/inheritance.hpp>
#include <boost/python/object/find_instance.hpp>
#include <boost/python/reference_existing_object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/optional/optional.hpp>

#include <numpy/arrayobject.h>
#include <stringobject.h>
#include <weakrefobject.h>

#include "dmtool/SharedModuleMakerInterface.hpp"

#include "libecs/Model.hpp"
#include "libecs/libecs.hpp"
#include "libecs/Process.hpp"
#include "libecs/Exceptions.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/DataPointVector.hpp"
#include "libecs/VariableReference.hpp"

#if PY_VERSION_HEX < 0x02050000

typedef inquiry lenfunc;
typedef intargfunc ssizeargfunc;
typedef intintargfunc ssizessizeargfunc;
typedef intobjargproc ssizeobjargproc;
typedef intintobjargproc ssizessizeobjargproc;
typedef int Py_ssize_t;

#endif

#if PY_VERSION_HEX < 0x02040000
#define Py_VISIT(op)                                    \
    do {                                                \
        if (op) {                                       \
            int vret = visit((PyObject *)(op), arg);    \
            if (vret)                                   \
                return vret;                            \
        }                                               \
    } while (0)
#endif

using namespace libecs;
namespace py = boost::python;

inline boost::optional< py::object > generic_getattr( py::object anObj, const char* aName, bool return_null_if_not_exists )
{
    py::handle<> aRetval( py::allow_null( PyObject_GenericGetAttr(
        anObj.ptr(),
        py::handle<>(
            PyString_InternFromString(
                const_cast< char* >( aName ) ) ).get() ) ) );
    if ( !aRetval )
    {
        if ( return_null_if_not_exists )
            PyErr_Clear();
        else
            py::throw_error_already_set();
        return boost::optional< py::object >();
    }

    return py::object( aRetval );
}

inline py::object generic_getattr( py::object anObj, const char* aName )
{
    py::handle<> aRetval( py::allow_null( PyObject_GenericGetAttr(
        anObj.ptr(),
        py::handle<>(
            PyString_InternFromString(
                const_cast< char* >( aName ) ) ).get() ) ) );
    if ( !aRetval )
    {
        py::throw_error_already_set();
    }

    return py::object( aRetval );
}

struct PolymorphToPythonConverter
{
    static void addToRegistry()
    {
        py::to_python_converter< Polymorph, PolymorphToPythonConverter >();
    }

    static PyObject* convert( Polymorph const& aPolymorph )
    {
        switch( aPolymorph.getType() )
        {
        case PolymorphValue::REAL :
            return PyFloat_FromDouble( aPolymorph.as<Real>() );
        case PolymorphValue::INTEGER :
            return PyInt_FromLong( aPolymorph.as<Integer>() );
        case PolymorphValue::TUPLE :
            return rangeToPyTuple( aPolymorph.as<PolymorphValue::Tuple const&>() );
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
            PyTuple_SetItem( aPyTuple, i, PolymorphToPythonConverter::convert( *j ) );
        }
        
        return aPyTuple;
    }
};

struct PolymorphRetriever
{
    struct PySeqSTLIterator
    {
    public:
        typedef std::random_access_iterator_tag iterator_category;
        typedef Py_ssize_t difference_type;
        typedef Polymorph value_type;
        typedef void pointer;
        typedef Polymorph reference;

        PySeqSTLIterator( PyObject* seq, difference_type idx )
            : theSeq( seq ), theIdx( idx ) {}

        PySeqSTLIterator& operator++()
        {
            ++theIdx;
            return *this;
        }

        PySeqSTLIterator operator++( int )
        {
            PySeqSTLIterator retval( *this );
            ++theIdx;
            return retval;
        }

        PySeqSTLIterator& operator--()
        {
            --theIdx;
            return *this;
        }

        PySeqSTLIterator operator--( int )
        {
            PySeqSTLIterator retval( *this );
            --theIdx;
            return retval;
        }

        PySeqSTLIterator operator+( difference_type offset ) const
        {
            return PySeqSTLIterator( theSeq, theIdx + offset );
        }

        PySeqSTLIterator operator-( difference_type offset ) const
        {
            return PySeqSTLIterator( theSeq, theIdx - offset );
        }

        difference_type operator-( PySeqSTLIterator const& rhs ) const
        {
            return theIdx - rhs.theIdx;
        }

        PySeqSTLIterator& operator+=( difference_type offset )
        {
            theIdx += offset;
            return *this;
        }

        PySeqSTLIterator& operator-=( difference_type offset )
        {
            theIdx -= offset;
            return *this;
        }

        bool operator==( PySeqSTLIterator const& rhs )
        {
            return theIdx == rhs.theIdx;
        }

        bool operator!=( PySeqSTLIterator const& rhs )
        {
            return !operator==( rhs );
        }

        bool operator<( PySeqSTLIterator const& rhs )
        {
            return theIdx < rhs.theIdx;
        }

        bool operator>=( PySeqSTLIterator const& rhs )
        {
            return theIdx >= rhs.theIdx;
        }

        bool operator>( PySeqSTLIterator const& rhs )
        {
            return theIdx > rhs.theIdx;
        }

        bool operator<=( PySeqSTLIterator const& rhs )
        {
            return theIdx <= rhs.theIdx;
        }

        value_type operator*();
        
    private:
        PyObject* theSeq;
        Py_ssize_t theIdx;
    };

    static boost::iterator_range< PySeqSTLIterator > pyseq_range( PyObject *pyo )
    {
        return boost::make_iterator_range(
            PySeqSTLIterator( pyo, 0 ),
            PySeqSTLIterator( pyo, PySequence_Length( pyo ) ) );
    }

    static void addToRegistry()
    { 
        py::converter::registry::insert( &convertible, &construct,
                                          py::type_id< Polymorph >() );
    }

    static const Polymorph convert( PyObject* aPyObjectPtr )
    {
        if( PyFloat_Check( aPyObjectPtr ) )
        {
            return Polymorph( PyFloat_AS_DOUBLE( aPyObjectPtr ) );
        }
        else if( PyInt_Check( aPyObjectPtr ) )
        {
            return Polymorph( PyInt_AS_LONG( aPyObjectPtr ) );
        }
        else if( PyString_Check( aPyObjectPtr ) )
        {
            return Polymorph( PyString_AS_STRING( aPyObjectPtr ),
                              PyString_GET_SIZE( aPyObjectPtr ) );
        }
        else if( PyUnicode_Check( aPyObjectPtr ) )
        {
            aPyObjectPtr = PyUnicode_AsEncodedString( aPyObjectPtr, NULL, NULL );
            if ( aPyObjectPtr )
            {
                char *str;
                Py_ssize_t str_len;
                if ( !PyString_AsStringAndSize( aPyObjectPtr, &str, &str_len ) )
                {
                    return Polymorph( str, str_len );
                }
                else
                {
                    PyErr_Clear();
                }
            }
        }
        else if ( PySequence_Check( aPyObjectPtr ) )
        {
            return Polymorph( PolymorphValue::create( pyseq_range( aPyObjectPtr ) ) );
        }            
        // conversion is failed. ( convert with repr() ? )
        PyErr_SetString( PyExc_TypeError, 
                         "Unacceptable type of an object in the tuple." );
        py::throw_error_already_set();
        // never get here: the following is for suppressing warnings
        return Polymorph();
    }

    static bool isConvertible( PyObject* aPyObjectPtr )
    {
        return PyFloat_Check( aPyObjectPtr )
                || PyInt_Check( aPyObjectPtr )
                || PyString_Check( aPyObjectPtr )
                || PyUnicode_Check( aPyObjectPtr )
                || PySequence_Check( aPyObjectPtr );
    }

    static void* convertible( PyObject* aPyObject )
    {
        // always passes the test for efficiency.    overload won't work.
        return aPyObject;
    }

    static void construct( PyObject* aPyObjectPtr, 
                           py::converter::rvalue_from_python_stage1_data* data )
    {
        void* storage( reinterpret_cast<
            py::converter::rvalue_from_python_storage<Polymorph>* >(
                data )->storage.bytes );
        new (storage) Polymorph( convert( aPyObjectPtr ) );
        data->convertible = storage;
    }
};

PolymorphRetriever::PySeqSTLIterator::value_type
PolymorphRetriever::PySeqSTLIterator::operator*()
{
    return PolymorphRetriever::convert(
            py::handle<>( PySequence_GetItem( theSeq, theIdx ) ).get() );
}


struct PropertySlotMapToPythonConverter
{
    typedef PropertyInterfaceBase::PropertySlotMap argument_type;

    static void addToRegistry()
    {
        py::to_python_converter< argument_type, PropertySlotMapToPythonConverter >();
    }

    static PyObject* convert( argument_type const& map )
    {
        PyObject* aPyDict( PyDict_New() );
        for ( argument_type::const_iterator i( map.begin() );
              i != map.end(); ++i )
        {
            PyDict_SetItem( aPyDict, PyString_FromStringAndSize(
                i->first.data(), i->first.size() ),
                py::incref( py::object( PropertyAttributes( *i->second ) ).ptr() ) );
                                            
        }
        return aPyDict;
    }
};

struct StringVectorToPythonConverter
{
    typedef std::vector< libecs::String > StringVector;

    static void addToRegistry()
    {
        py::to_python_converter< StringVector, StringVectorToPythonConverter >();
    }

    static PyObject* convert( StringVector const& aStringVector )
    {
        py::list retval;

        for ( StringVector::const_iterator i( aStringVector.begin() );
                i != aStringVector.end(); ++i )
        {
            retval.append( py::object( *i ) );
        }

        return py::incref( retval.ptr() );
    }
};


template< typename Ttcell_ >
inline void buildPythonTupleFromTuple(PyObject* pyt, const Ttcell_& cell,
        Py_ssize_t idx = 0)
{
    PyTuple_SetItem( pyt, idx,
        py::incref(
            py::object( cell.get_head()).ptr() ) );

    buildPythonTupleFromTuple(pyt, cell.get_tail(), idx + 1);
}

template<>
inline void buildPythonTupleFromTuple<boost::tuples::null_type>(
        PyObject*, const boost::tuples::null_type&, Py_ssize_t) {}

template< typename Ttuple_ >
struct TupleToPythonConverter
{
    typedef Ttuple_ argument_value_type;
    typedef const argument_value_type& argument_type;
    static PyObject* convert(argument_type val)
    {
        PyObject* retval(
            PyTuple_New( boost::tuples::length<Ttuple_>::value ) );
        buildPythonTupleFromTuple(retval, val);
        return retval;
    }
};

template< typename Tfirst_, typename Tsecond_ >
struct TupleToPythonConverter< std::pair<Tfirst_, Tsecond_> >
{
    typedef std::pair<Tfirst_, Tsecond_> argument_value_type;
    typedef const argument_value_type& argument_type;

    static PyObject* convert( argument_type val )
    {
        return py::incref(
                py::make_tuple(
                    val.first, val.second).ptr());
    }
};

template< typename Ttuple_ >
struct TupleToPythonConverter<boost::shared_ptr<Ttuple_> >
{
    typedef Ttuple_ argument_value_type;
    typedef boost::shared_ptr< argument_value_type > argument_type;
    static PyObject* convert( argument_type val )
    {
        return TupleToPythonConverter< argument_value_type >::convert( *val );
    }
};

template< typename Ttuple_ >
void registerTupleConverters()
{
    py::to_python_converter<
        Ttuple_, TupleToPythonConverter<Ttuple_> >();
    py::to_python_converter<
        boost::shared_ptr<Ttuple_>,
        TupleToPythonConverter<boost::shared_ptr<Ttuple_> > >();
}

struct FullIDToPythonConverter
{
    static void addToRegistry()
    {
        py::to_python_converter< FullID, FullIDToPythonConverter >();
    }

    static PyObject* convert( FullID const& aFullID )
    {
        return py::incref(
            ( aFullID.isValid() ? py::object( aFullID.asString() ):
                                py::object() ).ptr() );
    }
};

struct PythonToFullIDConverter
{
    static void* convertible(PyObject* pyo)
    {
        if ( !PyString_Check( pyo ) )
        {
            return 0;
        }

        return pyo;
    }

    static void construct(PyObject* pyo, py::converter::rvalue_from_python_stage1_data* data)
    {
        data->convertible = new( reinterpret_cast<
            py::converter::rvalue_from_python_storage<FullID>* >(
                data )->storage.bytes ) FullID(
                boost::python::extract< std::string >( pyo ) );
    }

    static void addToRegistry()
    {
        py::converter::registry::insert( &convertible, &construct,
                                         py::type_id< FullID >() );
    }
};



// exception translators

static void translateException( const std::exception& anException )
{
    PyErr_SetString( PyExc_RuntimeError, anException.what() );
}

static void translateRangeError( const std::range_error& anException )
{
    PyErr_SetString( PyExc_KeyError, anException.what() );
}

static PyObject* getLibECSVersionInfo()
{
    PyObject* aPyTuple( PyTuple_New( 3 ) );
        
    PyTuple_SetItem( aPyTuple, 0, PyInt_FromLong( getMajorVersion() ) );
    PyTuple_SetItem( aPyTuple, 1, PyInt_FromLong( getMinorVersion() ) );
    PyTuple_SetItem( aPyTuple, 2, PyInt_FromLong( getMicroVersion() ) );
    
    return aPyTuple;
}

static class PyEcsModule
{
public:
    PyEcsModule()
    {
        if (!initialize())
        {
            throw std::runtime_error( "Failed to initialize libecs" );
        }
    }

    ~PyEcsModule()
    {
        finalize();
    }
} theModule;

template< typename Tdp_ >
class DataPointVectorWrapper
{
public:
    typedef Tdp_ element_type;

private:
    struct GetItemFunc
    {
    };

public:
    class Iterator
    {
    protected:
        PyObject_VAR_HEAD
        DataPointVectorWrapper* theDPVW;
        std::size_t theIdx;
        
    public:
        static PyTypeObject __class__;

    public:
        void* operator new( size_t )
        {
            return PyObject_New( Iterator, &__class__ );
        }

        Iterator( DataPointVectorWrapper* dpvw, std::size_t idx )
            : theDPVW( dpvw ), theIdx( idx )
        {
            Py_INCREF( dpvw );
        }

        ~Iterator()
        {
            Py_XDECREF( theDPVW );
        }

    public:
        static PyTypeObject* __class_init__()
        {
            PyType_Ready( &__class__ );
            return &__class__;
        }

        static Iterator* create( DataPointVectorWrapper* dpvw,
                                 std::size_t idx = 0 )
        {
            return new Iterator( dpvw, idx );
        }

        static void __dealloc__( Iterator* self )
        {
            delete self;
        }

        static PyObject* __next__( Iterator* self )
        {
            DataPointVector const& vec( *self->theDPVW->theVector );
            if ( self->theIdx < vec.getSize() )
            {
                return toPyObject( &getItem( vec, self->theIdx++ ) );
            }
            return NULL;
        }
    };

protected:
    PyObject_VAR_HEAD
    boost::shared_ptr< DataPointVector > theVector;

public:
    static PyTypeObject __class__;
    static PySequenceMethods __seq__;
    static PyGetSetDef __getset__[];
    static const std::size_t theNumOfElemsPerEntry = sizeof( Tdp_ ) / sizeof( double );

private:
    void* operator new( size_t )
    {
        return PyObject_New( DataPointVectorWrapper, &__class__);
    }

    void operator delete( void* ptr )
    {
        reinterpret_cast< PyObject* >( ptr )->ob_type->tp_free( reinterpret_cast< PyObject* >( ptr ) );
    }

    DataPointVectorWrapper( boost::shared_ptr< DataPointVector > const& aVector )
        : theVector( aVector ) {}

    ~DataPointVectorWrapper()
    {
    }

    PyObject* asPyArray()
    {
        PyArray_Descr* descr( PyArray_DescrFromObject(
            reinterpret_cast< PyObject* >( this ), 0 ) );
        BOOST_ASSERT( descr != NULL );

        return PyArray_CheckFromAny(
                reinterpret_cast< PyObject* >( this ),
                descr, 0, 0, 0, NULL );
    }

public:
    static PyTypeObject* __class_init__()
    {
        Iterator::__class_init__();
        PyType_Ready( &__class__ );
        return &__class__;
    }

    static DataPointVectorWrapper* create( boost::shared_ptr< DataPointVector > const& aVector )
    {
        return new DataPointVectorWrapper( aVector ); 
    }

    static PyObject* __get__shape( DataPointVectorWrapper* self )
    {
        PyObject* retval( PyTuple_New( 2 ) );
        PyTuple_SET_ITEM( retval, 0, PyLong_FromUnsignedLong( self->theVector->getSize() ) );
        PyTuple_SET_ITEM( retval, 1, PyLong_FromUnsignedLong( theNumOfElemsPerEntry ) );
        return retval;
    }

    static void __dealloc__( DataPointVectorWrapper* self )
    {
        delete self;
    }

    static PyObject* __repr__( DataPointVectorWrapper* self )
    {
        return PyObject_Repr( self->asPyArray() );
    }

    static PyObject* __str__( DataPointVectorWrapper* self )
    {
        return PyObject_Str( self->asPyArray() );
    }

    static long __hash__( DataPointVectorWrapper* self )
    {
        PyErr_SetString(PyExc_TypeError, "DataPointVectors are unhashable");
        return -1;
    }

    static int __traverse__( DataPointVectorWrapper* self, visitproc visit,
            void *arg)
    {
        DataPointVector const& vec( *self->theVector );
        for ( std::size_t i( 0 ), len( vec.getSize() ); i < len; ++i )
        {
            Py_VISIT( toPyObject( &getItem( *self->theVector, i ) ) );
        }
        return 0;
    }

    static PyObject* __iter__( DataPointVectorWrapper* self )
    {
        Iterator* i( Iterator::create( self ) );
        return reinterpret_cast< PyObject* >( i );
    }

    static Py_ssize_t __len__( DataPointVectorWrapper* self )
    {
        return self->theVector->getSize();
    }

    static PyObject* __getitem__( DataPointVectorWrapper* self, Py_ssize_t idx )
    {
        if ( idx < 0 || idx >= static_cast< Py_ssize_t >( self->theVector->getSize() ) )
        {
            PyErr_SetObject(PyExc_IndexError,
                    PyString_FromString("index out of range"));
		    return NULL;
        }
            
        return toPyObject( &getItem( *self->theVector, idx ) );
    }

    static void __dealloc_array_struct( void* ptr,
                                        DataPointVectorWrapper* self )
    {
        Py_XDECREF( self );
        PyMem_FREE( ptr );
    }

    static PyObject* __get___array__struct( DataPointVectorWrapper* self,
                                            void* closure )
    {
        PyArrayInterface* aif(
            reinterpret_cast< PyArrayInterface* >(
                PyMem_MALLOC( sizeof( PyArrayInterface )
                              + sizeof ( Py_intptr_t ) * 4 ) ) );
        if ( !aif )
        {
            return NULL;
        }
        aif->two = 2;
        aif->nd = 2;
        aif->typekind = 'f';
        aif->itemsize = sizeof( double );
        aif->flags = NPY_CONTIGUOUS | NPY_ALIGNED | NPY_NOTSWAPPED;
        aif->shape = reinterpret_cast< Py_intptr_t* >( aif + 1 );
        aif->shape[ 0 ] = self->theVector->getSize();
        aif->shape[ 1 ] = theNumOfElemsPerEntry;
        aif->strides = reinterpret_cast< Py_intptr_t* >( aif + 1 ) + 2;
        aif->strides[ 0 ] = sizeof( double ) * aif->shape[ 1 ];
        aif->strides[ 1 ] = sizeof( double );
        aif->data = const_cast< void* >( self->theVector->getRawArray() );
        aif->descr = NULL;

        Py_INCREF( self );
        return PyCObject_FromVoidPtrAndDesc( aif, self,
                reinterpret_cast< void(*)(void*, void*) >(
                    __dealloc_array_struct ) );
    }

    static int __contains__( DataPointVectorWrapper* self, PyObject *e )
    {
        if ( !PyArray_Check( e ) || PyArray_NDIM( e ) != 1
                || PyArray_DIMS( e )[ 0 ] < static_cast< Py_ssize_t >( theNumOfElemsPerEntry ) )
        {
            return 1;
        }


        DataPoint const& dp( *reinterpret_cast< DataPoint* >( PyArray_DATA( e ) ) );
        DataPoint const* begin( reinterpret_cast< DataPoint const* >(
                          self->theVector->getRawArray() ) );
        DataPoint const* end( begin + self->theVector->getSize() );
        return end == std::find( begin, end, dp );
    }

    static PyObject* toPyObject( DataPoint const* dp )
    {
        static const npy_intp dims[] = { theNumOfElemsPerEntry };
        PyArrayObject* arr( reinterpret_cast< PyArrayObject* >(
            PyArray_NewFromDescr( &PyArray_Type,
                PyArray_DescrFromType( NPY_DOUBLE ),
                1, const_cast< npy_intp* >( dims ), NULL, 0, NPY_CONTIGUOUS, NULL )
            ) );
        std::memcpy( PyArray_DATA( arr ), const_cast< DataPoint* >( dp ), sizeof( double ) * theNumOfElemsPerEntry );
        return reinterpret_cast< PyObject * >( arr );
    }

    static Tdp_ const& getItem( DataPointVector const& vec, std::size_t idx )
    {
        return GetItemFunc()( vec, idx );
    }
};

template<>
struct DataPointVectorWrapper< DataPoint >::GetItemFunc
{
    DataPoint const& operator()( DataPointVector const& vec, std::size_t idx ) const
    {
        return vec.asShort( idx );
    }
};

template<>
struct DataPointVectorWrapper< LongDataPoint >::GetItemFunc
{
    LongDataPoint const& operator()( DataPointVector const& vec, std::size_t idx ) const
    {
        return vec.asLong( idx );
    }
};


template< typename Tdp_ >
PyTypeObject DataPointVectorWrapper< Tdp_ >::Iterator::__class__ = {
	PyObject_HEAD_INIT( &PyType_Type )
	0,					/* ob_size */
	"ecell._ecs.DataPointVectorWrapper.Iterator", /* tp_name */
	sizeof( typename DataPointVectorWrapper::Iterator ), /* tp_basicsize */
	0,					/* tp_itemsize */
	/* methods */
	(destructor)&DataPointVectorWrapper::Iterator::__dealloc__, /* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_HAVE_CLASS | Py_TPFLAGS_HAVE_WEAKREFS | Py_TPFLAGS_HAVE_ITER,/* tp_flags */
	0,					/* tp_doc */
	0,	/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	PyObject_SelfIter,  /* tp_iter */
	(iternextfunc)&DataPointVectorWrapper::Iterator::__next__,		/* tp_iternext */
	0,		        	/* tp_methods */
	0,					/* tp_members */
	0,                  /* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	0,			        /* tp_init */
	PyType_GenericAlloc,			/* tp_alloc */
	PyType_GenericNew,			/* tp_new */
	PyObject_Del,			/* tp_free */
};

template< typename Tdp_ >
PyTypeObject DataPointVectorWrapper< Tdp_ >::__class__ = {
	PyObject_HEAD_INIT( &PyType_Type )
	0,
	"ecell._ecs.DataPointVector",
	sizeof(DataPointVectorWrapper),
	0,
	(destructor)&DataPointVectorWrapper::__dealloc__, /* tp_dealloc */
	0,      			/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	(reprfunc)&DataPointVectorWrapper::__repr__,			/* tp_repr */
	0,					/* tp_as_number */
	&DataPointVectorWrapper::__seq__,			/* tp_as_sequence */
	0,			/* tp_as_mapping */
	(hashfunc)&DataPointVectorWrapper::__hash__,				/* tp_hash */
	0,					/* tp_call */
	(reprfunc)&DataPointVectorWrapper::__str__,				/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_HAVE_CLASS | Py_TPFLAGS_HAVE_WEAKREFS | Py_TPFLAGS_HAVE_SEQUENCE_IN,		/* tp_flags */
 	0,				/* tp_doc */
 	(traverseproc)&DataPointVectorWrapper::__traverse__,		/* tp_traverse */
 	0,			/* tp_clear */
	0,			/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	(getiterfunc)&DataPointVectorWrapper::__iter__,				/* tp_iter */
	0,					/* tp_iternext */
	0,				/* tp_methods */
	0,					/* tp_members */
	DataPointVectorWrapper::__getset__,					/* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	0,			/* tp_init */
	PyType_GenericAlloc,			/* tp_alloc */
	PyType_GenericNew,			/* tp_new */
	PyObject_Del,			/* tp_free */
};

template< typename Tdp_ >
PySequenceMethods DataPointVectorWrapper< Tdp_ >::__seq__ = {
	(lenfunc)&DataPointVectorWrapper::__len__,			/* sq_length */
	(binaryfunc)0,		/* sq_concat */
	(ssizeargfunc)0,		/* sq_repeat */
	(ssizeargfunc)&DataPointVectorWrapper::__getitem__,		/* sq_item */
	(ssizessizeargfunc)0,		/* sq_slice */
	(ssizeobjargproc)0,		/* sq_ass_item */
	(ssizessizeobjargproc)0,	/* sq_ass_slice */
	(objobjproc)&DataPointVectorWrapper::__contains__,		/* sq_contains */
	(binaryfunc)0,	/* sq_inplace_concat */
	(ssizeargfunc)0	/* sq_inplace_repeat */
};

template< typename Tdp_ >
PyGetSetDef DataPointVectorWrapper< Tdp_ >::__getset__[] = {
    { const_cast< char* >( "__array_struct__" ), (getter)&DataPointVectorWrapper::__get___array__struct, NULL },
    { const_cast< char* >( "shape" ), (getter)&DataPointVectorWrapper::__get__shape, NULL },
    { NULL }
};


template< typename Titer_ >
class STLIteratorWrapper
{
protected:
    PyObject_VAR_HEAD
    Titer_ theIdx;
    Titer_ theEnd; 

public:
    static PyTypeObject __class__;

public:
    void* operator new( size_t )
    {
        return PyObject_New( STLIteratorWrapper, &__class__ );
    }

    void operator delete( void* ptr )
    {
        reinterpret_cast< PyObject* >( ptr )->ob_type->tp_free( reinterpret_cast< PyObject* >( ptr ) );
    }

    template< typename Trange_ >
    STLIteratorWrapper( Trange_ const& range )
        : theIdx( boost::begin( range ) ), theEnd( boost::end( range ) )
    {
    }

    ~STLIteratorWrapper()
    {
    }

public:
    static PyTypeObject* __class_init__()
    {
        PyType_Ready( &__class__ );
        return &__class__;
    }

    template< typename Trange_ >
    static PyObject* create( Trange_ const& range )
    {
        return reinterpret_cast< PyObject* >( new STLIteratorWrapper( range ) );
    }

    static void __dealloc__( STLIteratorWrapper* self )
    {
        delete self;
    }

    static PyObject* __next__( STLIteratorWrapper* self )
    {
        if ( self->theIdx == self->theEnd )
            return NULL;

        return py::incref( py::object( *( self->theIdx++ ) ).ptr() );
    }
};

template< typename Titer_ >
PyTypeObject STLIteratorWrapper< Titer_ >::__class__ = {
	PyObject_HEAD_INIT( &PyType_Type )
	0,					/* ob_size */
	"ecell._ecs.STLIteratorWrapper", /* tp_name */
	sizeof( STLIteratorWrapper ), /* tp_basicsize */
	0,					/* tp_itemsize */
	/* methods */
	(destructor)&STLIteratorWrapper::__dealloc__, /* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_HAVE_CLASS | Py_TPFLAGS_HAVE_WEAKREFS | Py_TPFLAGS_HAVE_ITER,/* tp_flags */
	0,					/* tp_doc */
	0,	/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	PyObject_SelfIter,  /* tp_iter */
	(iternextfunc)&STLIteratorWrapper::__next__,		/* tp_iternext */
	0,		        	/* tp_methods */
	0,					/* tp_members */
	0,                  /* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	0,			        /* tp_init */
	PyType_GenericAlloc,			/* tp_alloc */
	PyType_GenericNew,			/* tp_new */
	PyObject_Del,			/* tp_free */
};


class PropertyAttributesIterator
{
protected:
    PyObject_VAR_HEAD
    PropertyAttributes const& theImpl;
    int theIdx;

public:
    static PyTypeObject __class__;

public:
    void* operator new( size_t )
    {
        return PyObject_New( PropertyAttributesIterator, &__class__ );
    }

    void operator delete( void* ptr )
    {
        reinterpret_cast< PyObject* >( ptr )->ob_type->tp_free( reinterpret_cast< PyObject* >( ptr ) );
    }

    PropertyAttributesIterator( PropertyAttributes const& impl )
        : theImpl( impl ), theIdx( 0 )
    {
    }

    ~PropertyAttributesIterator()
    {
    }

public:
    static PyTypeObject* __class_init__()
    {
        PyType_Ready( &__class__ );
        return &__class__;
    }

    static PropertyAttributesIterator* create( PropertyAttributes const& impl )
    {
        return new PropertyAttributesIterator( impl );
    }

    static void __dealloc__( PropertyAttributesIterator* self )
    {
        delete self;
    }

    static PyObject* __next__( PropertyAttributesIterator* self );
};


PyTypeObject PropertyAttributesIterator::__class__ = {
	PyObject_HEAD_INIT( &PyType_Type )
	0,					/* ob_size */
	"ecell._ecs.PropertyAttributesIterator", /* tp_name */
	sizeof( PropertyAttributesIterator ), /* tp_basicsize */
	0,					/* tp_itemsize */
	/* methods */
	(destructor)&PropertyAttributesIterator::__dealloc__, /* tp_dealloc */
	0,					/* tp_print */
	0,					/* tp_getattr */
	0,					/* tp_setattr */
	0,					/* tp_compare */
	0,					/* tp_repr */
	0,					/* tp_as_number */
	0,					/* tp_as_sequence */
	0,					/* tp_as_mapping */
	0,					/* tp_hash */
	0,					/* tp_call */
	0,					/* tp_str */
	PyObject_GenericGetAttr,		/* tp_getattro */
	0,					/* tp_setattro */
	0,					/* tp_as_buffer */
	Py_TPFLAGS_HAVE_CLASS | Py_TPFLAGS_HAVE_WEAKREFS | Py_TPFLAGS_HAVE_ITER,/* tp_flags */
	0,					/* tp_doc */
	0,	/* tp_traverse */
	0,					/* tp_clear */
	0,					/* tp_richcompare */
	0,					/* tp_weaklistoffset */
	PyObject_SelfIter,  /* tp_iter */
	(iternextfunc)&PropertyAttributesIterator::__next__,		/* tp_iternext */
	0,		        	/* tp_methods */
    0,                  /* tp_members */
	0,                  /* tp_getset */
	0,					/* tp_base */
	0,					/* tp_dict */
	0,					/* tp_descr_get */
	0,					/* tp_descr_set */
	0,					/* tp_dictoffset */
	0,			        /* tp_init */
	PyType_GenericAlloc,			/* tp_alloc */
	PyType_GenericNew,			/* tp_new */
	PyObject_Del,			/* tp_free */
};

static std::string VariableReference___str__( VariableReference const* self )
{
    std::string retval;
    retval += "[";
    retval += self->getName().empty() ? "<anonymous>": self->getName();
    retval += " (#";
    retval += stringCast( self->getSerial() );
    retval += "): ";
    retval += "coefficient=";
    retval += stringCast( self->getCoefficient() );
    retval += ", ";
    retval += "variable=";
    retval += self->getVariable() ? self->getVariable()->asString():
                                    self->getFullID().asString();
    retval += ", ";
    retval += "accessor=";
    retval += ( self->isAccessor() ? "true": "false" );
    retval += "]";
    return retval;
}


class VariableReferences
{
public:
    VariableReferences( Process* proc ): theProc( proc ) {}

    Integer add( String const& name, String const& fullID, Integer const& coef,
              bool isAccessor )
    {
        return theProc->registerVariableReference( name, FullID( fullID ),
                                                   coef, isAccessor );
    }

    Integer add( String const& name, String const& fullID, Integer const& coef )
    {
        return theProc->registerVariableReference( name, FullID( fullID ),
                                                   coef, false );
    }

    Integer add( String const& fullID, Integer const& coef, bool isAccessor )
    {
        return theProc->registerVariableReference( FullID( fullID ),
                                                   coef, isAccessor );
    }

    Integer add( String const& fullID, Integer const& coef )
    {
        return theProc->registerVariableReference( FullID( fullID ),
                                                   coef, false );
    }

    Integer add( String const& name, Variable* var, Integer const& coef,
              bool isAccessor )
    {
        return theProc->registerVariableReference( name, var,
                                                   coef, isAccessor );
    }

    Integer add( String const& name, Variable* var, Integer const& coef )
    {
        return theProc->registerVariableReference( name, var, coef, false );
    }

    Integer add( Variable* var, Integer const& coef, bool isAccessor )
    {
        return theProc->registerVariableReference( var, coef, isAccessor );
    }

    Integer add( Variable* var, Integer const& coef )
    {
        return theProc->registerVariableReference( var, coef, false );
    }

    void remove( String const& name )
    {
        theProc->removeVariableReference( name );
    }

    void remove( Integer const id )
    {
        theProc->removeVariableReference( id );
    }

    VariableReference const& __getitem__( py::object name )
    {
        if ( PyInt_Check( name.ptr() ) )
        {
            Integer id( PyInt_AS_LONG( name.ptr() ) );
            return theProc->getVariableReference( id );
        }
        else if ( PyString_Check( name.ptr() ) )
        {
            std::string nameStr( PyString_AS_STRING( name.ptr() ),
                                 PyString_GET_SIZE( name.ptr() ) );
            return theProc->getVariableReference( nameStr );
        }
        PyErr_SetString( PyExc_TypeError,
                         "The argument is neither an integer nor a string" );
        py::throw_error_already_set();
        throw std::exception();
    }

    Py_ssize_t __len__()
    {
        return theProc->getVariableReferenceVector().size();
    }

    py::list getPositivesReferences()
    {
        Process::VariableReferenceVector const& refs(
                theProc->getVariableReferenceVector() );

        py::list retval;
        std::for_each(
            refs.begin() + theProc->getPositiveVariableReferenceOffset(),
            refs.end(), boost::bind(
                &py::list::append< VariableReference >, &retval,
                _1 ) );
        return retval;
    }

    py::list getNegativeReferences()
    {
        Process::VariableReferenceVector const& refs(
                theProc->getVariableReferenceVector() );

        py::list retval;
        std::for_each(
            refs.begin(),
            refs.begin() + theProc->getZeroVariableReferenceOffset(),
            boost::bind( &py::list::append< VariableReference >, &retval,
                         _1 ) );
        return retval;
    }

    py::list getZeroReferences()
    {
        Process::VariableReferenceVector const& refs(
                theProc->getVariableReferenceVector() );

        py::list retval;
        std::for_each(
            refs.begin() + theProc->getZeroVariableReferenceOffset(),
            refs.begin() + theProc->getPositiveVariableReferenceOffset(),
            boost::bind( &py::list::append< VariableReference >, &retval,
                         _1 ) );
        return retval;
    }

    PyObject* __iter__()
    {
        return STLIteratorWrapper< Process::VariableReferenceVector::const_iterator >::create( theProc->getVariableReferenceVector() );
    }

    std::string __str__()
    {
        Process::VariableReferenceVector const& refs(
                theProc->getVariableReferenceVector() );

        std::string retval;

        retval += '[';
        for ( Process::VariableReferenceVector::const_iterator
                b( refs.begin() ), i( b ), e( refs.end() );
                i != e; ++i )
        {
            if ( i != b )
                retval += ", ";
            retval += VariableReference___str__( &*i );
        }
        retval += ']';

        return retval;
    } 

private:
    Process* theProc;
};

class DataPointVectorSharedPtrConverter
{
public:
    static void addToRegistry()
    {
        py::to_python_converter< boost::shared_ptr< DataPointVector >,
                                 DataPointVectorSharedPtrConverter >();
    }

    static PyObject* 
    convert( boost::shared_ptr< DataPointVector > const& aVectorSharedPtr )
    {
        return aVectorSharedPtr->getElementSize() == sizeof( DataPoint ) ?
                reinterpret_cast< PyObject* >(
                    DataPointVectorWrapper< DataPoint >::create(
                        aVectorSharedPtr ) ):
                reinterpret_cast< PyObject* >(
                    DataPointVectorWrapper< LongDataPoint >::create(
                        aVectorSharedPtr ) );
    }
};

class PythonWarningHandler: public libecs::WarningHandler
{
public:
    PythonWarningHandler() {}

    PythonWarningHandler( py::handle<> aCallable )
        : thePyObject( aCallable )
    {
    }
      
    virtual ~PythonWarningHandler() {}

    virtual void operator()( String const& msg ) const
    {
        if ( thePyObject )
          PyObject_CallFunctionObjArgs( thePyObject.get(), py::object( msg ).ptr(), NULL );
    }

private:
    py::handle<> thePyObject;
};

template< typename T >
class PythonDynamicModule;

struct PythonEntityBaseBase
{
protected:
    static void appendDictToSet( std::set< String >& retval, std::string const& aPrivPrefix, PyObject* aObject )
    {
        py::handle<> aSelfDict( py::allow_null( PyObject_GetAttrString( aObject, const_cast< char* >( "__dict__" ) ) ) );
        if ( !aSelfDict )
        {
            PyErr_Clear();
            return;
        }

        if ( !PyMapping_Check( aSelfDict.get() ) )
        {
            return;
        }

        py::handle<> aKeyList( PyMapping_Items( aSelfDict.get() ) );
        BOOST_ASSERT( PyList_Check( aKeyList.get() ) );
        for ( Py_ssize_t i( 0 ), e( PyList_GET_SIZE( aKeyList.get() ) ); i < e; ++i )
        {
            py::handle<> aKeyValuePair( py::borrowed( PyList_GET_ITEM( aKeyList.get(), i ) ) );
            BOOST_ASSERT( PyTuple_Check( aKeyValuePair.get() ) && PyTuple_GET_SIZE( aKeyValuePair.get() ) == 2 );
            py::handle<> aKey( py::borrowed( PyTuple_GET_ITEM( aKeyValuePair.get(), 0 ) ) );
            BOOST_ASSERT( PyString_Check( aKey.get() ) );
            if ( PyString_GET_SIZE( aKey.get() ) >= static_cast< Py_ssize_t >( aPrivPrefix.size() )
                    && memcmp( PyString_AS_STRING( aKey.get() ), aPrivPrefix.data(), aPrivPrefix.size() ) == 0 )
            {
                continue;
            }

            if ( PyString_GET_SIZE( aKey.get() ) >= 2
                    && memcmp( PyString_AS_STRING( aKey.get() ), "__", 2 ) == 0 )
            {
                continue;
            }

            py::handle<> aValue( py::borrowed( PyTuple_GET_ITEM( aKeyValuePair.get(), 1 ) ) );
            if ( !PolymorphRetriever::isConvertible( aValue.get() ) )
            {
                continue;
            }

            retval.insert( String( PyString_AS_STRING( aKey.get() ), PyString_GET_SIZE( aKey.get() ) ) );
        }
    }

    static void addAttributesFromBases( std::set< String >& retval, std::string const& aPrivPrefix, PyObject* anUpperBound, PyObject* tp )
    {
        BOOST_ASSERT( PyType_Check( tp ) );

        if ( anUpperBound == tp )
        {
            return;
        }

        py::handle<> aBasesList( py::allow_null( PyObject_GetAttrString( tp, const_cast< char* >( "__bases__" ) ) ) );
        if ( !aBasesList )
        {
            PyErr_Clear();
            return;
        }

        if ( !PyTuple_Check( aBasesList.get() ) )
        {
            return;
        }

        for ( Py_ssize_t i( 0 ), ie( PyTuple_GET_SIZE( aBasesList.get() ) ); i < ie; ++i )
        {
            py::handle<> aBase( py::borrowed( PyTuple_GET_ITEM( aBasesList.get(), i ) ) );
            appendDictToSet( retval, aPrivPrefix, aBase.get() );
            addAttributesFromBases( retval, aPrivPrefix, anUpperBound, aBase.get() );
        }
    }

    static void removeAttributesFromBases( std::set< String >& retval, PyObject *tp )
    {
        BOOST_ASSERT( PyType_Check( tp ) );

        py::handle<> aBasesList( py::allow_null( PyObject_GetAttrString( tp, const_cast< char* >( "__bases__" ) ) ) );
        if ( !aBasesList )
        {
            PyErr_Clear();
            return;
        }

        if ( !PyTuple_Check( aBasesList.get() ) )
        {
            return;
        }

        for ( Py_ssize_t i( 0 ), ie( PyTuple_GET_SIZE( aBasesList.get() ) ); i < ie; ++i )
        {
            py::handle<> aBase( py::borrowed( PyTuple_GET_ITEM( aBasesList.get(), i ) ) );
            removeAttributesFromBases( retval, aBase.get() );

            py::handle<> aBaseDict( py::allow_null( PyObject_GetAttrString( aBase.get(), const_cast< char* >( "__dict__" ) ) ) );
            if ( !aBaseDict )
            {
                PyErr_Clear();
                return;
            }

            if ( !PyMapping_Check( aBaseDict.get() ) )
            {
                return;
            }

            py::handle<> aKeyList( PyMapping_Keys( aBaseDict.get() ) );
            BOOST_ASSERT( PyList_Check( aKeyList.get() ) );
            for ( Py_ssize_t j( 0 ), je( PyList_GET_SIZE( aKeyList.get() ) ); j < je; ++j )
            {
                py::handle<> aKey( py::borrowed( PyList_GET_ITEM( aKeyList.get(), i ) ) );
                BOOST_ASSERT( PyString_Check( aKey.get() ) );
                String aKeyStr( PyString_AS_STRING( aKey.get() ), PyString_GET_SIZE( aKey.get() ) );  
                retval.erase( aKeyStr );
            }
        }
    }

    PythonEntityBaseBase() {}
};

template< typename Tderived_, typename Tbase_ >
class PythonEntityBase: public Tbase_, public PythonEntityBaseBase, public py::wrapper< Tbase_ >
{
public:
    virtual ~PythonEntityBase()
    {
        py::decref( py::detail::wrapper_base_::owner( this ) );
    }

    PythonDynamicModule< Tderived_ > const& getModule() const
    {
        return theModule;
    }

    Polymorph defaultGetProperty( String const& aPropertyName ) const
    {
        PyObject* aSelf( py::detail::wrapper_base_::owner( this ) );
        py::handle<> aValue( py::allow_null( PyObject_GenericGetAttr( aSelf, py::handle<>( PyString_InternFromString( const_cast< char* >( aPropertyName.c_str() ) ) ).get() ) ) );
        if ( !aValue )
        {
            PyErr_Clear();
            THROW_EXCEPTION_INSIDE( NoSlot, 
                    "failed to retrieve property attributes "
                    "for [" + aPropertyName + "]" );
        }

        return py::extract< Polymorph >( aValue.get() );
    }

    PropertyAttributes defaultGetPropertyAttributes( String const& aPropertyName ) const
    {
        return PropertyAttributes( PropertySlotBase::POLYMORPH, true, true, true, true, true );
    }


    void defaultSetProperty( String const& aPropertyName, Polymorph const& aValue )
    {
        PyObject* aSelf( py::detail::wrapper_base_::owner( this ) );
        PyObject_GenericSetAttr( aSelf, py::handle<>( PyString_InternFromString( const_cast< char* >( aPropertyName.c_str() ) ) ).get(), py::object( aValue ).ptr() );
        if ( PyErr_Occurred() )
        {
            PyErr_Clear();
            THROW_EXCEPTION_INSIDE( NoSlot, 
                            "failed to set property [" + aPropertyName + "]" );
        }
    }

    std::vector< String > defaultGetPropertyList() const
    {
        PyObject* aSelf( py::detail::wrapper_base_::owner( this ) );
        std::set< String > aPropertySet;

        if ( thePrivPrefix.empty() )
        {
            PyObject* anOwner( py::detail::wrapper_base_::owner( this ) );
            BOOST_ASSERT( anOwner != NULL );
            thePrivPrefix = String( "_" ) + anOwner->ob_type->tp_name;
        }

        appendDictToSet( aPropertySet, thePrivPrefix, aSelf );

        PyObject* anUpperBound(
                reinterpret_cast< PyObject* >(
                    py::objects::registered_class_object(
                        typeid( Tbase_ ) ).get() ) );
        addAttributesFromBases( aPropertySet, thePrivPrefix, anUpperBound,
                reinterpret_cast< PyObject* >( aSelf->ob_type ) );
        removeAttributesFromBases( aPropertySet, anUpperBound );

        std::vector< String > retval;
        for ( std::set< String >::iterator i( aPropertySet.begin() ), e( aPropertySet.end() ); i != e; ++i )
        {
            retval.push_back( *i );
        }

        return retval;
    }

    PropertyInterface< Tbase_ > const& _getPropertyInterface() const;

    PropertySlotBase const* getPropertySlot( String const& aPropertyName ) const
    {
        return _getPropertyInterface().getPropertySlot( aPropertyName ); \
    }

    virtual void setProperty( String const& aPropertyName, Polymorph const& aValue )
    {
    }

    Polymorph getProperty( String const& aPropertyName ) const
    {
        return _getPropertyInterface().getProperty( *this, aPropertyName );
    }

    void loadProperty( String const& aPropertyName, Polymorph const& aValue )
    {
        return _getPropertyInterface().loadProperty( *this, aPropertyName, aValue );
    }

    Polymorph saveProperty( String const& aPropertyName ) const
    {
        return _getPropertyInterface().saveProperty( *this, aPropertyName );
    }

    std::vector< String > getPropertyList() const
    {
        return _getPropertyInterface().getPropertyList( *this );
    }

    PropertySlotProxy* createPropertySlotProxy( String const& aPropertyName )
    {
        return _getPropertyInterface().createPropertySlotProxy( *this, aPropertyName );
    }

    PropertyAttributes
    getPropertyAttributes( String const& aPropertyName ) const
    {
        return _getPropertyInterface().getPropertyAttributes( *this, aPropertyName );
    }

    virtual PropertyInterfaceBase const& getPropertyInterface() const
    {
        return _getPropertyInterface();
    }

    void __init__()
    {
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( py::detail::wrapper_base_::owner( this ) ) ), "__init__", true ) );
        if ( meth )
        {
            meth.get()();
        }
    }

    PythonEntityBase( PythonDynamicModule< Tderived_ > const& aModule )
        : theModule( aModule ) {}

    static void addToRegistry()
    {
        py::objects::register_dynamic_id< Tderived_ >();
        py::objects::register_dynamic_id< Tbase_ >();
        py::objects::register_conversion< Tderived_, Tbase_ >( false );
        py::objects::register_conversion< Tbase_, Tderived_ >( true );
    }

protected:

    PythonDynamicModule< Tderived_ > const& theModule;
    mutable String thePrivPrefix;
};

class PythonProcess: public PythonEntityBase< PythonProcess, Process >
{
public:
    virtual ~PythonProcess() {}

    LIBECS_DM_INIT_PROP_INTERFACE()
    {
        INHERIT_PROPERTIES( Process );
    }

    virtual void preinitialize()
    {
        Process::preinitialize();
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( py::detail::wrapper_base_::owner( this ) ) ), "preinitialize", true ) );
        if ( meth )
            meth.get()();
    }

    virtual void initialize()
    {
        Process::initialize();
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( py::detail::wrapper_base_::owner( this ) ) ), "initialize", true ) );
        if ( meth )
            meth.get()();
    }

    virtual void fire()
    {
        py::object retval( theFireMethod() );
        if ( retval )
        {
            setActivity( py::extract< Real >( retval ) );
        } 
    }

    virtual bool isContinuous() const
    {
        PyObject* aSelf( py::detail::wrapper_base_::owner( this ) );
        py::handle<> anIsContinuousDescr( py::allow_null( PyObject_GenericGetAttr( reinterpret_cast< PyObject* >( aSelf->ob_type ), py::handle<>( PyString_InternFromString( "IsContinuous" ) ).get() ) ) );
        if ( !anIsContinuousDescr )
        {
            PyErr_Clear();
            return Process::isContinuous();
        }

        descrgetfunc aDescrGetFunc( anIsContinuousDescr.get()->ob_type->tp_descr_get );
        if ( ( anIsContinuousDescr.get()->ob_type->tp_flags & Py_TPFLAGS_HAVE_CLASS ) && aDescrGetFunc )
        {
            return py::extract< bool >( py::handle<>( aDescrGetFunc( anIsContinuousDescr.get(), aSelf, reinterpret_cast< PyObject* >( aSelf->ob_type ) ) ).get() );
        }

        return py::extract< bool >( anIsContinuousDescr.get() );
    }

    PythonProcess( PythonDynamicModule< PythonProcess > const& aModule )
        : PythonEntityBase< PythonProcess, Process >( aModule ) {}

    py::object theFireMethod;
};


class PythonVariable: public PythonEntityBase< PythonVariable, Variable >
{
public:
    LIBECS_DM_INIT_PROP_INTERFACE()
    {
        INHERIT_PROPERTIES( Variable );
    }

    virtual void preinitialize()
    {
        Variable::preinitialize();
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( py::detail::wrapper_base_::owner( this ) ) ), "preinitialize", true ) );
        if ( meth )
            meth.get()();
    }

    virtual void initialize()
    {
        Variable::initialize();
        PyObject* aSelf( py::detail::wrapper_base_::owner( this ) );
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( aSelf ) ), "initialize", true ) );
        if ( meth )
            meth.get()();
        theOnValueChangingMethod = py::handle<>( py::allow_null( PyObject_GenericGetAttr( aSelf, py::handle<>( PyString_InternFromString( const_cast< char* >( "onValueChanging" ) ) ).get() ) ) );
        if ( !theOnValueChangingMethod )
        {
            PyErr_Clear();
        }
    }

    virtual SET_METHOD( Real, Value )
    {
        if ( theOnValueChangingMethod )
        {
            if ( !PyCallable_Check( theOnValueChangingMethod.get() ) )
            {
                PyErr_SetString( PyExc_TypeError, "object is not callable" );
                py::throw_error_already_set();
            }

            py::handle<> aResult( PyObject_CallFunction( theOnValueChangingMethod.get(), const_cast<char*>("f"), value ) );
            if ( !aResult )
            {
                py::throw_error_already_set();
            }
            else
            {
                if ( !PyObject_IsTrue( aResult.get() ) )
                {
                    return;
                }
            }
        }
        else
        {
            PyErr_Clear();
        }
        Variable::setValue( value );
    }

    PythonVariable( PythonDynamicModule< PythonVariable > const& aModule )
        : PythonEntityBase< PythonVariable, Variable >( aModule ) {}

    py::handle<> theOnValueChangingMethod;
};


class PythonSystem: public PythonEntityBase< PythonSystem, System >
{
public:
    LIBECS_DM_INIT_PROP_INTERFACE()
    {
        INHERIT_PROPERTIES( System );
    }

    virtual void preinitialize()
    {
        System::preinitialize();
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( py::detail::wrapper_base_::owner( this ) ) ), "preinitialize", true ) );
        if ( meth )
            meth.get()();
    }

    virtual void initialize()
    {
        System::initialize();
        boost::optional< py::object > meth( generic_getattr( py::object( py::borrowed( py::detail::wrapper_base_::owner( this ) ) ), "initialize", true ) );
        if ( meth )
            meth.get()();
    }

    PythonSystem( PythonDynamicModule< PythonSystem > const& aModule )
        : PythonEntityBase< PythonSystem, System >( aModule ) {}
};

template<typename T_>
struct DeduceEntityType
{
    static EntityType value;
};

template<>
EntityType DeduceEntityType< PythonProcess >::value( EntityType::PROCESS );

template<>
EntityType DeduceEntityType< PythonVariable >::value( EntityType::VARIABLE );

template<>
EntityType DeduceEntityType< PythonSystem >::value( EntityType::SYSTEM );

template< typename T_ >
class PythonDynamicModule: public DynamicModule< EcsObject >
{
public:
    typedef DynamicModule< EcsObject > Base;

    struct make_ptr_instance: public py::objects::make_instance_impl< T_, py::objects::pointer_holder< T_*, T_ >, make_ptr_instance > 
    {
        typedef py::objects::pointer_holder< T_*, T_ > holder_t;

        template <class Arg>
        static inline holder_t* construct(void* storage, PyObject* arg, Arg& x)
        {
            py::detail::initialize_wrapper( arg, boost::get_pointer(x) );
            return new (storage) holder_t(x);
        }

        template<typename Ptr>
        static inline PyTypeObject* get_class_object(Ptr const& x)
        {
            return static_cast< T_ const* >( boost::get_pointer(x) )->getModule().getPythonType();
        }
    };

    virtual EcsObject* createInstance() const;

    virtual const char* getFileName() const
    {
        const char* aRetval( 0 );
        try
        {
            py::handle<> aPythonModule( PyImport_Import( py::getattr( thePythonClass, "__module__" ).ptr() ) );
            if ( !aPythonModule )
            {
                py::throw_error_already_set();
            }
            aRetval = py::extract< const char * >( py::getattr( py::object( aPythonModule ), "__file__" ) );
        }
        catch ( py::error_already_set )
        {
            PyErr_Clear();
        }
        return aRetval;
    }

    virtual const char *getModuleName() const 
    {
        return reinterpret_cast< PyTypeObject* >( thePythonClass.ptr() )->tp_name;
    }

    virtual const DynamicModuleInfo* getInfo() const
    {
        return &thePropertyInterface;
    }

    PyTypeObject* getPythonType() const
    {
        return reinterpret_cast< PyTypeObject* >( thePythonClass.ptr() );
    }
 
    PythonDynamicModule( py::object aPythonClass )
        : Base( DM_TYPE_DYNAMIC ),
           thePythonClass( aPythonClass ),
           thePropertyInterface( getModuleName(),
                                 DeduceEntityType< T_ >::value.asString() )
    {
    }

private:
    py::object thePythonClass;
    PropertyInterface< T_ > thePropertyInterface;
};

template< typename Tderived_, typename Tbase_ >
inline PropertyInterface< Tbase_ > const&
PythonEntityBase< Tderived_, Tbase_ >::_getPropertyInterface() const
{
    return *reinterpret_cast< PropertyInterface< Tbase_ > const* >( theModule.getInfo() );
}

template< typename T_ >
EcsObject* PythonDynamicModule< T_ >::createInstance() const
{
    T_* retval( new T_( *this ) );

    if ( !make_ptr_instance::execute( retval ) )
    {
        delete retval;
        std::string anErrorStr( "Instantiation failure" );
        PyObject* aPyErrObj( PyErr_Occurred() );
        if ( aPyErrObj )
        {
            anErrorStr += "(";
            anErrorStr += aPyErrObj->ob_type->tp_name;
            anErrorStr += ": ";
            py::handle<> aPyErrStrRepr( PyObject_Str( aPyErrObj ) );
            BOOST_ASSERT( PyString_Check( aPyErrStrRepr.get() ) );
            anErrorStr.insert( anErrorStr.size(),
                PyString_AS_STRING( aPyErrStrRepr.get() ),
                PyString_GET_SIZE( aPyErrStrRepr.get() ) );
            anErrorStr += ")";
            PyErr_Clear();
        }
        throw std::runtime_error( anErrorStr );
    }

    retval->__init__();
    return retval;
}

template< typename T_ >
inline PyObject* to_python_indirect_fun( T_* arg )
{
    return py::to_python_indirect< T_, py::detail::make_reference_holder >()( arg );
}

class AbstractSimulator: public Model
{
public:
    py::list getStepperList() const
    {
        Model::StepperMap const& aStepperMap( getStepperMap() );
        py::list retval;

        for( Model::StepperMap::const_iterator i( aStepperMap.begin() );
             i != aStepperMap.end(); ++i )
        {
            retval.append( py::object( (*i).first ) );
        }

        return retval;
    }

    std::vector< String >
    getStepperPropertyList( String const& aStepperID ) const
    {
        return getStepper( aStepperID )->getPropertyList();
    }

    PropertyAttributes
    getStepperPropertyAttributes( String const& aStepperID, 
                                  String const& aPropertyName ) const
    {
        return getStepper( aStepperID )->getPropertyAttributes( aPropertyName );
    }

    void setStepperProperty( String const& aStepperID,
                             String const& aPropertyName,
                             Polymorph const& aValue )
    {
        getStepper( aStepperID )->setProperty( aPropertyName, aValue );
    }

    Polymorph
    getStepperProperty( String const& aStepperID,
                        String const& aPropertyName ) const
    {
        return getStepper( aStepperID )->getProperty( aPropertyName );
    }

    void loadStepperProperty( String const& aStepperID,
                              String const& aPropertyName,
                              Polymorph const& aValue )
    {
        getStepper( aStepperID )->loadProperty( aPropertyName, aValue );
    }

    Polymorph
    saveStepperProperty( String const& aStepperID,
                         String const& aPropertyName ) const
    {
        return getStepper( aStepperID )->saveProperty( aPropertyName );
    }

    String
    getStepperClassName( String const& aStepperID ) const
    {
        return getStepper( aStepperID )->getPropertyInterface().getClassName();
    }

    py::dict getClassInfo( String const& aClassname ) const
    {
        py::dict retval;
        for ( DynamicModuleInfo::EntryIterator* anInfo(
              getPropertyInterface( aClassname ).getInfoFields() );
              anInfo->next(); )
        {
            retval[ anInfo->current().first ] =
                *reinterpret_cast< const libecs::Polymorph* >(
                    anInfo->current().second );
        }
        return retval;
    }

    Polymorph 
    getEntityList( String const& anEntityTypeString,
                   String const& aSystemPathString ) const
    {
        const EntityType anEntityType( anEntityTypeString );
        const SystemPath aSystemPath( aSystemPathString );

        if( aSystemPath.size() == 0 )
        {
            PolymorphVector aVector;
            if( anEntityType == EntityType::SYSTEM )
            {
                aVector.push_back( Polymorph( "/" ) );
            }
            return Polymorph( aVector );
        }

        System const* aSystemPtr( getSystem( aSystemPath ) );

        switch( anEntityType )
        {
        case EntityType::VARIABLE:
            return aSystemPtr->getVariableList();
        case EntityType::PROCESS:
            return aSystemPtr->getProcessList();
        case EntityType::SYSTEM:
            return aSystemPtr->getSystemList();
        default:
            break;
        }

        NEVER_GET_HERE;
    }

    std::vector< String >
    getEntityPropertyList( String const& aFullIDString ) const
    {
        return getEntity( FullID( aFullIDString ) )->getPropertyList();
    }

    bool entityExists( String const& aFullIDString ) const
    {
        try
        {
            (void)getEntity( FullID( aFullIDString ) );
        }
        catch( const NotFound& )
        {
            return false;
        }

        return true;
    }

    void setEntityProperty( String const& aFullPNString,
                            Polymorph const& aValue )
    {
        FullPN aFullPN( aFullPNString );
        Entity* const anEntityPtr( getEntity( aFullPN.getFullID() ) );

        anEntityPtr->setProperty( aFullPN.getPropertyName(), aValue );
    }

    Polymorph
    getEntityProperty( String const& aFullPNString ) const
    {
        FullPN aFullPN( aFullPNString );
        Entity const * const anEntityPtr( getEntity( aFullPN.getFullID() ) );
                
        return anEntityPtr->getProperty( aFullPN.getPropertyName() );
    }

    void loadEntityProperty( String const& aFullPNString,
                             Polymorph const& aValue )
    {
        FullPN aFullPN( aFullPNString );
        Entity* const anEntityPtr( getEntity( aFullPN.getFullID() ) );

        anEntityPtr->loadProperty( aFullPN.getPropertyName(), aValue );
    }

    Polymorph
    saveEntityProperty( String const& aFullPNString ) const
    {
        FullPN aFullPN( aFullPNString );
        Entity const* const anEntityPtr( getEntity( aFullPN.getFullID() ) );

        return anEntityPtr->saveProperty( aFullPN.getPropertyName() );
    }

    PropertyAttributes
    getEntityPropertyAttributes( String const& aFullPNString ) const
    {
        FullPN aFullPN( aFullPNString );
        Entity const* const anEntityPtr( getEntity( aFullPN.getFullID() ) );

        return anEntityPtr->getPropertyAttributes( aFullPN.getPropertyName() );
    }

    String
    getEntityClassName( String const& aFullIDString ) const
    {
        FullID aFullID( aFullIDString );
        Entity const* const anEntityPtr( getEntity( aFullID ) );

        return anEntityPtr->getPropertyInterface().getClassName();
    }

    Logger* createLogger( String const& aFullPNString )
    {
        return createLogger( aFullPNString, Logger::Policy() );
    }

    Logger* createLogger( String const& aFullPNString,
                          Logger::Policy const& aParamList = Logger::Policy() )
    {
        Logger* retval( getLoggerBroker().createLogger(
            FullPN( aFullPNString ), aParamList ) );

        return retval;
    }

    Logger* createLogger( String const& aFullPNString,
                          py::object aParamList )
    {
        if ( !PySequence_Check( aParamList.ptr() )
             || PySequence_Size( aParamList.ptr() ) != 4 )
        {
            THROW_EXCEPTION( Exception,
                             "second argument must be a tuple of 4 items");
        }

        return createLogger( aFullPNString,
                Logger::Policy(
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 0 ] ).ptr() ),
                    PyFloat_AsDouble( static_cast< py::object >( aParamList[ 1 ] ).ptr() ),
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 2 ] ).ptr() ),
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 3 ] ).ptr() ) ) );
    }

    py::list getLoggerList() const
    {
        py::list retval;

        LoggerBroker const& aLoggerBroker( getLoggerBroker() );

        for( LoggerBroker::const_iterator
                i( aLoggerBroker.begin() ), end( aLoggerBroker.end() );
             i != end; ++i )
        {
            retval.append( py::object( (*i).first.asString() ) );
        }

        return retval;
    }

    boost::shared_ptr< DataPointVector > 
    getLoggerData( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getData();
    }

    boost::shared_ptr< DataPointVector >
    getLoggerData( String const& aFullPNString, 
                   Real const& startTime, Real const& endTime ) const
    {
        return getLogger( aFullPNString )->getData( startTime, endTime );
    }

    boost::shared_ptr< DataPointVector >
    getLoggerData( String const& aFullPNString,
                   Real const& start, Real const& end, 
                   Real const& interval ) const
    {
        return getLogger( aFullPNString )->getData( start, end, interval );
    }

    Real 
    getLoggerStartTime( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getStartTime();
    }

    Real 
    getLoggerEndTime( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getEndTime();
    }


    void setLoggerPolicy( String const& aFullPNString, 
                          Logger::Policy const& pol )
    {
        typedef PolymorphValue::Tuple Tuple;
        getLogger( aFullPNString )->setLoggerPolicy( pol );
    }

    void setLoggerPolicy( String const& aFullPNString,
                          py::object aParamList )
    {
        if ( !PySequence_Check( aParamList.ptr() )
            || PySequence_Size( aParamList.ptr() ) != 4 )
        {
            THROW_EXCEPTION( Exception,
                             "second parameter must be a tuple of 4 items");
        }

        return setLoggerPolicy( aFullPNString,
                Logger::Policy(
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 0 ] ).ptr() ),
                    PyFloat_AsDouble( static_cast< py::object >( aParamList[ 1 ] ).ptr() ),
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 2 ] ).ptr() ),
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 3 ] ).ptr() ) ) );
    }

    Logger::Policy
    getLoggerPolicy( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getLoggerPolicy();
    }

    Logger::size_type 
    getLoggerSize( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getSize();
    }

    std::pair< Real, String > getNextEvent() const
    {
        StepperEvent const& aNextEvent( getTopEvent() );

        return std::make_pair(
            static_cast< Real >( aNextEvent.getTime() ),
            aNextEvent.getStepper()->getID() );
    }

    py::object getDMInfo() const
    {
        typedef ModuleMaker< EcsObject >::ModuleMap ModuleMap;
        const ModuleMap& modules( theEcsObjectMaker.getModuleMap() );
        py::list retval;

        for( ModuleMap::const_iterator i( modules.begin() );
                    i != modules.end(); ++i )
        {
            const PropertyInterfaceBase* info(
                reinterpret_cast< const PropertyInterfaceBase *>(
                    i->second->getInfo() ) );
            const char* aFilename( i->second->getFileName() );

            retval.append( py::make_tuple(
                py::object( info->getTypeName() ),
                py::object( i->second->getModuleName() ),
                py::object( aFilename ? aFilename: "" ) ) );
        }

        return retval;
    }

    PropertyInterfaceBase::PropertySlotMap const&
    getPropertyInfo( String const& aClassname ) const
    {
        return getPropertyInterface( aClassname ).getPropertySlotMap();
    }

    Logger* getLogger( String const& aFullPNString ) const
    {
        return getLoggerBroker().getLogger( aFullPNString );
    }

    void removeLogger( String const& aFullPNString )
    {
        getLoggerBroker().removeLogger( FullPN( aFullPNString ) );
    }

    static char getDMSearchPathSeparator()
    {
        return Model::PATH_SEPARATOR;
    }

    AbstractSimulator( ModuleMaker< EcsObject >& maker )
        : Model( maker ) {}

private:
    AbstractSimulator( AbstractSimulator const& );
};

struct CompositeModuleMaker: public ModuleMaker< EcsObject >,
                             public SharedModuleMakerInterface
{
    virtual ~CompositeModuleMaker() {}

    virtual void setSearchPath( const std::string& path )
    {
        SharedModuleMakerInterface* anInterface(
            dynamic_cast< SharedModuleMakerInterface* >( theDefaultModuleMaker.get() ) );
        if ( anInterface )
        {
            anInterface->setSearchPath( path );
        }
    }

    virtual std::string getSearchPath() const
    {
        SharedModuleMakerInterface* anInterface(
            dynamic_cast< SharedModuleMakerInterface* >( theDefaultModuleMaker.get() ) );
        if ( anInterface )
        {
            return anInterface->getSearchPath();
        }

        return "";
    }

    virtual const Module& getModule( const std::string& aClassName, bool forceReload = false )
    {
        ModuleMap::iterator i( theRealModuleMap.find( aClassName ) );
        if ( i == theRealModuleMap.end() )
        {
            const Module& retval( theDefaultModuleMaker->getModule( aClassName, forceReload ) );
            theRealModuleMap[ retval.getModuleName() ] = const_cast< Module* >( &retval );
            return retval;
        }
        return *(*i).second;
    }

    virtual void addClass( Module* dm )
    {
        assert( dm != NULL && dm->getModuleName() != NULL );
        this->theRealModuleMap[ dm->getModuleName() ] = dm;
        ModuleMaker< EcsObject >::addClass( dm );
    }

    virtual const ModuleMap& getModuleMap() const
    {
        return theRealModuleMap;
    }

    CompositeModuleMaker( std::auto_ptr< ModuleMaker< EcsObject > > aDefaultModuleMaker )
        : theDefaultModuleMaker( aDefaultModuleMaker ) {}

private:
    std::auto_ptr< ModuleMaker< EcsObject > > theDefaultModuleMaker;
    ModuleMap theRealModuleMap;
};


class Simulator: public AbstractSimulator
{
private:
    struct DMTypeResolverHelper
    {
        EntityType operator()( py::object aClass )
        {
            if ( !PyType_Check( aClass.ptr() ) )
            {
                return EntityType( EntityType::NONE );
            }

            if (  thePyTypeProcess.get() == reinterpret_cast< PyTypeObject* >( aClass.ptr() ) )
            {
                return EntityType( EntityType::PROCESS );
            }
            else if ( thePyTypeVariable.get() == reinterpret_cast< PyTypeObject* >( aClass.ptr() ) )
            {
                return EntityType( EntityType::VARIABLE );
            }
            else if ( thePyTypeSystem.get() == reinterpret_cast< PyTypeObject* >( aClass.ptr() ) )
            {
                return EntityType( EntityType::SYSTEM );
            }

            py::handle<> aBasesList( PyObject_GetAttrString( aClass.ptr(), const_cast< char* >( "__bases__" ) ) );
            if ( !aBasesList )
            {
                PyErr_Clear();
                return EntityType( EntityType::NONE );
            }

            if ( !PyTuple_Check( aBasesList.get() ) )
            {
                return EntityType( EntityType::NONE );
            }


            for ( Py_ssize_t i( 0 ), ie( PyTuple_GET_SIZE( aBasesList.get() ) ); i < ie; ++i )
            {
                py::object aBase( py::borrowed( PyTuple_GET_ITEM( aBasesList.get(), i ) ) );
                EntityType aResult( ( *this )( aBase ) );
                if ( aResult != EntityType::NONE )
                {
                    return aResult;
                }
            }

            return EntityType( EntityType::NONE );
        }

        DMTypeResolverHelper()
            : thePyTypeProcess( py::objects::registered_class_object( typeid( Process ) ) ),
              thePyTypeVariable( py::objects::registered_class_object( typeid( Variable ) ) ),
              thePyTypeSystem( py::objects::registered_class_object( typeid( System ) ) )
        {
        }

        py::type_handle thePyTypeProcess;
        py::type_handle thePyTypeVariable;
        py::type_handle thePyTypeSystem;
    };            

public:
    Simulator( ModuleMaker< EcsObject >& anEcsObjectMaker )
        : AbstractSimulator( anEcsObjectMaker ),
          theRunningFlag( false ),
          theDirtyFlag( true ),
          theEventCheckInterval( 20 ),
          theEventHandler()
    {
        setup();
    }

    virtual ~Simulator()
    {
    }

    void step( const Integer aNumSteps )
    {
        if( aNumSteps <= 0 )
        {
            THROW_EXCEPTION( Exception,
                             "step( n ): n must be 1 or greater ("
                             + stringCast( aNumSteps ) + " given)" );
        }

        start();

        Integer aCounter( aNumSteps );
        do
        {
            Model::step();
            
            --aCounter;
            
            if( aCounter == 0 )
            {
                stop();
                break;
            }

            if( aCounter % theEventCheckInterval == 0 )
            {
                handleEvent();

                if( ! theRunningFlag )
                {
                    break;
                }
            }
        }
        while( 1 );

    }

    void run()
    {
        start();

        do
        {
            unsigned int aCounter( theEventCheckInterval );
            do
            {
                Model::step();
                --aCounter;
            }
            while( aCounter != 0 );
            
            handleEvent();

        }
        while( theRunningFlag );
    }

    void run( Real aDuration )
    {
        if( aDuration <= 0.0 )
        {
            THROW_EXCEPTION( Exception,
                             "duration must be greater than 0 ("
                             + stringCast( aDuration ) + " given)" );
        }

        start();

        const Real aCurrentTime( getCurrentTime() );
        const Real aStopTime( aCurrentTime + aDuration );

        // setup SystemStepper to step at aStopTime

        //FIXME: dirty, ugly!
        Stepper* aSystemStepper( getSystemStepper() );
        aSystemStepper->setCurrentTime( aCurrentTime );
        aSystemStepper->setNextTime( aStopTime );

        getScheduler().updateEvent( 0, aStopTime );


        if ( theEventHandler )
        {
            while ( theRunningFlag )
            {
                unsigned int aCounter( theEventCheckInterval );
                do 
                {
                    if( getTopEvent().getTime() > aStopTime )
                    {
                        stop();
                        break;
                    }
                    
                    Model::step();

                    --aCounter;
                }
                while( aCounter != 0 );

                handleEvent();
            }
        }
        else
        {
            while ( theRunningFlag )
            {
                if( getTopEvent().getTime() > aStopTime )
                {
                    stop();
                    break;
                }

                Model::step();
            }
        }

    }

    void stop()
    {
        theRunningFlag = false;

        flushLoggers();
    }

    void setEventHandler( py::handle<> const& anEventHandler )
    {
        theEventHandler = anEventHandler;
    }

    void addPythonDM( py::object obj )
    {
        if ( !PyType_Check( obj.ptr() ) )
        {
            PyErr_SetString( PyExc_TypeError, "argument must be a type object" );
            py::throw_error_already_set();
        }

        EntityType aDMType( DMTypeResolverHelper()( obj ) );

        DynamicModule< EcsObject >* aModule( 0 );

        switch ( aDMType.getType() )
        {
        case EntityType::PROCESS:
            aModule = new PythonDynamicModule< PythonProcess >( obj );
            break;

        case EntityType::VARIABLE:
            aModule = new PythonDynamicModule< PythonVariable >( obj );
            break;

        case EntityType::SYSTEM:
            aModule = new PythonDynamicModule< PythonSystem >( obj );
            break;

        default:
            THROW_EXCEPTION( NotImplemented, "not implemented" );
        }
    
        theEcsObjectMaker.addClass( aModule );
    }

protected:

    inline void handleEvent()
    {
        for (;;)
        {
            if ( PyErr_CheckSignals() )
            {
                stop();
                break;
            }

            if ( PyErr_Occurred() )
            {
                stop();
                py::throw_error_already_set();
            }

            if ( !theEventHandler )
            {
                break;
            }

            if ( !PyObject_IsTrue( py::handle<>(
                    PyObject_CallFunction( theEventHandler.get(), NULL ) ).get() ) )
            {
                break;
            }
        }
    }

    void start()
    {
        theRunningFlag = true;
    }

private:

    bool                    theRunningFlag;

    mutable bool            theDirtyFlag;

    Integer         theEventCheckInterval;

    py::handle<>    theEventHandler;
};

struct SimulatorLifecycleSupport
{
    SimulatorLifecycleSupport( ModuleMaker< EcsObject >* anEcsObjectMaker )
        : theEcsObjectMaker( anEcsObjectMaker ) {}

    void operator()( Simulator* aSimulator )
    {
        delete aSimulator;
        delete theEcsObjectMaker;
        theEcsObjectMaker = 0;
    }

    ModuleMaker< EcsObject >* theEcsObjectMaker;
};

static boost::shared_ptr< Simulator > newSimulator()
{
    CompositeModuleMaker* anEcsObjectMaker(
        new CompositeModuleMaker(
            std::auto_ptr< ModuleMaker< EcsObject > >(
                createDefaultModuleMaker() ) ) );
    return boost::shared_ptr< Simulator >(
        new Simulator( *anEcsObjectMaker ),
        SimulatorLifecycleSupport( anEcsObjectMaker ) );
}

inline const char* typeCodeToString( enum PropertySlotBase::Type aTypeCode )
{
    switch ( aTypeCode )
    {
    case PropertySlotBase::POLYMORPH:
        return "polymorph";
    case PropertySlotBase::REAL:
        return "real";
    case PropertySlotBase::INTEGER:
        return "integer";
    case PropertySlotBase::STRING:
        return "string";
    }
    return "???";
}

static std::string PropertyAttributes___str__( PropertyAttributes const* self )
{
    std::string retval;
    retval += "{type=";
    retval += typeCodeToString( self->getType() );
    retval += ", ";
    retval += "settable="
              + stringCast( self->isSetable() )
              + ", ";
    retval += "gettable="
              + stringCast( self->isGetable() )
              + ", ";
    retval += "loadable="
              + stringCast( self->isLoadable() )
              + ", ";
    retval += "savable="
              + stringCast( self->isSavable() )
              + ", ";
    retval += "dynamic="
              + stringCast( self->isDynamic() )
              + "}";
    return retval;
}

static int PropertyAttributes___getitem__( PropertyAttributes const* self, int idx )
{
    switch ( idx )
    {
    case 0:
        return self->isSetable();
    case 1:
        return self->isGetable();
    case 2:
        return self->isLoadable();
    case 3:
        return self->isSavable();
    case 4:
        return self->isDynamic();
    case 5:
        return self->getType();
    }

    throw std::range_error("Index out of bounds");
}

static PyObject* PropertyAttributes___iter__( PropertyAttributes const* self )
{
    return reinterpret_cast< PyObject* >( PropertyAttributesIterator::create( *self ) );
}

PyObject* PropertyAttributesIterator::__next__( PropertyAttributesIterator* self )
{
    try
    {
        return py::incref( py::object( PropertyAttributes___getitem__( &self->theImpl, self->theIdx++ ) ).ptr() );
    }
    catch ( std::range_error const& )
    {
        PyErr_SetNone( PyExc_StopIteration );
    }
    return 0;
}


static py::object LoggerPolicy_GetItem( Logger::Policy const* self, int idx )
{
    switch ( idx )
    {
    case 0:
        return py::object( self->getMinimumStep() );
    case 1:
        return py::object( self->getMinimumTimeInterval() );
    case 2:
        return py::object( self->doesContinueOnError() );
    case 3:
        return py::object( self->getMaxSpace() );
    }

    throw std::range_error("Index out of bounds");
}

static py::object Process_get_variableReferences( Process* self )
{
    return py::object( VariableReferences( self ) );
}

template< typename TecsObject_ >
static Polymorph EcsObject___getattr__( TecsObject_* self, std::string key )
try
{
    if ( key == "__members__" || key == "__methods__" )
    {
        PyErr_SetString( PyExc_KeyError, key.c_str() );
        py::throw_error_already_set();
    }

    return self->getProperty( key );
}
catch ( NoSlot const& anException )
{
    PyErr_SetString( PyExc_AttributeError, anException.what() );
    py::throw_error_already_set();
    return Polymorph();
}

template< typename TecsObject_ >
static void EcsObject___setattr__( py::back_reference< TecsObject_* > aSelf, py::object key, py::object value )
try
{
    py::handle<> aDescr( py::allow_null( PyObject_GetAttr( reinterpret_cast< PyObject* >( aSelf.source().ptr()->ob_type ), key.ptr() ) ) );
    if ( !aDescr || !( aDescr->ob_type->tp_flags & Py_TPFLAGS_HAVE_CLASS ) || !aDescr.get()->ob_type->tp_descr_set )
    {
        PyErr_Clear();
        EcsObject* self = aSelf.get();
        std::string keyStr = py::extract< std::string >( key );
        self->setProperty( keyStr, py::extract< Polymorph >( value ) );
    }
    else
    {
        aDescr.get()->ob_type->tp_descr_set( aDescr.get(), aSelf.source().ptr(), value.ptr() );
        if (PyErr_Occurred())
        {
            py::throw_error_already_set();
        }
    }
}
catch ( NoSlot const& anException )
{
    PyErr_SetString( PyExc_AttributeError, anException.what() );
    py::throw_error_already_set();
}

template< typename T_ >
static PyObject* writeOnly( T_* )
{
    PyErr_SetString( PyExc_AttributeError, "Write-only attributes." );
    return py::incref( Py_None );
}

static void setWarningHandler( py::handle<> const& handler )
{
    static PythonWarningHandler thehandler;
    thehandler = PythonWarningHandler( handler );
    libecs::setWarningHandler( &thehandler );
}

struct return_entity : public py::default_call_policies
{
    struct result_converter
    {
        template< typename T_ >
        struct apply
        {
            struct type
            {
                PyTypeObject const *get_pytype() const
				{
					return 0;
				}

                PyObject* operator()( Entity* ptr ) const
                {
                    if ( ptr == 0 )
                        return py::detail::none();
                    else
                        return ( *this )( *ptr );
                }
                
                PyObject* operator()( Entity const& x ) const
                {
                    Entity* const ptr( const_cast< Entity* >( &x ) );
                    PyObject* aRetval( py::detail::wrapper_base_::owner( ptr ) );
                    if ( !aRetval )
                    {
                        if ( Process* tmp = dynamic_cast< Process* >( ptr ) )
                        {
                            aRetval = py::detail::make_reference_holder::execute( tmp );
                        }
                        else if ( Variable* tmp = dynamic_cast< Variable* >( ptr ) )
                        {
                            aRetval = py::detail::make_reference_holder::execute( tmp );
                        }
                        else if ( System* tmp = dynamic_cast< System* >( ptr ) )
                        {
                            aRetval = py::detail::make_reference_holder::execute( tmp );
                        }
                        else
                        {
                            aRetval = py::detail::make_reference_holder::execute( ptr );
                        }
                    }
                    else
                    {
                        py::incref( aRetval );
                    }
                    return aRetval;
                }
            };
        };
    };
};

AbstractSimulator* Entity_getModel(Entity const& entity)
{
    return dynamic_cast<AbstractSimulator*>(entity.getModel());
}

BOOST_PYTHON_MODULE( _ecs )
{
    DataPointVectorWrapper< DataPoint >::__class_init__();
    DataPointVectorWrapper< LongDataPoint >::__class_init__();
    STLIteratorWrapper< Process::VariableReferenceVector::const_iterator >::__class_init__();

    // without this it crashes when Logger::getData() is called. why?
    import_array();

    registerTupleConverters< std::pair< Real, String > >();
    PolymorphToPythonConverter::addToRegistry();
    StringVectorToPythonConverter::addToRegistry();
    PropertySlotMapToPythonConverter::addToRegistry();
    DataPointVectorSharedPtrConverter::addToRegistry();
    FullIDToPythonConverter::addToRegistry();
    PythonToFullIDConverter::addToRegistry();

    PolymorphRetriever::addToRegistry();

    // functions
    py::register_exception_translator< Exception >( &translateException );
    py::register_exception_translator< std::exception >( &translateException );
    py::register_exception_translator< std::range_error >( &translateRangeError );
    PythonVariable::addToRegistry();
    PythonProcess::addToRegistry();
    PythonSystem::addToRegistry();

    py::def( "getLibECSVersionInfo", &getLibECSVersionInfo );
    py::def( "getLibECSVersion",     &getVersion );
    py::def( "setWarningHandler",    &::setWarningHandler );

    typedef py::return_value_policy< py::reference_existing_object >
            return_existing_object;
    typedef py::return_value_policy< py::copy_const_reference >
            return_copy_const_reference;

    py::class_< PropertyAttributes >( "PropertyAttributes",
        py::init< enum PropertySlotBase::Type, bool, bool, bool, bool, bool >() )
        .add_property( "type", &PropertyAttributes::getType )
        .add_property( "setable", &PropertyAttributes::isSetable )
        .add_property( "getable", &PropertyAttributes::isGetable )
        .add_property( "loadable", &PropertyAttributes::isLoadable )
        .add_property( "savable", &PropertyAttributes::isSavable )
        .add_property( "dynamic", &PropertyAttributes::isDynamic )
        .def( "__str__", &PropertyAttributes___str__ )
        .def( "__getitem__", &PropertyAttributes___getitem__ )
        .def( "__iter__", &PropertyAttributes___iter__ )
        ;

    py::enum_< PropertySlotBase::Type >( "PropertyType" )
        .value( "POLYMORPH", PropertySlotBase::POLYMORPH )
        .value( "REAL", PropertySlotBase::REAL )
        .value( "INTEGER", PropertySlotBase::INTEGER )
        .value( "STRING", PropertySlotBase::STRING )
        ;

    py::class_< Logger::Policy >( "LoggerPolicy", py::init<>() )
        .add_property( "minimumStep", &Logger::Policy::getMinimumStep,
                                      &Logger::Policy::setMinimumStep )
        .add_property( "minimumTimeInterval",
                       &Logger::Policy::getMinimumTimeInterval,
                       &Logger::Policy::setMinimumTimeInterval )
        .add_property( "continueOnError",
                       &Logger::Policy::doesContinueOnError,
                       &Logger::Policy::setContinueOnError )
        .add_property( "maxSpace",
                       &Logger::Policy::getMaxSpace,
                       &Logger::Policy::setMaxSpace )
        .def( "__getitem__", &LoggerPolicy_GetItem )
        ;

    py::class_< VariableReferences >( "VariableReferences", py::no_init )
        .add_property( "positiveReferences",
                       &VariableReferences::getPositivesReferences )
        .add_property( "zeroReferences",
                       &VariableReferences::getZeroReferences )
        .add_property( "negativeReferences",
                       &VariableReferences::getNegativeReferences )
        .def( "add",
              ( Integer ( VariableReferences::* )( String const&, String const&, Integer const&, bool ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( String const&, String const&, Integer const& ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( String const&, Integer const&, bool ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( String const&, Integer const& ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( String const&, Variable*, Integer const&, bool ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( String const&, Variable*, Integer const& ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( Variable*, Integer const&, bool ) )
              &VariableReferences::add )
        .def( "add",
              ( Integer ( VariableReferences::* )( Variable*, Integer const& ) )
              &VariableReferences::add )
        .def( "remove", ( void ( VariableReferences::* )( String const& ) )
              &VariableReferences::remove )
        .def( "remove", ( void ( VariableReferences::* )( Integer const ) )
              &VariableReferences::remove )
        .def( "__getitem__", &VariableReferences::__getitem__,
              return_copy_const_reference() )
        .def( "__len__", &VariableReferences::__len__ )
        .def( "__iter__", &VariableReferences::__iter__ )
        .def( "__str__", &VariableReferences::__str__ )
        ;

    py::class_< VariableReference >( "VariableReference", py::no_init )
        // properties
        .add_property( "coefficient", &VariableReference::getCoefficient )
        .add_property( "serial",      &VariableReference::getSerial )
        .add_property( "name",        &VariableReference::getName )
        .add_property( "isAccessor",  &VariableReference::isAccessor )
        .add_property( "FullID",
                py::make_function(
                    &VariableReference::getFullID,
                    return_copy_const_reference() ) )
        .add_property( "variable",
                py::make_function(
                    &VariableReference::getVariable,
                    return_existing_object() ) )
        .def( "__str__", &VariableReference___str__ )
        ;

    py::class_< Stepper, py::bases<>, Stepper, boost::noncopyable >
        ( "Stepper", py::no_init )
        .add_property( "id", &Stepper::getID, &Stepper::setID )
        .add_property( "Priority",
                       &Stepper::getPriority,
                       &Stepper::setPriority )
        .add_property( "StepInterval",
                       &Stepper::getStepInterval, 
                       &Stepper::setStepInterval )
        .add_property( "MaxStepInterval",
                       &Stepper::getMaxStepInterval,
                       &Stepper::setMaxStepInterval )
        .add_property( "MinStepInterval",
                       &Stepper::getMinStepInterval,
                       &Stepper::setMinStepInterval )
        .add_property( "RngSeed", &writeOnly<Stepper>, &Stepper::setRngSeed )
        .def( "__setattr__", &EcsObject___setattr__< Stepper > )
        .def( "__getattr__", &EcsObject___getattr__< Stepper > )
        ;

    py::class_< Entity, py::bases<>, Entity, boost::noncopyable >
        ( "Entity", py::no_init )
        // properties
        .add_property( "model",
            py::make_function( &Entity_getModel,
            return_existing_object() ) )
        .add_property( "simulator",
            py::make_function( &Entity_getModel,
            return_existing_object() ) )
        .add_property( "superSystem",
            py::make_function( &Entity::getSuperSystem,
            return_existing_object() ) )
        .add_property( "ID", &Entity::getID, &Entity::setID )
        .add_property( "FullID", &Entity::getFullID )
        .add_property( "Name", &Entity::getName, &Entity::setName )
        .def( "getSuperSystem", &Entity::getSuperSystem, return_existing_object() )
        .def( "__setattr__", &EcsObject___setattr__< Entity > )
        .def( "__getattr__", &EcsObject___getattr__< Entity > )
        ;

    py::class_< System, py::bases< Entity >, System, boost::noncopyable>
        ( "System", py::no_init )
        // properties
        .add_property( "Size",        &System::getSize )
        .add_property( "SizeN_A",     &System::getSizeN_A )
        .add_property( "StepperID",   &System::getStepperID, &System::setStepperID )
        .def( "registerEntity",       ( void( System::* )( Entity* ) )&System::registerEntity )
        ;

    py::class_< Process, py::bases< Entity >, Process, boost::noncopyable >
        ( "Process", py::no_init )
        .add_property( "Activity",  &Process::getActivity,
                                    &Process::setActivity )
        .add_property( "IsContinuous", &Process::isContinuous )
        .add_property( "Priority",  &Process::getPriority,
                                    &Process::setPriority )
        .add_property( "StepperID", &Process::getStepperID,
                                    &Process::setStepperID )
        .add_property( "variableReferences",
              py::make_function( &Process_get_variableReferences ) )
        ;

    py::class_< Variable, py::bases< Entity >, Variable, boost::noncopyable >
        ( "Variable", py::no_init )
        .add_property( "Value",  &Variable::getValue,
                                 &Variable::setValue )
        .add_property( "MolarConc",  &Variable::getMolarConc,
                                     &Variable::setMolarConc  )
        .add_property( "NumberConc", &Variable::getNumberConc,
                                     &Variable::setNumberConc )
        ;

    py::class_< Logger, py::bases<>, Logger, boost::noncopyable >( "Logger", py::no_init )
        .add_property( "StartTime", &Logger::getStartTime )
        .add_property( "EndTime", &Logger::getEndTime )
        .add_property( "Size", &Logger::getSize )
        .add_property( "Policy",
            py::make_function(
                &Logger::getLoggerPolicy,
                return_copy_const_reference() ) )
        .def( "getData", 
              ( boost::shared_ptr< DataPointVector >( Logger::* )( void ) const )
              &Logger::getData )
        .def( "getData", 
              ( boost::shared_ptr< DataPointVector >( Logger::* )(
                Real, Real ) const )
              &Logger::getData )
        .def( "getData",
              ( boost::shared_ptr< DataPointVector >( Logger::* )(
                     Real, Real, Real ) const )
              &Logger::getData )
        ;

    py::class_< AbstractSimulator, py::bases<>, boost::shared_ptr< AbstractSimulator >, boost::noncopyable  >( "AbstractSimulator", py::no_init )
        .add_static_property( "DM_SEARCH_PATH_SEPARATOR",
              &AbstractSimulator::getDMSearchPathSeparator )
        .add_property( "rootSystem",
              py::make_function(
                    &AbstractSimulator::getRootSystem,
                  py::return_value_policy< py::reference_existing_object >() ) )
        .add_property( "rootSystem",
              py::make_function(
                    static_cast< System*( AbstractSimulator::* )() const >(
                        &AbstractSimulator::getRootSystem ),
                  py::return_value_policy< py::reference_existing_object >() ) )
        .def( "getClassInfo",
              &AbstractSimulator::getClassInfo )
        // Stepper-related methods
        .def( "createStepper",
              (Stepper*(AbstractSimulator::*)(String const&, String const&))
              &AbstractSimulator::createStepper,
              py::return_value_policy< py::reference_existing_object >() )
        .def( "getStepper",
              &AbstractSimulator::getStepper,
              py::return_value_policy< py::reference_existing_object >() )
        .def( "deleteStepper",
              &AbstractSimulator::deleteStepper )
        .def( "deleteEntity",
              &AbstractSimulator::deleteEntity )
        .def( "getStepperList",
              &AbstractSimulator::getStepperList )
        .def( "getStepperPropertyList",
              &AbstractSimulator::getStepperPropertyList )
        .def( "getStepperPropertyAttributes", 
              &AbstractSimulator::getStepperPropertyAttributes )
        .def( "setStepperProperty",
              &AbstractSimulator::setStepperProperty )
        .def( "getStepperProperty",
              &AbstractSimulator::getStepperProperty )
        .def( "loadStepperProperty",
              &AbstractSimulator::loadStepperProperty )
        .def( "saveStepperProperty",
              &AbstractSimulator::saveStepperProperty )
        .def( "getStepperClassName",
              &AbstractSimulator::getStepperClassName )

        // Entity-related methods
        .def( "createEntity",
              &AbstractSimulator::createEntity,
              return_entity() )
        .def( "createVariable",
              &AbstractSimulator::createVariable,
              py::return_value_policy< py::reference_existing_object >() )
        .def( "createProcess",
              &AbstractSimulator::createProcess,
              py::return_value_policy< py::reference_existing_object >() )
        .def( "createSystem",
              &AbstractSimulator::createSystem,
              py::return_value_policy< py::reference_existing_object >() )
        .def( "getEntity",
              &AbstractSimulator::getEntity,
              return_entity() )
        .def( "deleteEntity",
              &AbstractSimulator::deleteEntity )
        .def( "getEntityList",
              &AbstractSimulator::getEntityList )
        .def( "entityExists",
              &AbstractSimulator::entityExists )
        .def( "getEntityPropertyList",
              &AbstractSimulator::getEntityPropertyList )
        .def( "setEntityProperty",
              &AbstractSimulator::setEntityProperty )
        .def( "getEntityProperty",
              &AbstractSimulator::getEntityProperty )
        .def( "loadEntityProperty",
              &AbstractSimulator::loadEntityProperty )
        .def( "saveEntityProperty",
              &AbstractSimulator::saveEntityProperty )
        .def( "getEntityPropertyAttributes", 
              &AbstractSimulator::getEntityPropertyAttributes )
        .def( "getEntityClassName",
              &AbstractSimulator::getEntityClassName )

        // Logger-related methods
        .def( "getLoggerList",
                    &AbstractSimulator::getLoggerList )    
        .def( "createLogger",
              ( Logger* ( AbstractSimulator::* )( String const& ) )
                    &AbstractSimulator::createLogger,
              py::return_internal_reference<> () )
        .def( "createLogger",                                 
              ( Logger* ( AbstractSimulator::* )( String const&, Logger::Policy const& ) )
              &AbstractSimulator::createLogger,
              py::return_internal_reference<>() )
        .def( "createLogger",                                 
              ( Logger* ( AbstractSimulator::* )( String const&, py::object ) )
                    &AbstractSimulator::createLogger,
              py::return_internal_reference<> () )
        .def( "getLogger", &AbstractSimulator::getLogger,
              py::return_internal_reference<>() )
        .def( "removeLogger", &AbstractSimulator::removeLogger )
        .def( "getLoggerData", 
              ( boost::shared_ptr< DataPointVector >( AbstractSimulator::* )(
                    String const& ) const )
              &AbstractSimulator::getLoggerData )
        .def( "getLoggerData", 
              ( boost::shared_ptr< DataPointVector >( AbstractSimulator::* )(
                    String const&, Real const&, Real const& ) const )
              &AbstractSimulator::getLoggerData )
        .def( "getLoggerData",
              ( boost::shared_ptr< DataPointVector >( AbstractSimulator::* )(
                     String const&, Real const&, 
                     Real const&, Real const& ) const )
              &AbstractSimulator::getLoggerData )
        .def( "getLoggerStartTime",
              &AbstractSimulator::getLoggerStartTime )    
        .def( "getLoggerEndTime",
              &AbstractSimulator::getLoggerEndTime )        
        .def( "getLoggerPolicy",
              &AbstractSimulator::getLoggerPolicy )
        .def( "setLoggerPolicy",
              ( void (AbstractSimulator::*)(
                    String const&, Logger::Policy const& ) )
              &AbstractSimulator::setLoggerPolicy )
        .def( "setLoggerPolicy",
              ( void (AbstractSimulator::*)(
                    String const& aFullPNString,
                    py::object aParamList ) )
              &AbstractSimulator::setLoggerPolicy )
        .def( "getLoggerSize",
              &AbstractSimulator::getLoggerSize )

        // Simulation-related methods
        .def( "initialize",
              &AbstractSimulator::initialize )
        .def( "getCurrentTime",
              &AbstractSimulator::getCurrentTime )
        .def( "getNextEvent",
              &AbstractSimulator::getNextEvent )

        // DM inspection methods
        .def( "getPropertyInfo",
              &AbstractSimulator::getPropertyInfo,
              return_copy_const_reference() )
        .def( "getDMInfo",
              &AbstractSimulator::getDMInfo )
        .def( "markDirty",
              &AbstractSimulator::markDirty )
        ;

    // Simulator class
    py::class_< Simulator, py::bases< AbstractSimulator >, boost::shared_ptr< Simulator >, boost::noncopyable >( "Simulator", py::no_init )
        .def( "__init__", py::make_constructor( newSimulator ) )
        .def( "stop",
              &Simulator::stop )
        .def( "step",
              ( void ( Simulator::* )( Integer ) )
              &Simulator::step )
        .def( "run",
              ( void ( Simulator::* )() )
              &Simulator::run )
        .def( "run",
              ( void ( Simulator::* )( Real ) ) 
              &Simulator::run )
        .def( "setEventHandler",
              &Simulator::setEventHandler )
        .def( "setDMSearchPath", &Simulator::setDMSearchPath )
        .def( "getDMSearchPath", &Simulator::getDMSearchPath )
        .def( "addPythonDM", &Simulator::addPythonDM )
        ;
}
