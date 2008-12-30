//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
#include <boost/cast.hpp>
#include <boost/format.hpp>
#include <boost/format/group.hpp>
#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/object.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <numpy/arrayobject.h>
#include <stringobject.h>

#include "libecs/Model.hpp"
#include "libecs/libecs.hpp"
#include "libecs/Process.hpp"
#include "libecs/Exceptions.hpp"
#include "libecs/Polymorph.hpp"
#include "libecs/DataPointVector.hpp"
#include "libecs/VariableReference.hpp"

using namespace libecs;
namespace py = boost::python;

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
            return PyFloat_AS_DOUBLE( aPyObjectPtr );
        }
        else if( PyInt_Check( aPyObjectPtr ) )
        {
            return PyInt_AS_LONG( aPyObjectPtr );
        }
        else if( PyString_Check( aPyObjectPtr ) )
        {
            return Polymorph( PyString_AsString( aPyObjectPtr ) );
        }
        else if( PyUnicode_Check( aPyObjectPtr ) )
        {
            aPyObjectPtr = PyUnicode_AsEncodedString( aPyObjectPtr, NULL, NULL );
            if ( aPyObjectPtr )
                return Polymorph( PyString_AsString( aPyObjectPtr ) );
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
            PySequence_GetItem( theSeq, theIdx ) );
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
struct TupleToPythonConverter<std::pair<Tfirst_, Tsecond_> >
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

template< typename T_ >
struct StringKeyedMapToPythonConverter
{
    static void addToRegistry()
    {
        py::to_python_converter< T_, StringKeyedMapToPythonConverter>();
    }

    static PyObject* convert( T_ const& aStringKeyedMap )
    {
        PyObject* aPyDict( PyDict_New() );
        for ( typename T_::const_iterator i( aStringKeyedMap.begin() );
              i != aStringKeyedMap.end(); ++i )
        {
            PyDict_SetItem( aPyDict, PyString_FromStringAndSize(
                i->first.data(), i->first.size() ),
                py::incref( py::object( i->second ).ptr() ) );
                                            
        }
        return aPyDict;
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

class PythonCallable
{
public:
    PythonCallable( PyObject* aPyObjectPtr )
        : thePyObject( py::incref( aPyObjectPtr ) )
    {
        // this check isn't needed actually, because BPL does this automatically
        if ( !PyCallable_Check( thePyObject.get() ) )
        {
            PyErr_SetString( PyExc_TypeError,
                             "the argument is not a callable object" );
            py::throw_error_already_set();
        }
    }

    ~PythonCallable()
    {
        ; // do nothing
    }

protected:
    py::handle<> thePyObject;
};


class PythonEventHandler: public PythonCallable
{
public:
    PythonEventHandler( PyObject* aPyObjectPtr )
        : PythonCallable( aPyObjectPtr )
    {
        ; // do nothing
    }
        
    ~PythonEventHandler() {}

    bool operator()( void ) const
    {
        PyObject* aPyObjectPtr( PyObject_CallFunction( thePyObject.get(), NULL  ) );
        const bool aResult( PyObject_IsTrue( aPyObjectPtr ) );
        Py_XDECREF( aPyObjectPtr );

        return aResult;
    }
};


template< typename Tcallable_ >
struct CallableSharedPtrRetriever
{
public:
    typedef Tcallable_ Callable;

public:

    static void addToRegistry()
    {
        py::converter::registry::insert(
                &convertible, &construct,
                py::type_id< boost::shared_ptr< Callable > >() );
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
               py::converter::rvalue_from_python_stage1_data* data )
    {
        void* storage( reinterpret_cast<
            py::converter::rvalue_from_python_storage<boost::shared_ptr< Callable > >* >(
                data )->storage.bytes );

        data->convertible = new (storage) boost::shared_ptr< Callable >(
            new Callable( aPyObjectPtr ) );
    }

};


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
            self->~Iterator();
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
    DataPointVectorSharedPtr theVector;

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

    DataPointVectorWrapper( DataPointVectorSharedPtr const& aVector )
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

    static DataPointVectorWrapper* create( DataPointVectorSharedPtr const& aVector )
    {
        return new DataPointVectorWrapper( aVector ); 
    }

    static void __dealloc__( DataPointVectorWrapper* self )
    {
        self->~DataPointVectorWrapper();
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
	0					/* tp_members */
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
	PyObject_Free,			/* tp_free */
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
    { "__array_struct__", (getter)&DataPointVectorWrapper::__get___array__struct, NULL },
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
    static STLIteratorWrapper* create( Trange_ const& range )
    {
        return new STLIteratorWrapper( range );
    }

    static void __dealloc__( STLIteratorWrapper* self )
    {
        self->~STLIteratorWrapper();
    }

    static PyObject* __next__( STLIteratorWrapper* self )
    {
        if ( self->theIdx == self->theEnd )
            return NULL;

        return py::incref( py::object( *self->theIdx ).ptr() );
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
	0					/* tp_members */
};


static std::string VariableReference___repr__( VariableReference const* self )
{
    return std::string( "[" ) + self->getName() + ": "
            + "coefficient=" + stringCast( self->getCoefficient() ) + ", "
            + "variable=" + self->getVariable()->getFullID().asString() + ", "
            + "accessor=" + ( self->isAccessor() ? "true": "false" ) + "]";
}


class VariableReferences
{
public:
    VariableReferences( Process* proc ): theProc( proc ) {}

    void add( String const& name, String const& fullID, Integer const& coef,
              bool isAccessor )
    {
        theProc->registerVariableReference( name, FullID( fullID ),
                                            coef, isAccessor );
    }

    void add( String const& name, String const& fullID, Integer const& coef )
    {
        theProc->registerVariableReference( name, FullID( fullID ),
                                            coef, false );
    }


    void remove( String const& name )
    {
        theProc->removeVariableReference( name );
    }

    VariableReference const& __getitem__( py::object name )
    {
        if ( PyInt_Check( name.ptr() ) )
        {
            long idx( PyInt_AS_LONG( name.ptr() ) );
            VariableReferenceVector const& refs(
                    theProc->getVariableReferenceVector() );
            if ( idx < 0
                 || static_cast< VariableReferenceVector::size_type >( idx )
                    >= refs.size() )
            {
                throw std::range_error( "Index out of bounds");
            }
            return refs[ idx ];
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
        VariableReferenceVector const& refs(
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
        VariableReferenceVector const& refs(
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
        VariableReferenceVector const& refs(
                theProc->getVariableReferenceVector() );

        py::list retval;
        std::for_each(
            refs.begin() + theProc->getZeroVariableReferenceOffset(),
            refs.begin() + theProc->getPositiveVariableReferenceOffset(),
            boost::bind( &py::list::append< VariableReference >, &retval,
                         _1 ) );
        return retval;
    }

    py::object __iter__()
    {
        return py::object( STLIteratorWrapper< VariableReferenceVector::const_iterator >( theProc->getVariableReferenceVector() ) );
    }

    std::string __repr__()
    {
        VariableReferenceVector const& refs(
                theProc->getVariableReferenceVector() );

        std::string retval;

        retval += '[';
        for ( VariableReferenceVector::const_iterator b( refs.begin() ),
                                                      i( b ),
                                                      e( refs.end() );
                i != e; ++i )
        {
            if ( i != b )
                retval += ", ";
            retval += VariableReference___repr__( &*i );
            retval += ']';
        }

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
        py::to_python_converter< DataPointVectorSharedPtr,
                                 DataPointVectorSharedPtrConverter >();
    }

    static PyObject* 
    convert( DataPointVectorSharedPtr const& aVectorSharedPtr )
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

template< typename T_ >
inline PyObject* to_python_indirect_fun( T_ arg )
{
    return py::to_python_indirect< T_, py::detail::make_reference_holder >()( arg );
}

template< typename TeventHander_ >
class Simulator
{
public:
    typedef TeventHander_ EventHandler;

public:
    Simulator()
        : theRunningFlag( false ),
          theDirtyFlag( true ),
          theEventCheckInterval( 20 ),
          theEventHandler(),
          thePropertiedObjectMaker( createDefaultModuleMaker() ),
          theModel( *thePropertiedObjectMaker )
    {
    }

    ~Simulator()
    {
        delete thePropertiedObjectMaker;
    }

    void initialize()
    {
        theModel.initialize();
    }

    Stepper* createStepper( String const& aClassname,
                            String const& anId )
    {
        if( theRunningFlag )
        {
            THROW_EXCEPTION( Exception, 
                             "Cannot create a Stepper during simulation." );
        }

        return theModel.createStepper( aClassname, anId );
    }

    Stepper* getStepper( String const& anId )
    {
        return theModel.getStepper( anId );
    }

    void deleteStepper( String const& anID )
    {
        theModel.deleteStepper( anID );
    }

    const Polymorph getStepperList() const
    {
        Model::StepperMap const& aStepperMap( theModel.getStepperMap() );

        PolymorphVector aPolymorphVector; 
        aPolymorphVector.reserve( aStepperMap.size() );
        
        for( Model::StepperMap::const_iterator i( aStepperMap.begin() );
             i != aStepperMap.end(); ++i )
        {
            aPolymorphVector.push_back( (*i).first );
        }

        return aPolymorphVector;
    }

    const Polymorph 
    getStepperPropertyList( String const& aStepperID ) const
    {
        StepperPtr aStepperPtr( theModel.getStepper( aStepperID ) );
        
        return aStepperPtr->getPropertyList();
    }

    PropertyAttributes
    getStepperPropertyAttributes( String const& aStepperID, 
                                  String const& aPropertyName ) const
    {
        StepperPtr aStepperPtr( theModel.getStepper( aStepperID ) );
        return aStepperPtr->getPropertyAttributes( aPropertyName );
    }

    void setStepperProperty( String const& aStepperID,
                             String const& aPropertyName,
                             Polymorph const& aValue )
    {
        StepperPtr aStepperPtr( theModel.getStepper( aStepperID ) );
        
        aStepperPtr->setProperty( aPropertyName, aValue );
    }

    const Polymorph
    getStepperProperty( String const& aStepperID,
                        String const& aPropertyName ) const
    {
        Stepper const * aStepperPtr( theModel.getStepper( aStepperID ) );

        return aStepperPtr->getProperty( aPropertyName );
    }

    void loadStepperProperty( String const& aStepperID,
                              String const& aPropertyName,
                              Polymorph const& aValue )
    {
        StepperPtr aStepperPtr( theModel.getStepper( aStepperID ) );
        
        aStepperPtr->loadProperty( aPropertyName, aValue );
    }

    const Polymorph
    saveStepperProperty( String const& aStepperID,
                         String const& aPropertyName ) const
    {
        Stepper const * aStepperPtr( theModel.getStepper( aStepperID ) );

        return aStepperPtr->saveProperty( aPropertyName );
    }

    const String
    getStepperClassName( String const& aStepperID ) const
    {
        Stepper const * aStepperPtr( theModel.getStepper( aStepperID ) );

        return aStepperPtr->getClassName();
    }

    const PolymorphMap getClassInfo( String const& aClassname ) const
    {
        libecs::PolymorphMap aBuiltInfoMap;
        for ( DynamicModuleInfo::EntryIterator* anInfo(
              theModel.getPropertyInterface( aClassname ).getInfoFields() );
              anInfo->next(); )
        {
            aBuiltInfoMap.insert( std::make_pair( anInfo->current().first,
                                  *reinterpret_cast< const libecs::Polymorph* >(
                                    anInfo->current().second ) ) );
        }
        return aBuiltInfoMap;
    }

    
    PyObject* createEntity( String const& aClassname, 
                          String const& aFullIDString )
    {
        if( theRunningFlag )
        {
            THROW_EXCEPTION( Exception, 
                             "Cannot create an Entity during simulation." );
        }

        PyObject* retval( 0 );
        Entity* ent( theModel.createEntity( aClassname, FullID( aFullIDString ) ) );

        switch ( static_cast< enum EntityType::Type >( ent->getEntityType() ) )
        {
        case EntityType::VARIABLE:
            retval = to_python_indirect_fun( static_cast< Variable* >( ent ) );
            break;
        case EntityType::PROCESS:
            retval = to_python_indirect_fun( static_cast< Process* >( ent ) );
            break;
        case EntityType::SYSTEM:
            retval = to_python_indirect_fun( static_cast< System* >( ent ) );
            break;
        default:
            retval = py::incref( Py_None );
        }

        return retval;
    }

    PyObject* getEntity( String const& aFullIDString )
    {
        PyObject* retval( 0 );
        Entity* ent( theModel.getEntity( FullID( aFullIDString ) ) );
        switch ( static_cast< enum EntityType::Type >( ent->getEntityType() ) )
        {
        case EntityType::VARIABLE:
            retval = to_python_indirect_fun( static_cast< Variable* >( ent ) );
            break;
        case EntityType::PROCESS:
            retval = to_python_indirect_fun( static_cast< Process* >( ent ) );
            break;
        case EntityType::SYSTEM:
            retval = to_python_indirect_fun( static_cast< System* >( ent ) );
            break;
        default:
            return py::incref( Py_None );
        }

        return retval;
    }


    void deleteEntity( String const& aFullIDString )
    {
        THROW_EXCEPTION( NotImplemented,
                         "deleteEntity() method is not supported yet." );
    }

    const Polymorph 
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
            return aVector;
        }

        System const* aSystemPtr( theModel.getSystem( aSystemPath ) );

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

    const Polymorph 
    getEntityPropertyList( String const& aFullIDString ) const
    {
        Entity const * anEntityPtr( theModel.getEntity( FullID( aFullIDString ) ) );

        return anEntityPtr->getPropertyList();
    }

    const bool entityExists( String const& aFullIDString ) const
    {
        try
        {
            IGNORE_RETURN theModel.getEntity( FullID( aFullIDString ) );
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
        EntityPtr anEntityPtr( theModel.getEntity( aFullPN.getFullID() ) );

        anEntityPtr->setProperty( aFullPN.getPropertyName(), aValue );
    }

    const Polymorph
    getEntityProperty( String const& aFullPNString ) const
    {
        FullPN aFullPN( aFullPNString );
        Entity const * anEntityPtr( theModel.getEntity( aFullPN.getFullID() ) );
                
        return anEntityPtr->getProperty( aFullPN.getPropertyName() );
    }

    void loadEntityProperty( String const& aFullPNString,
                             Polymorph const& aValue )
    {
        FullPN aFullPN( aFullPNString );
        EntityPtr anEntityPtr( theModel.getEntity( aFullPN.getFullID() ) );

        anEntityPtr->loadProperty( aFullPN.getPropertyName(), aValue );
    }

    const Polymorph
    saveEntityProperty( String const& aFullPNString ) const
    {
        FullPN aFullPN( aFullPNString );
        Entity const * anEntityPtr( theModel.getEntity( aFullPN.getFullID() ) );

        return anEntityPtr->saveProperty( aFullPN.getPropertyName() );
    }

    PropertyAttributes
    getEntityPropertyAttributes( String const& aFullPNString ) const
    {
        FullPN aFullPN( aFullPNString );
        Entity const * anEntityPtr( theModel.getEntity( aFullPN.getFullID() ) );

        return anEntityPtr->getPropertyAttributes( aFullPN.getPropertyName() );
    }

    const String
    getEntityClassName( String const& aFullIDString ) const
    {
        FullID aFullID( aFullIDString );
        Entity const * anEntityPtr( theModel.getEntity( aFullID ) );

        return anEntityPtr->getClassName();
    }

    Logger* createLogger( String const& aFullPNString )
    {
        return createLogger( aFullPNString, Logger::Policy() );
    }

    Logger* createLogger( String const& aFullPNString,
                          Logger::Policy const& aParamList )
    {
        if( theRunningFlag )
        {
            THROW_EXCEPTION( Exception, 
                             "Cannot create a Logger during simulation." );
        }

        Logger* retval( theModel.getLoggerBroker().createLogger(
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
                             "second argument must be a tuple of 4 items.");
        }

        return createLogger( aFullPNString,
                Logger::Policy(
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 0 ] ).ptr() ),
                    PyFloat_AsDouble( static_cast< py::object >( aParamList[ 1 ] ).ptr() ),
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 2 ] ).ptr() ),
                    PyInt_AsLong( static_cast< py::object >( aParamList[ 3 ] ).ptr() ) ) );
    }

    const Polymorph getLoggerList() const
    {
        PolymorphVector aLoggerList;

        LoggerBroker const& aLoggerBroker( theModel.getLoggerBroker() );

        for( LoggerBroker::const_iterator
                i( aLoggerBroker.begin() ), end( aLoggerBroker.end() );
             i != end; ++i )
        {
            aLoggerList.push_back( (*i).first.asString() );
        }

        return aLoggerList;
    }

    DataPointVectorSharedPtr 
    getLoggerData( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getData();
    }

    DataPointVectorSharedPtr
    getLoggerData( String const& aFullPNString, 
                   Real const& startTime, Real const& endTime ) const
    {
        return getLogger( aFullPNString )->getData( startTime, endTime );
    }

    DataPointVectorSharedPtr
    getLoggerData( String const& aFullPNString,
                   Real const& start, Real const& end, 
                   Real const& interval ) const
    {
        return getLogger( aFullPNString )->getData( start, end, interval );
    }

    const Real 
    getLoggerStartTime( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getStartTime();
    }

    const Real 
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
                             "second parameter must be a tuple of 4 items.");
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

    const Logger::size_type 
    getLoggerSize( String const& aFullPNString ) const
    {
        return getLogger( aFullPNString )->getSize();
    }

    const std::pair< Real, String > getNextEvent() const
    {
        StepperEvent const& aNextEvent( theModel.getTopEvent() );

        return std::make_pair(
            static_cast< Real >( aNextEvent.getTime() ),
            aNextEvent.getStepper()->getID() );
    }

    void step( const Integer aNumSteps )
    {
        if( aNumSteps <= 0 )
        {
            THROW_EXCEPTION( Exception,
                             "step( n ): n must be 1 or greater. ("
                             + stringCast( aNumSteps ) + " given.)" );
        }

        start();

        Integer aCounter( aNumSteps );
        do
        {
            theModel.step();
            
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

    const Real getCurrentTime() const
    {
        return theModel.getCurrentTime();
    }

    void run()
    {
        start();

        do
        {
            unsigned int aCounter( theEventCheckInterval );
            do
            {
                theModel.step();
                --aCounter;
            }
            while( aCounter != 0 );
            
            handleEvent();

        }
        while( theRunningFlag );
    }

    void run( const Real aDuration )
    {
        if( aDuration <= 0.0 )
        {
            THROW_EXCEPTION( Exception,
                             "duration must be greater than 0. ("
                             + stringCast( aDuration ) + " given.)" );
        }

        start();

        const Real aStopTime( theModel.getCurrentTime() + aDuration );

        // setup SystemStepper to step at aStopTime

        //FIXME: dirty, ugly!
        Stepper* aSystemStepper( theModel.getSystemStepper() );
        aSystemStepper->setCurrentTime( aStopTime );
        aSystemStepper->setStepInterval( 0.0 );

        theModel.getScheduler().updateEvent( 0, aStopTime );


        if ( theEventHandler )
        {
            while ( theRunningFlag )
            {
                unsigned int aCounter( theEventCheckInterval );
                do 
                {
                    if( theModel.getTopEvent().getStepper() == aSystemStepper )
                    {
                        theModel.step();
                        stop();
                        break;
                    }
                    
                    theModel.step();

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
                if( theModel.getTopEvent().getStepper() == aSystemStepper )
                {
                    theModel.step();
                    stop();
                    break;
                }

                theModel.step();
            }
        }

    }

    void stop()
    {
        theRunningFlag = false;

        theModel.flushLoggers();
    }

    void setEventHandler( boost::shared_ptr< EventHandler > const& anEventHandler )
    {
        theEventHandler = anEventHandler;
    }

    const PolymorphVector getDMInfo() const
    {
        typedef ModuleMaker< EcsObject >::ModuleMap ModuleMap;
        PolymorphVector aVector;
        const ModuleMap& modules( thePropertiedObjectMaker->getModuleMap() );

        for( ModuleMap::const_iterator i( modules.begin() );
                    i != modules.end(); ++i )
        {
            PolymorphVector anInnerVector;
            const PropertyInterfaceBase* info(
                reinterpret_cast< const PropertyInterfaceBase *>(
                    i->second->getInfo() ) );
            const char* aFilename( i->second->getFileName() );

            aVector.push_back( boost::make_tuple(
                info->getTypeName(), i->second->getModuleName(),
                String( aFilename ? aFilename: "" ) ) );
        }

        return aVector;
    }

    PropertyInterfaceBase::PropertySlotMap const&
    getPropertyInfo( String const& aClassname ) const
    {
        return theModel.getPropertyInterface( aClassname ).getPropertySlotMap();
    }

    const char getDMSearchPathSeparator() const
    {
        return Model::PATH_SEPARATOR;
    }

    const std::string getDMSearchPath() const
    {
        return theModel.getDMSearchPath();
    }

    void setDMSearchPath( const std::string& aDMSearchPath )
    {
        theModel.setDMSearchPath( aDMSearchPath );
    }

    Logger* getLogger( String const& aFullPNString ) const
    {
        return theModel.getLoggerBroker().getLogger( aFullPNString );
    }

protected:

    ModelRef getModel() 
    { 
        return theModel; 
    }

    Model const& getModel() const 
    { 
        return theModel; 
    }

    inline void handleEvent()
    {
        if ( theEventHandler )
        { 
            while ( ( *theEventHandler )() );
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

    boost::shared_ptr< EventHandler >   theEventHandler;

    ModuleMaker< EcsObject >* thePropertiedObjectMaker;
    Model           theModel;
};

static int PropertyAttributes_GetItem( PropertyAttributes const* self, int idx )
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

static Polymorph Entity___getattr__( Entity* self, std::string key )
{
    if ( key.size() > 0 )
        key[ 0 ] = std::toupper( key[ 0 ] );
    return self->getProperty( key );
}

static void Entity___setattr__( Entity* self, std::string key, Polymorph value )
{
    if ( key.size() > 0 )
        key[ 0 ] = std::toupper( key[ 0 ] );
    self->setProperty( key, value );
}

template< typename T_ >
static PyObject* writeOnly( T_* )
{
    PyErr_SetString( PyExc_AttributeError, "Write-only attributes." );
    return py::incref( Py_None );
}

BOOST_PYTHON_MODULE( _ecs )
{
    typedef Simulator< PythonEventHandler > SimulatorImpl;

    DataPointVectorWrapper< DataPoint >::__class_init__();
    DataPointVectorWrapper< LongDataPoint >::__class_init__();
    STLIteratorWrapper< VariableReferenceVector::const_iterator >::__class_init__();

    // without this it crashes when Logger::getData() is called. why?
    import_array();

    registerTupleConverters< std::pair< Real, String > >();
    PolymorphToPythonConverter::addToRegistry();
    StringKeyedMapToPythonConverter< PolymorphMap >::addToRegistry();
    PropertySlotMapToPythonConverter::addToRegistry();
    DataPointVectorSharedPtrConverter::addToRegistry();

    PolymorphRetriever::addToRegistry();
    CallableSharedPtrRetriever< PythonEventHandler >::addToRegistry();

    // functions
    py::register_exception_translator< Exception >( &translateException );
    py::register_exception_translator< std::exception >( &translateException );
    py::register_exception_translator< std::range_error >( &translateRangeError );

    py::def( "getLibECSVersionInfo", &getLibECSVersionInfo );
    py::def( "getLibECSVersion",         &getVersion );

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
        .def( "__getitem__", &PropertyAttributes_GetItem )
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
              ( void ( VariableReferences::* )( String const&, String const&, Integer const&, bool ) )
              &VariableReferences::add )
        .def( "add",
              ( void ( VariableReferences::* )( String const&, String const&, Integer const& ) )
              &VariableReferences::add )
        .def( "remove", &VariableReferences::remove )
        .def( "__getitem__", &VariableReferences::__getitem__,
              return_copy_const_reference() )
        .def( "__len__", &VariableReferences::__len__ )
        .def( "__iter__", &VariableReferences::__iter__ )
        .def( "__repr__", &VariableReferences::__repr__ )
        .def( "__str__", &VariableReferences::__repr__ )
        ;

    py::class_< VariableReference >( "VariableReference", py::no_init )
        // properties
        .add_property( "superSystem",
            py::make_function(
                &VariableReference::getSuperSystem,
                py::return_internal_reference<> () ) )
        .add_property( "coefficient", &VariableReference::getCoefficient )
        .add_property( "name",        &VariableReference::getName )
        .add_property( "isAccessor",  &VariableReference::isAccessor )
        .add_property( "variable",
                py::make_function(
                    &VariableReference::getVariable,
                    py::return_internal_reference<>() ) )
        .def( "__str__", &VariableReference___repr__ )
        .def( "__repr__", &VariableReference___repr__ )
        ;

    py::class_< Stepper, py::bases<>, Stepper, boost::noncopyable >
        ( "Stepper", py::no_init )
        .add_property( "priority",
                       &Stepper::getPriority,
                       &Stepper::setPriority )
        .add_property( "stepInterval",
                       &Stepper::getStepInterval, 
                       &Stepper::setStepInterval )
        .add_property( "maxStepInterval",
                       &Stepper::getMaxStepInterval,
                       &Stepper::setMaxStepInterval )
        .add_property( "minStepInterval",
                       &Stepper::getMinStepInterval,
                       &Stepper::setMinStepInterval )
        .add_property( "rngSeed", &writeOnly<Stepper>, &Stepper::setRngSeed )
        ;

    py::class_< Entity, py::bases<>, Entity, boost::noncopyable >
        ( "Entity", py::no_init )
        // properties
        .add_property( "superSystem",
            py::make_function( &Entity::getSuperSystem,
            py::return_internal_reference<> () ) )
        .def( "__setattr__", &Entity___setattr__ )
        .def( "__getattr__", &Entity___getattr__ )
        ;

    py::class_< System, py::bases< Entity >, System, boost::noncopyable>
        ( "System", py::no_init )
        // properties
        .add_property( "size",        &System::getSize )
        .add_property( "sizeN_A",     &System::getSizeN_A )
        .add_property( "stepperID",   &System::getStepperID )
        ;

    py::class_< Process, py::bases< Entity >, Process, boost::noncopyable >
        ( "Process", py::no_init )
        .add_property( "activity",  &Process::getActivity,
                                    &Process::setActivity )
        .add_property( "priority",  &Process::getPriority )
        .add_property( "stepperID", &Process::getStepperID )
        .add_property( "variableReferences",
              py::make_function( &Process_get_variableReferences ) )
        // methods
        .def( "addValue",        &Process::addValue )
        .def( "setFlux",         &Process::setFlux )
        ;

    py::class_< Variable, py::bases< Entity >, Variable, boost::noncopyable >
        ( "Variable", py::no_init )
        .add_property( "value",  &Variable::getValue,
                                 &Variable::setValue )
        .add_property( "molarConc",  &Variable::getMolarConc,
                                     &Variable::setMolarConc  )
        .add_property( "numberConc", &Variable::getNumberConc,
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
              ( DataPointVectorSharedPtr( Logger::* )( void ) const )
              &Logger::getData )
        .def( "getData", 
              ( DataPointVectorSharedPtr( Logger::* )(
                Real const&, Real const& ) const )
              &Logger::getData )
        .def( "getData",
              ( DataPointVectorSharedPtr( Logger::* )(
                     Real const&, Real const&, Real const& ) const )
              &Logger::getData )
        ;

    // Simulator class
    py::class_< SimulatorImpl, py::bases<>, boost::shared_ptr< SimulatorImpl >, boost::noncopyable >( "Simulator" )
        .def( py::init<>() )
        .def( "getClassInfo",
              &SimulatorImpl::getClassInfo )
        // Stepper-related methods
        .def( "createStepper",
              &SimulatorImpl::createStepper,
              py::return_internal_reference<>() )
        .def( "getStepper",
              &SimulatorImpl::getStepper,
              py::return_internal_reference<>() )
        .def( "deleteStepper",
              &SimulatorImpl::deleteStepper )
        .def( "getStepperList",
              &SimulatorImpl::getStepperList )
        .def( "getStepperPropertyList",
              &SimulatorImpl::getStepperPropertyList )
        .def( "getStepperPropertyAttributes", 
              &SimulatorImpl::getStepperPropertyAttributes )
        .def( "setStepperProperty",
              &SimulatorImpl::setStepperProperty )
        .def( "getStepperProperty",
              &SimulatorImpl::getStepperProperty )
        .def( "loadStepperProperty",
              &SimulatorImpl::loadStepperProperty )
        .def( "saveStepperProperty",
              &SimulatorImpl::saveStepperProperty )
        .def( "getStepperClassName",
              &SimulatorImpl::getStepperClassName )

        // Entity-related methods
        .def( "createEntity",
              &SimulatorImpl::createEntity )
        .def( "geteEntity",
              &SimulatorImpl::getEntity )
        .def( "deleteEntity",
              &SimulatorImpl::deleteEntity )
        .def( "getEntityList",
              &SimulatorImpl::getEntityList )
        .def( "entityExists",
              &SimulatorImpl::entityExists )
        .def( "getEntityPropertyList",
              &SimulatorImpl::getEntityPropertyList )
        .def( "setEntityProperty",
              &SimulatorImpl::setEntityProperty )
        .def( "getEntityProperty",
              &SimulatorImpl::getEntityProperty )
        .def( "loadEntityProperty",
              &SimulatorImpl::loadEntityProperty )
        .def( "saveEntityProperty",
              &SimulatorImpl::saveEntityProperty )
        .def( "getEntityPropertyAttributes", 
              &SimulatorImpl::getEntityPropertyAttributes )
        .def( "getEntityClassName",
              &SimulatorImpl::getEntityClassName )

        // Logger-related methods
        .def( "getLoggerList",
                    &SimulatorImpl::getLoggerList )    
        .def( "createLogger",
              ( Logger* ( SimulatorImpl::* )( String const& ) )
                    &SimulatorImpl::createLogger,
              py::return_internal_reference<> () )
        .def( "createLogger",                                 
              ( Logger* ( SimulatorImpl::* )( String const&, Logger::Policy const& ) )
              &SimulatorImpl::createLogger,
              py::return_internal_reference<>() )
        .def( "createLogger",                                 
              ( Logger* ( SimulatorImpl::* )( String const&, py::object ) )
                    &SimulatorImpl::createLogger,
              py::return_internal_reference<> () )
        .def( "getLogger", &SimulatorImpl::getLogger,
              py::return_internal_reference<>() )
        .def( "getLoggerData", 
              ( DataPointVectorSharedPtr( SimulatorImpl::* )(
                    String const& ) const )
              &SimulatorImpl::getLoggerData )
        .def( "getLoggerData", 
              ( DataPointVectorSharedPtr( SimulatorImpl::* )(
                    String const&, Real const&, Real const& ) const )
              &SimulatorImpl::getLoggerData )
        .def( "getLoggerData",
              ( DataPointVectorSharedPtr( SimulatorImpl::* )(
                     String const&, Real const&, 
                     Real const&, Real const& ) const )
              &SimulatorImpl::getLoggerData )
        .def( "getLoggerStartTime",
              &SimulatorImpl::getLoggerStartTime )    
        .def( "getLoggerEndTime",
              &SimulatorImpl::getLoggerEndTime )        
        .def( "getLoggerPolicy",
              &SimulatorImpl::getLoggerPolicy )
        .def( "setLoggerPolicy",
              ( void (SimulatorImpl::*)(
                    String const&, Logger::Policy const& ) )
              &SimulatorImpl::setLoggerPolicy )
        .def( "setLoggerPolicy",
              ( void (SimulatorImpl::*)(
                    String const& aFullPNString,
                    py::object aParamList ) )
              &SimulatorImpl::setLoggerPolicy )
        .def( "getLoggerSize",
              &SimulatorImpl::getLoggerSize )

        // Simulation-related methods
        .def( "initialize",
              &SimulatorImpl::initialize )
        .def( "getCurrentTime",
              &SimulatorImpl::getCurrentTime )
        .def( "getNextEvent",
              &SimulatorImpl::getNextEvent )
        .def( "stop",
              &SimulatorImpl::stop )
        .def( "step",
              ( void ( SimulatorImpl::* )( void ) )
              &SimulatorImpl::step )
        .def( "step",
              ( void ( SimulatorImpl::* )( const Integer ) )
              &SimulatorImpl::step )
        .def( "run",
              ( void ( SimulatorImpl::* )() )
              &SimulatorImpl::run )
        .def( "run",
              ( void ( SimulatorImpl::* )( const Real ) ) 
              &SimulatorImpl::run )
        .def( "getPropertyInfo",
              &SimulatorImpl::getPropertyInfo,
              return_copy_const_reference() )
        .def( "getDMInfo",
              &SimulatorImpl::getDMInfo )
        .def( "setEventHandler",
              &SimulatorImpl::setEventHandler )
        .add_property( "DMSearchPathSeparator",
                       &SimulatorImpl::getDMSearchPathSeparator )
        .def( "setDMSearchPath", &SimulatorImpl::setDMSearchPath )
        .def( "getDMSearchPath", &SimulatorImpl::getDMSearchPath )
        ;
}
