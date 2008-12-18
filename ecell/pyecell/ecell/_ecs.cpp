//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//             This file is part of the E-Cell System
//
//             Copyright (C) 1996-2008 Keio University
//             Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// //
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

#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/range/size.hpp>
#include <boost/range/size_type.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/cast.hpp>
#include <boost/format.hpp>
#include <boost/format/group.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <numpy/arrayobject.h>
#include <stringobject.h>

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
namespace py = boost::python;

class PolymorphToPythonConverter
{
public:
    static void addToRegistry()
    {
        py::to_python_converter< Polymorph, PolymorphToPythonConverter >();
    }

    static PyObject* convert( PolymorphCref aPolymorph )
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

class PolymorphMapToPythonConverter
{
public:

    static void addToRegistry()
    {
        py::to_python_converter< PolymorphMap, PolymorphMapToPythonConverter>();
    }

    static PyObject* convert(const PolymorphMap& aPolymorphMapCref )
    {
        //PolymorphToPythonConverter aPolymorphConverter;
        PyObject * aPyDict(PyDict_New());
        PolymorphMap aPolymorphMap( aPolymorphMapCref );
        for ( PolymorphMap::iterator i( aPolymorphMap.begin() );
              i != aPolymorphMap.end(); ++i )
        {
            PyDict_SetItem( aPyDict, PyString_FromStringAndSize(
                i->first.data(), i->first.size() ),
                PolymorphToPythonConverter::convert( i->second ) );
                                            
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

class PythonCallable
{
public:
    PythonCallable( PyObject* aPyObjectPtr )
        : thePyObject( py::handle<>( aPyObjectPtr ) )
    {
        // this check isn't needed actually, because BPL does this automatically
        if ( !PyCallable_Check( thePyObject.ptr() ) )
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
    py::object thePyObject;
};


class PythonEventChecker
    : public PythonCallable, public EventChecker
{
public:

    PythonEventChecker( PyObject* aPyObjectPtr )
        : PythonCallable( aPyObjectPtr )
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
        Py_XDECREF( aPyObjectPtr );

        return aResult;
    }

};

class PythonEventHandler
    : public PythonCallable, public EventHandler
{
public:
    PythonEventHandler( PyObject* aPyObjectPtr )
        : PythonCallable( aPyObjectPtr )
    {
        ; // do nothing
    }
        
    virtual ~PythonEventHandler() {}

    virtual void operator()( void ) const
    {
        PyObject_CallFunction( thePyObject.ptr(), NULL );
    }
};


struct EventCheckerSharedPtrRetriever
{
public:

    static void addToRegistry()
    {
        py::converter::registry::insert(
                &convertible, &construct,
                py::type_id<EventCheckerSharedPtr>() );
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
            py::converter::rvalue_from_python_storage<EventCheckerSharedPtr>* >(
                data )->storage.bytes );

        data->convertible = new (storage) EventCheckerSharedPtr(
            new PythonEventChecker( aPyObjectPtr ) );
    }

};


class EventHandlerSharedPtrRetriever
{
public:

    static void addToRegistry() 
    {
        py::converter::registry::insert(
                &convertible, &construct,
                py::type_id<EventHandlerSharedPtr>() );
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
                           py::converter::rvalue_from_python_stage1_data* data )
    {
        void* storage( reinterpret_cast<
            py::converter::rvalue_from_python_storage<EventHandlerSharedPtr>* >(
                 data )->storage.bytes );

        data->convertible = new (storage) EventHandlerSharedPtr(
            new PythonEventHandler( aPyObjectPtr ) );
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
	PyObject_HEAD_INIT(&PyType_Type)
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
	PyObject_HEAD_INIT(&PyType_Type)
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

BOOST_PYTHON_MODULE( _ecs )
{
    if (!initialize())
    {
        throw std::runtime_error( "Failed to initialize libecs" );
    }

    DataPointVectorWrapper< DataPoint >::__class_init__();
    DataPointVectorWrapper< LongDataPoint >::__class_init__();

    // without this it crashes when Logger::getData() is called. why?
    import_array();

    PolymorphToPythonConverter::addToRegistry();
    PolymorphMapToPythonConverter::addToRegistry();
    DataPointVectorSharedPtrConverter::addToRegistry();

    PolymorphRetriever::addToRegistry();
    EventCheckerSharedPtrRetriever::addToRegistry();
    EventHandlerSharedPtrRetriever::addToRegistry();

    // functions
    py::register_exception_translator< Exception >( &translateException );
    py::register_exception_translator< std::exception >( &translateException );

    py::def( "getLibECSVersionInfo", &getLibECSVersionInfo );
    py::def( "getLibECSVersion",         &getVersion );

    typedef py::return_value_policy< py::reference_existing_object >
            return_existing_object;

    py::class_< VariableReference >( "VariableReference", py::no_init )
        // properties
        .add_property( "SuperSystem",
            py::make_function(
                &VariableReference::getSuperSystem,
                return_existing_object() ) )
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
              return_existing_object() )
        ;

    py::class_< Entity, py::bases<>, Entity, boost::noncopyable >
        ( "Entity", py::no_init )
        // properties
        .add_property( "SuperSystem",
            py::make_function(
                &Entity::getSuperSystem,
                return_existing_object() ) )
        .def( "getSuperSystem",     // this can be a property, but not supported
              &Entity::getSuperSystem,
              return_existing_object() )
        ;

    py::class_< System, py::bases< Entity >, System, boost::noncopyable>
        ( "System", py::no_init )
        // properties
        .add_property( "Size",        &System::getSize )
        .add_property( "SizeN_A",     &System::getSizeN_A )
        .add_property( "StepperID",   &System::getStepperID )
        ;

    py::class_< Process, py::bases< Entity >, Process, boost::noncopyable >
        ( "Process", py::no_init )
        .add_property( "Activity",  &Process::getActivity,
                                    &Process::setActivity )
        .add_property( "Priority",  &Process::getPriority )
        .add_property( "StepperID", &Process::getStepperID )

        // methods
        .def( "addValue",        &Process::addValue )
        .def( "getPositiveVariableReferenceOffset",         
              &Process::getPositiveVariableReferenceOffset )
        .def( "getVariableReference",             // this should be a property
              &Process::getVariableReference,
              py::return_internal_reference<>() )
        .def( "getVariableReferenceVector",             // this should be a property
              &Process::getVariableReferenceVector,
              return_existing_object() )
        .def( "getZeroVariableReferenceOffset",         
              &Process::getZeroVariableReferenceOffset )
        .def( "setFlux",         &Process::setFlux )
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

    py::class_< VariableReferenceVector >( "VariableReferenceVector" ) //, bases<>, VariableReferenceVector>
        .def( py::vector_indexing_suite< VariableReferenceVector >() )
        ;

    // Simulator class
    py::class_< Simulator >( "Simulator" )
        .def( py::init<>() )
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
              ( void ( Simulator::* )( StringCref, Polymorph ) )
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
