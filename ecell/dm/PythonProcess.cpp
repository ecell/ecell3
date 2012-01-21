//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
// authors:
// Koichi Takahashi <shafi@e-cell.org>
// Nayuta Iwata
//
// E-Cell Project.
//

#include "PythonProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_MIXIN( PythonProcess, Process, PythonProcessBase )
{
public:
    LIBECS_DM_OBJECT( PythonProcess, Process )
    {
        INHERIT_PROPERTIES( _LIBECS_MIXIN_CLASS_ );
        INHERIT_PROPERTIES( Process );

        PROPERTYSLOT_SET_GET( Integer, IsContinuous );
        PROPERTYSLOT_SET_GET( Real, StepInterval );
        PROPERTYSLOT_SET( Real, Flux );
        PROPERTYSLOT_SET_GET( String, FireMethod );
        PROPERTYSLOT_SET_GET( String, InitializeMethod );
    }

    PythonProcess()
        : theIsContinuous( false )
    {
        setInitializeMethod( "" );
        setFireMethod( "" );

        //FIXME: additional properties:
        // Unidirectional     -> call declareUnidirectional() in initialize()
        //                                         if this is set
    }

    virtual ~PythonProcess()
    {
        ; // do nothing
    }

    virtual bool isContinuous() const
    {
        return theIsContinuous;
    }

    SET_METHOD( Integer, IsContinuous )
    {
        theIsContinuous = value;
    }

    virtual Real getStepInterval() const
    {
        return theStepInterval;
    }

    SET_METHOD( Real, StepInterval)
    {
        theStepInterval = value;
    }

    SET_METHOD( String, FireMethod )
    {
        theFireMethod = value;

        theCompiledFireMethod = compilePythonCode(
                theFireMethod,
                asString() + ":FireMethod",
                Py_file_input );

        // error check
    }

    GET_METHOD( String, FireMethod )
    {
        return theFireMethod;
    }

    SET_METHOD( String, InitializeMethod )
    {
        theInitializeMethod = value;

        theCompiledInitializeMethod = compilePythonCode(
                theInitializeMethod,
                asString() + ":InitializeMethod",
                Py_file_input );
    }

    GET_METHOD( String, InitializeMethod )
    {
        return theInitializeMethod;
    }

    virtual void defaultSetProperty( libecs::String const& aPropertyName,
                                     libecs::Polymorph const& aValue )
    {
        return _LIBECS_MIXIN_CLASS_::defaultSetProperty( aPropertyName, aValue );
    }

    virtual libecs::Polymorph defaultGetProperty( libecs::String const& aPropertyName ) const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetProperty( aPropertyName );
    }

    virtual std::vector< libecs::String > defaultGetPropertyList() const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetPropertyList();
    }

    virtual libecs::PropertyAttributes
    defaultGetPropertyAttributes( libecs::String const& aPropertyName ) const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetPropertyAttributes( aPropertyName );
    }

    virtual void initialize()
    {
        Process::initialize();
        _LIBECS_MIXIN_CLASS_::initialize();

        boost::python::handle<> a(
            PyEval_EvalCode(
                reinterpret_cast< PyCodeObject* >(
                    theCompiledInitializeMethod.get() ),
                theGlobalNamespace.ptr(), 
                theLocalNamespace.ptr() ) );
    }

    virtual void fire()
    {
        boost::python::handle<> a(
            PyEval_EvalCode(
                reinterpret_cast< PyCodeObject* >( theCompiledFireMethod.get() ),
                theGlobalNamespace.ptr(), 
                theLocalNamespace.ptr() ) );
    }

protected:

    String        theFireMethod;
    String        theInitializeMethod;

    boost::python::handle<> theCompiledFireMethod;
    boost::python::handle<> theCompiledInitializeMethod;

    bool theIsContinuous;
    Real theStepInterval;
};

LIBECS_DM_INIT( PythonProcess, Process );
