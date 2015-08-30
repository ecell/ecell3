//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2015 Keio University
//       Copyright (C) 2008-2015 RIKEN
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

#include <libecs/FullID.hpp>
#include <libecs/ContinuousProcess.hpp>
#include "PythonProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_MIXIN( PythonFluxProcess, ContinuousProcess,
                       PythonProcessBase )
{
public:
    LIBECS_DM_OBJECT( PythonFluxProcess, Process )
    {
        INHERIT_PROPERTIES( _LIBECS_MIXIN_CLASS_ );
        INHERIT_PROPERTIES( ContinuousProcess );

        PROPERTYSLOT_SET_GET( String, Expression );
    }

    PythonFluxProcess()
    {
        //FIXME: additional properties:
        // Unidirectional     -> call declareUnidirectional() in initialize()
        //                                         if this is set
    }

    virtual ~PythonFluxProcess()
    {
        ; // do nothing
    }

    SET_METHOD( String, Expression )
    {
        theExpression = value;

        theCompiledExpression = compilePythonCode(
                theExpression, asString() + ":Expression",
                Py_eval_input );
    }

    GET_METHOD( String, Expression )
    {
        return theExpression;
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
        _LIBECS_MIXIN_CLASS_::initialize();
        ContinuousProcess::initialize();
    }

    virtual void fire()
    {
        boost::python::handle<> aHandle(
            PyEval_EvalCode(
                reinterpret_cast< PyCodeObject* >( theCompiledExpression.get() ),
                theGlobalNamespace.ptr(), theLocalNamespace.ptr() ) );

        boost::python::object aResultObject( aHandle );
        
        // do not use extract<double> for efficiency
        if( ! PyFloat_Check( aResultObject.ptr() ) )
        {
            THROW_EXCEPTION_INSIDE( SimulationError, 
                             asString() + ": "
                             "The expression gave a non-float object." );
        }

        const Real aFlux( PyFloat_AS_DOUBLE( aResultObject.ptr() ) );

        setFlux( aFlux );
    }

protected:

    String        theExpression;

    boost::python::handle<> theCompiledExpression;
};

LIBECS_DM_INIT( PythonFluxProcess, Process );
