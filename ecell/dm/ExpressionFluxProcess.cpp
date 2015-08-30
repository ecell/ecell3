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
//    Tatsuya Ishida
//
// E-Cell Project.
//

#include <libecs/ContinuousProcess.hpp>
#include "SingleExpressionProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_MIXIN( ExpressionFluxProcess, ContinuousProcess,
                       SingleExpressionProcessBase )
{
public:
    LIBECS_DM_OBJECT( ExpressionFluxProcess, Process )
    {
        INHERIT_PROPERTIES( ContinuousProcess );
        INHERIT_PROPERTIES( _LIBECS_MIXIN_CLASS_ );
        CLASS_DESCRIPTION("ExpressionFluxProcess is designed for easy and efficient representations of continuous flux rate equations.    \"Expression\" property accepts a plain text rate expression.    The expression must be evaluated to give a flux rate in number per second, not concentration per second.");
    }

    ExpressionFluxProcess()
    {
        //FIXME: additional properties:
        // Unidirectional     -> call declareUnidirectional() in initialize()
        //                                         if this is set
    }

    virtual ~ExpressionFluxProcess()
    {
        ; // do nothing
    }

    virtual void initialize()
    {
        theVirtualMachine.setModel( getModel() );
        
        _LIBECS_MIXIN_CLASS_::initialize();
        ContinuousProcess::initialize();
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

    virtual void fire()
    { 
        setFlux( theVirtualMachine.execute( *theCompiledCode ) );
    }
};

LIBECS_DM_INIT( ExpressionFluxProcess, Process );
