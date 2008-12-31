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
// authors:
//    Tatsuya Ishida
//
// E-Cell Project.
//

#include "ExpressionProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS_MIXIN( ExpressionAssignmentProcess, Process,
                       ExpressionProcessBase )
{
public:
    LIBECS_DM_OBJECT( ExpressionAssignmentProcess, Process )
    {
        INHERIT_PROPERTIES( _LIBECS_MIXIN_CLASS_ );

        PROPERTYSLOT_SET_GET( String, Variable );
    }

    ExpressionAssignmentProcess()
    {
        //FIXME: additional properties:
        // Unidirectional     -> call declareUnidirectional() in initialize()
        //                                         if this is set
    }

    virtual ~ExpressionAssignmentProcess()
    {
        // ; do nothing
    }
 
    SET_METHOD( String, Variable )
    {
        theVariable = value;
    }

    GET_METHOD( String, Variable )
    {
        return theVariable;
    }

    virtual void defaultSetProperty( libecs::String const& aPropertyName,
                             libecs::PolymorphCref aValue )
    {
        return _LIBECS_MIXIN_CLASS_::defaultSetProperty( aPropertyName, aValue );
    }

    virtual const libecs::Polymorph defaultGetProperty( libecs::String const& aPropertyName ) const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetProperty( aPropertyName );
    }

    virtual const libecs::StringVector defaultGetPropertyList() const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetPropertyList();
    }

    virtual const libecs::PropertyAttributes
    defaultGetPropertyAttributes( libecs::String const& aPropertyName ) const
    {
        return _LIBECS_MIXIN_CLASS_::defaultGetPropertyAttributes( aPropertyName );
    }

    virtual void initialize()
    {
        _LIBECS_MIXIN_CLASS_::initialize();
            
        for( VariableReferenceVectorConstIterator i(
                    getVariableReferenceVector().begin() );
             i != getVariableReferenceVector().end(); ++i )
        {
            if( i->getCoefficient() != 0 )
            {
                theVariableReference = *i; 
            }
        }
    }

    virtual void fire()
    { 
        theVariableReference.setValue(
            theVariableReference.getCoefficient() * 
                theVirtualMachine.execute( *theCompiledCode ) );
    }

private:
    String theVariable;
    VariableReference theVariableReference;
};

LIBECS_DM_INIT( ExpressionAssignmentProcess, Process );
