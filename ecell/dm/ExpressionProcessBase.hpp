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
// authors:
//   Koichi Takahashi
//   Tatsuya Ishida
//
// E-Cell Project.
//

#ifndef __EXPRESSIONPROCESSBASE_HPP
#define __EXPRESSIONPROCESSBASE_HPP

#include <cassert>
#include <limits>

#include "libecs/libecs.hpp"
#include "ExpressionCompiler.hpp"
#include "VirtualMachine.hpp"

LIBECS_DM_CLASS( ExpressionProcessBase, libecs::Process )
{
public:

    LIBECS_DM_OBJECT_ABSTRACT( ExpressionProcessBase )
    {
        INHERIT_PROPERTIES( libecs::Process );

        PROPERTYSLOT_SET_GET( libecs::String, Expression );
    }


    ExpressionProcessBase()
        : theRecompileFlag( true ), theCompiledCode( 0 )
    {
        // ; do nothing
    }

    virtual ~ExpressionProcessBase()
    {
        delete theCompiledCode;
    }

    SET_METHOD( libecs::String, Expression )
    {
        theExpression = value;
        theRecompileFlag = true;
    }

    GET_METHOD( libecs::String, Expression )
    {
        return theExpression;
    }

    void defaultSetProperty( libecs::StringCref aPropertyName,
                             libecs::PolymorphCref aValue )
    {
        thePropertyMap[ aPropertyName ] = aValue.asReal();
    }

    const libecs::Polymorph defaultGetProperty( libecs::StringCref aPropertyName ) const
    {
        scripting::PropertyMapConstIterator aPropertyMapIterator(
            thePropertyMap.find( aPropertyName ) );

        if ( aPropertyMapIterator != thePropertyMap.end() ) {
            return aPropertyMapIterator->second;
        } else {
            THROW_EXCEPTION( libecs::NoSlot, getClassNameString() +
                             " : Property [" + aPropertyName +
                             "] is not defined " );
        }
    }

    const libecs::Polymorph defaultGetPropertyList() const
    {
        libecs::PolymorphVector aVector;

        for ( scripting::PropertyMapConstIterator aPropertyMapIterator(
                thePropertyMap.begin() );
                aPropertyMapIterator != thePropertyMap.end();
                ++aPropertyMapIterator ) {
            aVector.push_back( aPropertyMapIterator->first );
        }

        return aVector;
    }

    const libecs::Polymorph
    defaultGetPropertyAttributes( libecs::StringCref aPropertyName ) const
    {
        libecs::PolymorphVector aVector;

        libecs::Integer aPropertyFlag( 1 );

        aVector.push_back( aPropertyFlag ); // isSetable
        aVector.push_back( aPropertyFlag ); // isGetable
        aVector.push_back( aPropertyFlag ); // isLoadable
        aVector.push_back( aPropertyFlag ); // isSavable

        return aVector;
    }

    void compileExpression()
    {
        scripting::ExpressionCompiler theCompiler( *this, getPropertyMap() );

        delete theCompiledCode;
        // it is possible that compileExpression throws an expression and
        // "theCompiledCode" remains uninitialized
        theCompiledCode = 0;

        theCompiledCode = theCompiler.compileExpression( theExpression );
    }

    scripting::PropertyMapCref getPropertyMap() const
    {
        return thePropertyMap;
    }

    virtual void initialize()
    {
        libecs::Process::initialize();

        if ( theRecompileFlag ) {
            compileExpression();
            theRecompileFlag = false;
        }
    }

protected:

    scripting::PropertyMapRef getPropertyMap()
    {
        return thePropertyMap;
    }


protected:
    libecs::String    theExpression;

    const scripting::Code* theCompiledCode;
    scripting::VirtualMachine theVirtualMachine;

    bool theRecompileFlag;

    scripting::PropertyMap thePropertyMap;
};


LIBECS_DM_INIT_STATIC( ExpressionProcessBase, Process );

#endif /* __EXPRESSIONPROCESSBASE_HPP */

