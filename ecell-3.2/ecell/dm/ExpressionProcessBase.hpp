//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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
//   Koichi Takahashi
//   Tatsuya Ishida
//
// E-Cell Project.
//

#ifndef __EXPRESSIONPROCESSBASE_HPP
#define __EXPRESSIONPROCESSBASE_HPP

#include <cassert>
#include <limits>

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/AssocVector.h>
#include <libecs/scripting/ExpressionCompiler.hpp>
#include <libecs/scripting/VirtualMachine.hpp>

template< typename Tmixin_ >
class ExpressionProcessBase
{
protected:
    DECLARE_ASSOCVECTOR(
        libecs::String,
        libecs::Real,
        std::less<const libecs::String>,
        PropertyMap
    );

private:
    class PropertyAccess
          : public libecs::scripting::PropertyAccess
    {
    public:
        PropertyAccess( Tmixin_& outer )
            : outer_( outer )
        {
        }

        virtual libecs::Real* get( const libecs::String& name )
        {
            PropertyMapIterator pos = outer_.thePropertyMap.find( name );
            return pos == outer_.thePropertyMap.end() ? 0: &(pos->second);
        }

    private:
        Tmixin_& outer_;
    };
    friend class PropertyAccess;

    class EntityResolver
          : public libecs::scripting::EntityResolver
    {
    public:
        EntityResolver( Tmixin_& outer )
            : outer_( outer )
        {
        }

        virtual libecs::Entity* get( const libecs::String& name )
        {
            if ( name == "self" )
                return &outer_;
            try {
                return outer_.getVariableReference( name ).getVariable();
            } catch ( const libecs::Exception& ) {
                return 0;
            }
        }

    private:
        Tmixin_& outer_;
    };
    friend class EntityResolver;

    class VariableReferenceResolver
          : public libecs::scripting::VariableReferenceResolver
    {
    public:
        VariableReferenceResolver( Tmixin_& outer )
            : outer_( outer )
        {
        }

        virtual const libecs::VariableReference* get(
                const libecs::String& name) const
        {
            try {
                return &outer_.getVariableReference( name );
            } catch ( const libecs::Exception& ) {
                return 0;
            }
        }

    private:
        Tmixin_& outer_;
    };
    friend class VariableReferenceResolver;

    class ErrorReporter
          : public libecs::scripting::ErrorReporter
    {
    public:
        ErrorReporter( Tmixin_& outer )
            : outer_( outer )
        {
        }
        
        virtual void error( const libecs::String& type,
                const libecs::String& _msg ) const
        {
            libecs::String msg = outer_.asString() + ": " +  _msg;
            if ( type == "NoSlot" )
                THROW_EXCEPTION( libecs::NoSlot, msg );
            else if ( type == "NotFound" )
                THROW_EXCEPTION( libecs::NotFound, msg );
            else
                THROW_EXCEPTION( libecs::UnexpectedError, msg );
        }

    private:
        Tmixin_& outer_;
    };
    friend class ErrorReporter;

public:

    LIBECS_DM_OBJECT_MIXIN( ExpressionProcessBase, Tmixin_ )
    {
        PROPERTYSLOT_SET_GET( libecs::String, Expression );
    }


    ExpressionProcessBase()
        : theRecompileFlag( true ), theCompiledCode( 0 )
    {
        // ; do nothing
    }

    ~ExpressionProcessBase()
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

    void defaultSetProperty( libecs::String const& aPropertyName,
                             libecs::PolymorphCref aValue )
    {
        thePropertyMap[ aPropertyName ] = aValue.as< libecs::Real >();
    }

    const libecs::Polymorph defaultGetProperty( libecs::String const& aPropertyName ) const
    {
        PropertyMapConstIterator aPropertyMapIterator(
            thePropertyMap.find( aPropertyName ) );

        if ( aPropertyMapIterator != thePropertyMap.end() ) {
            return aPropertyMapIterator->second;
        } else {
            THROW_EXCEPTION( libecs::NoSlot,
                             static_cast< Tmixin_ const* >( this )->asString() +
                             ": property [" + aPropertyName +
                             "] is not defined." );
        }
    }

    const libecs::StringVector defaultGetPropertyList() const
    {
        libecs::StringVector aVector;

        std::transform( thePropertyMap.begin(), thePropertyMap.end(),
                std::back_inserter( aVector ),
                libecs::SelectFirst< PropertyMap::value_type >() );

        return aVector;
    }

    const libecs::PropertyAttributes
    defaultGetPropertyAttributes( libecs::String const& aPropertyName ) const
    {
        return libecs::PropertyAttributes( libecs::PropertySlotBase::POLYMORPH,
                true, true, true, true, true );
    }

    void compileExpression()
    {
        ErrorReporter anErrorReporter( *static_cast< Tmixin_* >( this ) );
        PropertyAccess aPropertyAccess( *static_cast< Tmixin_* >( this ) );
        EntityResolver anEntityResolver( *static_cast< Tmixin_* >( this ) );
        VariableReferenceResolver aVarRefResolver( *static_cast< Tmixin_*>( this ) );
        libecs::scripting::ExpressionCompiler theCompiler(
                anErrorReporter, aPropertyAccess, anEntityResolver,
                aVarRefResolver );

        delete theCompiledCode;
        // it is possible that compileExpression throws an exception and
        // "theCompiledCode" remains uninitialized
        theCompiledCode = 0;

        theCompiledCode = theCompiler.compileExpression( theExpression );
    }

    PropertyMapCref getPropertyMap() const
    {
        return thePropertyMap;
    }

    void initialize()
    {
        if ( theRecompileFlag ) {
            compileExpression();
            theRecompileFlag = false;
        }
    }

protected:

    PropertyMapRef getPropertyMap()
    {
        return thePropertyMap;
    }

protected:
    libecs::String    theExpression;

    const libecs::scripting::Code* theCompiledCode;
    libecs::scripting::VirtualMachine theVirtualMachine;

    bool theRecompileFlag;

    PropertyMap thePropertyMap;
};

#endif /* __EXPRESSIONPROCESSBASE_HPP */

