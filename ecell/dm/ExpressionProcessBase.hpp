//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2014 Keio University
//       Copyright (C) 2008-2014 RIKEN
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
//   Koichi Takahashi
//   Tatsuya Ishida
//   Yasuhiro Naito
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
    typedef Loki::AssocVector< libecs::String, libecs::Real,
                               std::less<const libecs::String> > PropertyMap;

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
            PropertyMap::iterator pos = outer_.thePropertyMap.find( name );
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
                THROW_EXCEPTION_ECSOBJECT( libecs::NoSlot, msg, &outer_ );
            else if ( type == "NotFound" )
                THROW_EXCEPTION_ECSOBJECT( libecs::NotFound, msg, &outer_ );
            else
                THROW_EXCEPTION_ECSOBJECT( libecs::UnexpectedError, msg, &outer_ );
        }

    private:
        Tmixin_& outer_;
    };
    friend class ErrorReporter;

public:

    LIBECS_DM_OBJECT_MIXIN( ExpressionProcessBase, Tmixin_ )
    {
    }

    ExpressionProcessBase()
    {
        // ; do nothing
    }

    virtual ~ExpressionProcessBase()
    {
    }

    void defaultSetProperty( libecs::String const& aPropertyName,
                             libecs::Polymorph const& aValue )
    {
        thePropertyMap[ aPropertyName ] = aValue.as< libecs::Real >();
    }

    libecs::Polymorph defaultGetProperty( libecs::String const& aPropertyName ) const
    {
        PropertyMap::const_iterator aPropertyMapIterator(
            thePropertyMap.find( aPropertyName ) );

        if ( aPropertyMapIterator != thePropertyMap.end() ) {
            return libecs::Polymorph( aPropertyMapIterator->second );
        } else {
            THROW_EXCEPTION_ECSOBJECT( libecs::NoSlot,
                             static_cast< Tmixin_ const* >( this )->asString() +
                             ": property [" + aPropertyName +
                             "] is not defined",
                             static_cast< Tmixin_ const* >( this ) );
        }
    }

    std::vector< libecs::String > defaultGetPropertyList() const
    {
        std::vector< libecs::String> aVector;

        std::transform( thePropertyMap.begin(), thePropertyMap.end(),
                std::back_inserter( aVector ),
                libecs::SelectFirst< PropertyMap::value_type >() );

        return aVector;
    }

    libecs::PropertyAttributes
    defaultGetPropertyAttributes( libecs::String const& aPropertyName ) const
    {
        return libecs::PropertyAttributes( libecs::PropertySlotBase::POLYMORPH,
                true, true, true, true, true );
    }

    void compileExpression( libecs::String const& anExpression, const libecs::scripting::Code* & aCompiledCode )
    {
        ErrorReporter anErrorReporter( *static_cast< Tmixin_* >( this ) );
        PropertyAccess aPropertyAccess( *static_cast< Tmixin_* >( this ) );
        EntityResolver anEntityResolver( *static_cast< Tmixin_* >( this ) );
        VariableReferenceResolver aVarRefResolver( *static_cast< Tmixin_*>( this ) );
        libecs::scripting::ExpressionCompiler theCompiler(
                anErrorReporter, aPropertyAccess, anEntityResolver,
                aVarRefResolver );

        delete aCompiledCode;
        // it is possible that compileExpression throws an exception and
        // "aCompiledCode" remains uninitialized
        aCompiledCode = 0;

        aCompiledCode = theCompiler.compileExpression( anExpression );
    }

    PropertyMap const& getPropertyMap() const
    {
        return thePropertyMap;
    }

    void initialize()
    {
    }

protected:

    PropertyMap& getPropertyMap()
    {
        return thePropertyMap;
    }

protected:
    PropertyMap thePropertyMap;
};

#endif /* __EXPRESSIONPROCESSBASE_HPP */

