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
//     Koichi Takahashi
//     Tatsuya Ishida
//
// E-Cell Project.
//

#ifndef __EXPRESSIONCOMPILER_HPP
#define __EXPRESSIONCOMPILER_HPP

#include <new>

#include <boost/spirit/core.hpp>
#include <boost/spirit/tree/ast.hpp>

#if SPIRIT_VERSION >= 0x1800
#define PARSER_CONTEXT parser_context<>
#else
#define PARSER_CONTEXT parser_context
#endif

#include "libecs/libecs.hpp"
#include "libecs/AssocVector.h"
#include "libecs/scripting/Instruction.hpp"

namespace libecs { namespace scripting {

DECLARE_VECTOR( unsigned char, Code );

class PropertyAccess
{
public:
    virtual Real* get(const libecs::String& name) = 0;

    inline Real* operator[](const libecs::String& name)
    {
        return get(name);
    }
};

class EntityResolver
{
public:
    virtual libecs::Entity* get(const libecs::String& name) = 0;

    inline libecs::Entity* operator[](const libecs::String& name)
    {
        return get(name);
    }
};

class VariableReferenceResolver 
{
public:
    virtual const libecs::VariableReference* get(const libecs::String& name) const = 0;

    inline const libecs::VariableReference* operator[](const libecs::String& name) const
    {
        return get(name);
    }
};

class ErrorReporter
{
public:
    virtual void error(const String& type, const String& msg) const = 0;

    inline const void operator()(const String& type, const String& msg)
    {
        error(type, msg);
    }
};

class LIBECS_API ExpressionCompiler
{
private:
public:
    DECLARE_VECTOR( char, CharVector );

public:

    ExpressionCompiler( ErrorReporter& anErrorReporter,
                        PropertyAccess& aPropertyAccess,
                        EntityResolver& anEntityResolver,
                        VariableReferenceResolver& aVarRefResolver )
            : theErrorReporter( anErrorReporter ),
              thePropertyAccess( aPropertyAccess ),
              theEntityResolver( anEntityResolver ),
              theVarRefResolver( aVarRefResolver )
    {
    }


    ~ExpressionCompiler() {
        ; // do nothing
    }

    const Code* compileExpression( libecs::StringCref anExpression );

private:
    ErrorReporter& theErrorReporter;
    PropertyAccess& thePropertyAccess;
    EntityResolver& theEntityResolver;
    VariableReferenceResolver& theVarRefResolver;
}; // ExpressionCompiler

} } // libecs::namespace scripting

#endif /* __EXPRESSIONCOMPILER_HPP */
