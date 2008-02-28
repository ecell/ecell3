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
#include "libecs/Process.hpp"
#include "libecs/MethodProxy.hpp"

#include "Instruction.hpp"

namespace scripting
{

DECLARE_ASSOCVECTOR(
    libecs::String,
    libecs::Real,
    std::less<const libecs::String>,
    PropertyMap
);

DECLARE_VECTOR( unsigned char, Code );

class ExpressionCompiler
{
private:
public:
    DECLARE_VECTOR( char, CharVector );

public:

    ExpressionCompiler( libecs::ProcessCref aProcess,
                        PropertyMapRef aPropertyMap )
            : theProcess( aProcess ), thePropertyMap( aPropertyMap )
    {
        populateMap();
    }


    ~ExpressionCompiler() {
        ; // do nothing
    }

    const Code* compileExpression( libecs::StringCref anExpression );

protected:
    void throw_exception( libecs::String type, libecs::String aString );

private:
    static void populateMap();

private:
    libecs::ProcessCref theProcess;
    PropertyMapRef thePropertyMap;
}; // ExpressionCompiler

} // namespace scripting

#endif /* __EXPRESSIONCOMPILER_HPP */
