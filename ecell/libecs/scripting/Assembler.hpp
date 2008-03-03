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

#ifndef __ASSEMBLER_HPP
#define __ASSEMBLER_HPP

#include "libecs/System.hpp"
#include "libecs/VariableReference.hpp"
#include "libecs/scripting/Instruction.hpp"
#include "libecs/scripting/ExpressionCompiler.hpp"

namespace libecs { namespace scripting {

class Assembler
{
public:
    Assembler( Code* code )
        : theCode( code )
    {
    }

    template < class Tinstr_ >
    void appendInstruction( const Tinstr_& anInstruction )
    {
        Code::size_type aCodeSize( theCode->size() );
        theCode->resize( aCodeSize + sizeof( Tinstr_ ) );
        // XXX: hackish!!!
        new (&(*theCode)[aCodeSize]) Tinstr_( anInstruction );
    }

    void
    appendVariableReferenceMethodInstruction(
            libecs::VariableReferencePtr aVariableReference,
            libecs::StringCref aMethodName );

    void
    appendSystemMethodInstruction( libecs::SystemPtr aSystemPtr,
                                   libecs::StringCref aMethodName );

private:
    Code* theCode;
};

} } // namespace libecs::scripting

#endif /* __ASSEMBLER_HPP */
