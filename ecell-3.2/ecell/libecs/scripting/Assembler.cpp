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
//     Koichi Takahashi
//     Tatsuya Ishida
//
// E-Cell Project.
//

#include "scripting/Assembler.hpp"

namespace libecs { namespace scripting {

#define APPEND_OBJECT_METHOD_REAL( OBJECT, CLASSNAME, METHODNAME )\
 appendInstruction\
   ( Instruction<OBJECT_METHOD_REAL>\
     ( RealObjectMethodProxy::\
       create< CLASSNAME, & CLASSNAME::METHODNAME >\
       ( OBJECT ) ) ) // \
 
#define APPEND_OBJECT_METHOD_INTEGER( OBJECT, CLASSNAME, METHODNAME )\
 appendInstruction\
   ( Instruction<OBJECT_METHOD_INTEGER>\
     ( IntegerObjectMethodProxy::\
       create< CLASSNAME, & CLASSNAME::METHODNAME >\
       ( OBJECT ) ) ) // \
 
void
Assembler::appendVariableReferenceMethodInstruction(
        libecs::VariableReference* aVariableReference,
        const libecs::String& aMethodName )
{

    if ( aMethodName == "MolarConc" ) {
        APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
                                   getMolarConc );
    } else if ( aMethodName == "NumberConc" ) {
        APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
                                   getNumberConc );
    } else if ( aMethodName == "Value" ) {
        APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
                                   getValue );
    } else if ( aMethodName == "Velocity" ) {
        APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
                                   getVelocity );
    } else if ( aMethodName == "Coefficient" ) {
        APPEND_OBJECT_METHOD_INTEGER( aVariableReference, VariableReference,
                                      getCoefficient );
    } else {
        THROW_EXCEPTION(
            NotFound, 
            String( "No such VariableReference attribute: " )
            + aMethodName
        );
    }
}

void
Assembler::appendSystemMethodInstruction(
        libecs::System* aSystemPtr, const libecs::String& aMethodName )
{
    if ( aMethodName == "Size" ) {
        APPEND_OBJECT_METHOD_REAL( aSystemPtr, System, getSize );
    } else if ( aMethodName == "SizeN_A" ) {
        APPEND_OBJECT_METHOD_REAL( aSystemPtr, System, getSizeN_A );
    } else {
        THROW_EXCEPTION(
            NotFound,
            String( "No such property: " ) + aMethodName
        );
    }

}

#undef APPEND_OBJECT_METHOD_REAL
#undef APPEND_OBJECT_METHOD_INTEGER

} } // namespace libecs::scripting
