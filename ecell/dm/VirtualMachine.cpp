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

#include "libecs/libecs.hpp"
#include "VirtualMachine.hpp"

namespace scripting
{
using namespace libecs;

#define ENABLE_STACKOPS_FOLDING 1

template < Opcode OPCODE >
class Opcode2Instruction
{
    typedef void type;
};


#define DEFINE_OPCODE2INSTRUCTION( CODE )\
  template<> class\
  Opcode2Instruction< CODE >\
  {\
  public:\
    typedef Instruction< CODE > type;\
  }

      
DEFINE_OPCODE2INSTRUCTION( PUSH_REAL );
DEFINE_OPCODE2INSTRUCTION( NEG );
DEFINE_OPCODE2INSTRUCTION( ADD );
DEFINE_OPCODE2INSTRUCTION( SUB );
DEFINE_OPCODE2INSTRUCTION( MUL );
DEFINE_OPCODE2INSTRUCTION( DIV );
DEFINE_OPCODE2INSTRUCTION( LOAD_REAL );
DEFINE_OPCODE2INSTRUCTION( CALL_FUNC1 );
DEFINE_OPCODE2INSTRUCTION( CALL_FUNC2 );
DEFINE_OPCODE2INSTRUCTION( OBJECT_METHOD_INTEGER );
DEFINE_OPCODE2INSTRUCTION( OBJECT_METHOD_REAL );
DEFINE_OPCODE2INSTRUCTION( RET );

#undef DEFINE_OPCODE2INSTRUCTION

const Real VirtualMachine::execute( CodeCref aCode )
{

#define FETCH_OPCODE()\
    reinterpret_cast<const InstructionHead* const>( aPC )->getOpcode()

#define DECODE_INSTRUCTION( OPCODE )\
    typedef Opcode2Instruction<OPCODE>::type \
     CurrentInstruction;\
     const CurrentInstruction* const anInstruction( \
        reinterpret_cast<const CurrentInstruction* const>( aPC ) )


#define INCREMENT_PC( OPCODE )\
    aPC += sizeof( Opcode2Instruction<OPCODE>::type );\
 
    //    std::cout << #OPCODE << std::endl;

    StackElement aStack[100];
    //  aStack[0].theReal = 0.0;
    StackElement* aStackPtr( aStack - 1 );

    const unsigned char* aPC( aCode.data() );

    for (;;) {
        Real bypass;
        switch ( FETCH_OPCODE() ) {

#define SIMPLE_ARITHMETIC( OPCODE, OP )\
            ( aStackPtr - 1)->theReal OP##= aStackPtr->theReal;\
            INCREMENT_PC( OPCODE );\
            --aStackPtr

        case ADD: {
            SIMPLE_ARITHMETIC( ADD, + );

            continue;
        }

        case SUB: {
            SIMPLE_ARITHMETIC( SUB, - );

            continue;
        }

        case MUL: {
            SIMPLE_ARITHMETIC( MUL, * );

            continue;
        }

        case DIV: {
            SIMPLE_ARITHMETIC( DIV, / );

            continue;
        }

#undef SIMPLE_ARITHMETIC

        case CALL_FUNC2: {
            DECODE_INSTRUCTION( CALL_FUNC2 );

            ( aStackPtr - 1 )->theReal
            = ( anInstruction->getOperand() )( ( aStackPtr - 1 )->theReal,
                                               aStackPtr->theReal );
            --aStackPtr;

            INCREMENT_PC( CALL_FUNC2 );
            continue;
        }


        case CALL_FUNC1: {
            DECODE_INSTRUCTION( CALL_FUNC1 );

            aStackPtr->theReal
            = ( anInstruction->getOperand() )( aStackPtr->theReal );

            INCREMENT_PC( CALL_FUNC1 );
            continue;
        }

        case NEG: {
            aStackPtr->theReal = - aStackPtr->theReal;

            INCREMENT_PC( NEG );
            continue;
        }

#if 0
        case PUSH_INTEGER: {
            DECODE_INSTRUCTION( PUSH_INTEGER );

            ++aStackPtr;
            aStackPtr->theInteger = anInstruction->getOperand();

            INCREMENT_PC( PUSH_INTEGER );
            continue;
        }

        case PUSH_POINTER: {
            DECODE_INSTRUCTION( PUSH_POINTER );

            ++aStackPtr;
            aStackPtr->thePointer = anInstruction->getOperand();

            INCREMENT_PC( PUSH_POINTER );
            continue;
        }

#endif // 0

        case PUSH_REAL: {
            DECODE_INSTRUCTION( PUSH_REAL );

            bypass = anInstruction->getOperand();

            INCREMENT_PC( PUSH_REAL );
            goto bypass_real;
        }

        case LOAD_REAL: {
            DECODE_INSTRUCTION( LOAD_REAL );

            bypass = *( anInstruction->getOperand() );

            INCREMENT_PC( LOAD_REAL );
            goto bypass_real;
        }

        case OBJECT_METHOD_REAL: {
            DECODE_INSTRUCTION( OBJECT_METHOD_REAL );

            bypass = ( anInstruction->getOperand() )();

            INCREMENT_PC( OBJECT_METHOD_REAL );
            goto bypass_real;
        }

        case OBJECT_METHOD_INTEGER: {
            DECODE_INSTRUCTION( OBJECT_METHOD_INTEGER );

            bypass = static_cast<Real>( ( anInstruction->getOperand() )() );

            INCREMENT_PC( OBJECT_METHOD_INTEGER );
            goto bypass_real;
        }

        case RET: {
            return aStackPtr->theReal;
        }

        default: {
            THROW_EXCEPTION( UnexpectedError, "Invalid instruction." );
        }

        }

#if defined( ENABLE_STACKOPS_FOLDING )

bypass_real:

        // Fetch next opcode, and if it is the target of of the stackops folding,
        // do it here.   If not (default case), start the next loop iteration.
        switch ( FETCH_OPCODE() ) {
        case ADD: {
            aStackPtr->theReal += bypass;

            INCREMENT_PC( ADD );
            break;
        }

        case SUB: {
            aStackPtr->theReal -= bypass;

            INCREMENT_PC( SUB );
            break;
        }

        case MUL: {
            aStackPtr->theReal *= bypass;

            INCREMENT_PC( MUL );
            break;
        }

        case DIV: {
            aStackPtr->theReal /= bypass;

            INCREMENT_PC( DIV );
            break;
        }

        case CALL_FUNC2: {
            DECODE_INSTRUCTION( CALL_FUNC2 );

            aStackPtr->theReal
            = ( anInstruction->getOperand() )( aStackPtr->theReal, bypass );

            INCREMENT_PC( CALL_FUNC2 );
            break;
        }

        default: {
            // no need to do invalid instruction check here because
            // it will be done in the next cycle.

            ++aStackPtr;
            aStackPtr->theReal = bypass;

            break;
        }
        }

        continue;

#else /* defined( ENABLE_STACKOPS_FOLDING ) */

bypass_real:

        ++aStackPtr;
        aStackPtr->theReal = bypass;

        continue;

#endif /* defined( ENABLE_STACKOPS_FOLDING ) */

    }

#undef DECODE_INSTRUCTION
#undef FETCH_INSTRUCTION
#undef INCREMENT_PC

}

} // namespace scripting
