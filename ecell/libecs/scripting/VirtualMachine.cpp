//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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
//
// E-Cell Project.
//

#include "libecs/libecs.hpp"
#include "libecs/Exceptions.hpp"
#include "libecs/scripting/VirtualMachine.hpp"

namespace libecs { namespace scripting {
using namespace libecs;

union StackElement
{
    libecs::Real    theReal;
    libecs::Integer theInteger;

    StackElement(): theReal( 0 ) {}
    StackElement(libecs::Real aReal ): theReal( aReal ) {}
    StackElement(libecs::Integer aInteger ): theInteger( aInteger ) {}
};

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

template< typename Telem_, std::size_t maxdepth_ >
class LightweightStack
{
public:
    typedef std::size_t size_type;

public:
    LightweightStack()
        : ptr_( elems_ )
    {
        elems_[ 0 ].theReal = 0x55aa55aa; // sentinel
    }

    void push_back(const Telem_& elem)
    {
        *(++ptr_) = elem;
        last = elem;
    }

    Telem_& pop()
    {
        last = *(ptr_ - 1);
        return *(ptr_--);
    }

    size_type size()
    {
        return ptr_ - elems_;
    }

    size_type capacity()
    {
        return maxdepth_;
    }

    template<size_type bkidx>
    Telem_& peek()
    {
        if ( bkidx == 0 )
            return last;
        return *( ptr_ - bkidx );
    }

    void pop_back()
    {
        last = *(--ptr_);
    }

private:
    Telem_ elems_[ maxdepth_ + 1 ];
    Telem_* ptr_;
    Telem_ last;
};


const Real VirtualMachine::execute( Code const& aCode )
{

#define FETCH_OPCODE()\
    reinterpret_cast<const InstructionHead* const>( aPC )->getOpcode()

#define DECODE_INSTRUCTION( OPCODE )\
    typedef Opcode2Instruction<OPCODE>::type \
     CurrentInstruction;\
     const CurrentInstruction* const anInstruction( \
        reinterpret_cast<const CurrentInstruction* const>( aPC ) )


#define INCREMENT_PC( OPCODE )\
    aPC += sizeof( Opcode2Instruction<OPCODE>::type )\
 
    LightweightStack<StackElement, 100> aStack;

    const unsigned char* aPC( &aCode.front() );

    for (;;) {
        switch ( FETCH_OPCODE() ) {

#define SIMPLE_ARITHMETIC( OPCODE, OP ) \
    aStack.peek< 1 >().theReal OP##= aStack.peek< 0 >().theReal, \
    aStack.pop_back(), \
    INCREMENT_PC(OPCODE)

        case ADD:
            SIMPLE_ARITHMETIC( ADD, + );
            break;

        case SUB:
            SIMPLE_ARITHMETIC( SUB, - );
            break;

        case MUL:
            SIMPLE_ARITHMETIC( MUL, * );
            break;

        case DIV:
            SIMPLE_ARITHMETIC( DIV, / );
            break;
#undef SIMPLE_ARITHMETIC

        case CALL_FUNC2:
            {
                DECODE_INSTRUCTION( CALL_FUNC2 );

                aStack.peek< 1 >().theReal = anInstruction->getOperand()(
                       aStack.peek< 1 >().theReal,
                       aStack.peek< 0 >().theReal );
                aStack.pop_back();

                INCREMENT_PC( CALL_FUNC2 );
                break;
            }


        case CALL_FUNC1:
            {
                DECODE_INSTRUCTION( CALL_FUNC1 );

                aStack.peek< 0 >().theReal = anInstruction->getOperand()(
                    aStack.peek< 0 >().theReal );

                INCREMENT_PC( CALL_FUNC1 );
                break;
            }

        case NEG:
            {
                aStack.peek< 0 >().theReal = -aStack.peek< 0 >().theReal;
                INCREMENT_PC( NEG );
                break;
            }

        case PUSH_REAL:
            {
                DECODE_INSTRUCTION( PUSH_REAL );

                aStack.push_back( StackElement( anInstruction->getOperand() ) );

                INCREMENT_PC( PUSH_REAL );
                break;
            }

        case LOAD_REAL:
            {
                DECODE_INSTRUCTION( LOAD_REAL );

                aStack.push_back( StackElement( *( anInstruction->getOperand() ) ) );

                INCREMENT_PC( LOAD_REAL );
                break;
            }

        case OBJECT_METHOD_REAL:
            {
                DECODE_INSTRUCTION( OBJECT_METHOD_REAL );

                aStack.push_back( StackElement( anInstruction->getOperand()() ) );

                INCREMENT_PC( OBJECT_METHOD_REAL );
                break;
            }

        case OBJECT_METHOD_INTEGER:
            {
                DECODE_INSTRUCTION( OBJECT_METHOD_INTEGER );

                aStack.push_back( static_cast<Real>( ( anInstruction->getOperand() )() ) );

                INCREMENT_PC( OBJECT_METHOD_INTEGER );
                break;
            }

        case RET:
            return aStack.peek< 0 >().theReal;

        default:
            THROW_EXCEPTION( UnexpectedError,
                             ": invalid instruction" );

        }
    }

#undef DECODE_INSTRUCTION
#undef FETCH_INSTRUCTION
#undef INCREMENT_PC

}

} } // namespace libecs::scripting
