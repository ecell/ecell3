//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2015 Keio University
//       Copyright (C) 2008-2015 RIKEN
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

// #include <iostream>

#include <boost/assert.hpp>

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
DEFINE_OPCODE2INSTRUCTION( CALL_FUNCA );
DEFINE_OPCODE2INSTRUCTION( CALL_DELAY );
DEFINE_OPCODE2INSTRUCTION( PUSH_TIME );
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
    }

    Telem_& pop()
    {
        Telem_& retval(*(ptr_--));
        return retval;
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
        return *( ptr_ - bkidx );
    }

    template<size_type bkidx>
    Telem_ const& peek() const
    {
        return *( ptr_ - bkidx );
    }

    void pop_back()
    {
        --ptr_;
    }

private:
    Telem_ elems_[ maxdepth_ + 1 ];
    Telem_* ptr_;
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
            /*
            aStack.peek< 1 >().theReal += aStack.peek< 0 >().theReal, 
            aStack.pop_back(), 
            aPC += sizeof( Opcode2Instruction<ADD>::type );
            */
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

        case CALL_DELAY:
            {
                DECODE_INSTRUCTION( CALL_DELAY );
                
                // std::cout << "Parse Dealy Function# " << aStack.peek< 0 >().theReal << std::endl;
                // std::cout << "  time  = " << aStack.peek< 1 >().theReal << std::endl;
                // std::cout << "  value = " << aStack.peek< 2 >().theReal << std::endl;
                
                Integer n = (Integer) aStack.peek< 0 >().theReal;
                
                if (( theDelayMap.find( n ) == theDelayMap.end() ) ||
                    ((*(theDelayMap[ n ].rbegin())).second != aStack.peek< 2 >().theReal ))
                {
                    theDelayMap[ n ][ theModel->getCurrentTime() ] = aStack.peek< 2 >().theReal;
                }

                // std::cout << "theDelayMap[ " << n << " ].begin() = " << (*(theDelayMap[ n ].begin())).second << std::endl;

                aStack.peek< 2 >().theReal =  getDelayedValue( n, aStack.peek< 1 >().theReal );
                aStack.pop_back();
                aStack.pop_back();

                INCREMENT_PC( CALL_DELAY );
                break;
            }

        case PUSH_TIME:
            {
                DECODE_INSTRUCTION( PUSH_TIME );

                aStack.push_back( StackElement( theModel->getCurrentTime() ) );

                INCREMENT_PC( PUSH_TIME );
                break;
            }

        case CALL_FUNCA:
            {
                DECODE_INSTRUCTION( CALL_FUNCA );
                
                std::vector< libecs::Real > args;

                // std::cout << "Parse Piecewise Function: numArg = " << aStack.peek< 0 >().theReal << std::endl;
/*
                libecs::Integer numArg( ( libecs::Integer ) aStack.pop().theReal ), i( 0 );
                for ( i = 0; i < numArg; i++ ) {
                    // std::cout << "  arg = " << aStack.peek< 0 >().theReal << std::endl;
                    args.push_back( aStack.pop().theReal );}
                
                aStack.push_back( anInstruction->getOperand()( args ));
*/
                libecs::Integer numArg( ( libecs::Integer ) aStack.peek< 0 >().theReal ), i( 0 );
                for ( i = 0; i < numArg; ++i ) {
                    aStack.pop_back();
                    // std::cout << "  arg = " << aStack.peek< 0 >().theReal << std::endl;
                    args.push_back( aStack.peek< 0 >().theReal );}
                
                aStack.peek< 0 >().theReal = anInstruction->getOperand()( args );
                
                INCREMENT_PC( CALL_FUNCA );
                break;
            }

        case CALL_FUNC2:
            {
                DECODE_INSTRUCTION( CALL_FUNC2 );

                aStack.peek< 1 >().theReal = anInstruction->getOperand()(
                       aStack.peek< 1 >().theReal,
                       aStack.peek< 0 >().theReal );
                aStack.pop_back();

                INCREMENT_PC( CALL_FUNC2 );
                /*
                typedef Opcode2Instruction<CALL_FUNC2>::type
                 CurrentInstruction;
                 const CurrentInstruction* const anInstruction( 
                    reinterpret_cast<const CurrentInstruction* const>( aPC ) );

                aStack.peek< 1 >().theReal = anInstruction->getOperand()(
                       aStack.peek< 1 >().theReal,
                       aStack.peek< 0 >().theReal );
                aStack.pop_back();

                aPC += sizeof( Opcode2Instruction<CALL_FUNC2>::type );
                */
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

const Real VirtualMachine::getDelayedValue( libecs::Integer n, libecs::Real t )
{
    BOOST_ASSERT( ! theDelayMap[ n ].empty() );
    BOOST_ASSERT( t >= 0.0 );

    TimeSeries::iterator i = theDelayMap[ n ].end();

    if ( theModel->getCurrentTime() >= t )
    {
        // std::cout << "(2) theDelayMap[ " << n << " ].size() = " << theDelayMap[ n ].size() << std::endl;
        i = theDelayMap[ n ].upper_bound( theModel->getCurrentTime() - t );
        // std::cout << "        upper_bound() = " << (*i).first << std::endl;
    } else {
        i = theDelayMap[ n ].begin();
    }
    
    if ( i != theDelayMap[ n ].begin() ) --i;
    return (*i).second;

}

} } // namespace libecs::scripting
