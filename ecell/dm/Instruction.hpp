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

#ifndef __INSTRUCTION_HPP
#define __INSTRUCTION_HPP

namespace scripting {

enum Opcode { // the order of items is optimized. don't change.
    NOP,
    ADD,    // no arg
    SUB,        // no arg
    MUL,        // no arg
    DIV,        // no arg
    CALL_FUNC2, // RealFunc2
    // Those instructions above are candidates of stack operations folding
    // in the stack machine, and grouped here to optimize the switch().
    CALL_FUNC1, // RealFunc1
    NEG,        // no arg
    PUSH_REAL,  // Real
    LOAD_REAL,  // Real*
    OBJECT_METHOD_REAL, //VariableReferencePtr, VariableReferenceMethodPtr
    OBJECT_METHOD_INTEGER, // VariableReferenceIntegerMethodPtr
    RET,   // no arg
    END = RET
};


typedef libecs::SystemPtr(libecs::VariableReference::*VariableReferenceSystemMethodPtr)() const;
typedef libecs::SystemPtr(libecs::Process::* ProcessMethodPtr)() const;
typedef const libecs::Real(libecs::System::* SystemMethodPtr)() const;

typedef libecs::Real(*RealFunc0)();
typedef libecs::Real(*RealFunc1)( libecs::Real );
typedef libecs::Real(*RealFunc2)( libecs::Real, libecs::Real );
typedef libecs::ObjectMethodProxy<libecs::Real> RealObjectMethodProxy;
typedef libecs::ObjectMethodProxy<libecs::Integer> IntegerObjectMethodProxy;

class NoOperand {}; // Null type.

template <Opcode OPCODE>
class Opcode2Operand
{
public:
    typedef NoOperand     type;
};

class InstructionHead
{
public:
    InstructionHead(Opcode opcode)
        : theOpcode( opcode )
    {
    }

    const Opcode getOpcode() const
    {
        return theOpcode;
    }

private:
    const Opcode theOpcode;
};

template< Opcode Eop_, typename Toper_ >
class InstructionBase: public InstructionHead
{
public:
    typedef Toper_ operand_type;

public:
    InstructionBase( const operand_type & anOperand )
        : InstructionHead( Eop_ ), theOperand( anOperand )
    {
        ; // do nothing
    }

    const operand_type& getOperand() const
    {
        return theOperand;
    }

private:
    const operand_type theOperand;
};

template< Opcode Eop_>
class InstructionBase< Eop_, NoOperand >: public InstructionHead
{
public:
    typedef NoOperand operand_type;

public:
    InstructionBase( const operand_type & anOperand )
        : InstructionHead( Eop_ )
    {
        ; // do nothing
    }

    const operand_type& getOperand()
    {
        return NoOperand();
    }
};

template< Opcode Eop_ >
class Instruction
    : public InstructionBase< Eop_, typename Opcode2Operand< Eop_ >::type >
{
public:
    typedef InstructionBase< Eop_, typename Opcode2Operand< Eop_ >::type > base_type;
    typedef typename base_type::operand_type operand_type;

public:
    Instruction( const operand_type& oper = operand_type() )
        : base_type( oper )
    {
    }
};

#define SPECIALIZE_OPCODE2OPERAND( OP, OPE )\
  template<> class Opcode2Operand<OP>\
  {\
  public:\
    typedef OPE type;\
  };


SPECIALIZE_OPCODE2OPERAND( PUSH_REAL,                libecs::Real );
SPECIALIZE_OPCODE2OPERAND( LOAD_REAL,                libecs::RealPtr const );
SPECIALIZE_OPCODE2OPERAND( CALL_FUNC1,               RealFunc1 );
SPECIALIZE_OPCODE2OPERAND( CALL_FUNC2,               RealFunc2 );
SPECIALIZE_OPCODE2OPERAND( OBJECT_METHOD_REAL,       RealObjectMethodProxy );
SPECIALIZE_OPCODE2OPERAND( OBJECT_METHOD_INTEGER,    IntegerObjectMethodProxy );

} // namespace scripting

#endif /* __INSTRUCTION_HPP */
