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

namespace scripting
{

DECLARE_ASSOCVECTOR(
    libecs::String,
    libecs::Real,
    std::less<const libecs::String>,
    PropertyMap
);

enum Opcode { // the order of items is optimized. don't change.
    ADD = 0,    // no arg
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

DECLARE_VECTOR( unsigned char, Code );

class InstructionHead
{
public:

    InstructionHead( Opcode anOpcode )
        : theOpcode( anOpcode )
    {
        ; // do nothing
    }

    const Opcode getOpcode() const
    {
        return theOpcode;
    }

private:

    const Opcode  theOpcode;

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
class Opcode2Instruction;

template <Opcode OPCODE>
class Opcode2Operand
{
public:
    typedef NoOperand     type;
};


template <class OPERAND >
class InstructionBase
            : public InstructionHead
{
public:

    DECLARE_TYPE( OPERAND, Operand );

    InstructionBase( Opcode anOpcode, OperandCref anOperand )
        : InstructionHead( anOpcode ), theOperand( anOperand )
    {
        ; // do nothing
    }

    OperandCref getOperand() const
    {
        return theOperand;
    }

private:

    InstructionBase( Opcode );

protected:

    const Operand theOperand;
};


template <>
class InstructionBase<NoOperand>
            :
            public InstructionHead
{
public:

    InstructionBase( Opcode anOpcode )
            :
            InstructionHead( anOpcode ) {
        ; // do nothing
    }

    InstructionBase( Opcode, const NoOperand& );

};


/**
   Instruction Class
*/

template < Opcode OPCODE >
class Instruction
            : public InstructionBase<typename Opcode2Operand<OPCODE>::type>
{
public:
    typedef typename Opcode2Operand<OPCODE>::type Operand_;
    DECLARE_TYPE( Operand_, Operand );

    Instruction( OperandCref anOperand )
        : InstructionBase<Operand>( OPCODE, anOperand )
    {
        ; // do nothing
    }

    Instruction()
        : InstructionBase<Operand>( OPCODE )
    {
        ; // do nothing
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


#define DEFINE_OPCODE2INSTRUCTION( CODE )\
  template<> class\
    Opcode2Instruction<CODE>\
  {\
  public:\
    typedef Instruction<CODE> type;\
    typedef type::Operand operandtype;\
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
