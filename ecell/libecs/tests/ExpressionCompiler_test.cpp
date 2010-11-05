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
// modify it under the terms of the GNU General Public // License as published by the Free Software Foundation; either
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
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#define BOOST_TEST_MODULE "ExpressionCompiler"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/type_traits.hpp>

#include "scripting/ExpressionCompiler.hpp"
#include "Variable.hpp"
#include "VariableReference.hpp"

#include <iostream>
#include <cmath>

using namespace libecs;

template< typename T >
inline bool isReal( const T& )
{
    return false;
}

template<>
inline bool isReal( const Real& )
{
    return true;
}


template< typename T, bool can_cast = boost::is_floating_point< T >::value >
struct _toReal
{
    Real operator ()( const T& val ) const {
        return 0;
    }    
};

template< typename T >
struct _toReal< T, true >
{
    Real operator ()( const T& val ) const {
        return val;
    }    
};

template< typename T >
inline Real toReal( const T& val )
{
    return _toReal<T>()( val );
}


#define CHECK_INSTRUCTION( pc, op, oper ) do { \
    BOOST_REQUIRE_EQUAL( op, reinterpret_cast< const scripting::InstructionHead* >(pc)->getOpcode() ); \
    if ( isReal( oper ) ) \
    { \
        BOOST_CHECK_CLOSE_FRACTION( toReal( oper ), toReal( reinterpret_cast< const scripting::Instruction< op >* >( pc )->getOperand() ), 50 ); \
    } \
    else \
    { \
        BOOST_REQUIRE_EQUAL( oper, reinterpret_cast< const scripting::Instruction<op>* >( pc )->getOperand() ); \
    } \
    pc += sizeof( scripting::Instruction< op > ); \
} while (0)

BOOST_AUTO_TEST_CASE(testBasic)
{
    class ErrorReporter: public scripting::ErrorReporter {
    public:
        ErrorReporter() {}

        virtual void error( const String& type, const String& msg ) const {
            throw type;
        }
    } anErrorReporter;

    class PropertyAccess: public scripting::PropertyAccess {
    public:
        virtual Real* get( const String& name ) {
            return 0;
        }
    } aPropertyAccess;

    class VariableReferenceResolver: public scripting::VariableReferenceResolver {
    public:
        virtual const VariableReference* get(
                const String& name ) const
        {
            return 0;
        }
    } aVarRefResolver;


    class EntityResolver: public scripting::EntityResolver {
    public:
        virtual Entity* get( const String& name )
        {
            return 0;
        }
    } anEntityResolver;

    scripting::ExpressionCompiler ec(
        anErrorReporter, aPropertyAccess,
        anEntityResolver, aVarRefResolver );

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + 2") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 3.0 );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + 2 * 3 / 6 - 2" ) );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1.0 );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 6.0 );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 6.0 );
        CHECK_INSTRUCTION( pc, scripting::DIV, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 2.0 );
        CHECK_INSTRUCTION( pc, scripting::SUB, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL( eoc, pc );
    }

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1e-10 - 1e+10 * 1e-20" ) );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1e-10 );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1e+10 );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1e-20 );
        CHECK_INSTRUCTION( pc, scripting::MUL, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::SUB, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL( eoc, pc );
    }

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("sin(0)" ) );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 0. );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL( eoc, pc );
    }

    try {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 blah") );
        BOOST_FAIL("The preceeding expression unexpectedly succeeded");
    } catch (const String& msg) {
    }
}

BOOST_AUTO_TEST_CASE(testPropertyAccess)
{
    class ErrorReporter: public scripting::ErrorReporter {
    public:
        ErrorReporter() {}

        virtual void error( const String& type, const String& msg ) const {
            throw type;
        }
    } anErrorReporter;

    class PropertyAccess: public scripting::PropertyAccess {
    public:
        PropertyAccess()
            : a(0), b(0), c(0)
        {
        }

        virtual Real* get( const String& name ) {
            if (name == "a") {
                return &a;
            } else if (name == "b") {
                return &b;
            } else if (name == "c") {
                return &c;
            }
            return 0;
        }
    public:
        Real a, b, c;
    } aPropertyAccess;

    class VariableReferenceResolver: public scripting::VariableReferenceResolver {
    public:
        virtual const VariableReference* get(
                const String& name ) const
        {
            return 0;
        }
    } aVarRefResolver;


    class EntityResolver: public scripting::EntityResolver {
    public:
        virtual Entity* get( const String& name )
        {
            return 0;
        }
    } anEntityResolver;

    scripting::ExpressionCompiler ec(
        anErrorReporter, aPropertyAccess,
        anEntityResolver, aVarRefResolver );

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + a") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.a );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + a + b") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.a );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.b );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + a + b + 1 + c") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.a );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.b );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.c );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }

    try {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + d") );
        BOOST_FAIL("The preceeding expression unexpectedly succeeded");
    } catch (const String& type) {
        BOOST_CHECK_EQUAL(type, "NoSlot");
    }
}

BOOST_AUTO_TEST_CASE(testVariableReferenceResolver)
{
    class ErrorReporter: public scripting::ErrorReporter {
    public:
        ErrorReporter() {}

        virtual void error( const String& type, const String& msg ) const {
            throw type;
        }
    } anErrorReporter;

    class PropertyAccess: public scripting::PropertyAccess {
    public:
        PropertyAccess()
            : a(0), b(0), c(0)
        {
        }

        virtual Real* get( const String& name ) {
            if (name == "a") {
                return &a;
            } else if (name == "b") {
                return &b;
            } else if (name == "c") {
                return &c;
            }
            return 0;
        }
    public:
        Real a, b, c;
    } aPropertyAccess;

    class VariableReferenceResolver: public scripting::VariableReferenceResolver {
    public:
        virtual const VariableReference* get(
                const String& name ) const
        {
            if (name == "A") {
                return &a;
            } else if (name == "B") {
                return &b;
            } else if (name == "C") {
                return &c;
            }
            return 0;
        }
    public:
        VariableReference a, b, c;
    } aVarRefResolver;


    class EntityResolver: public scripting::EntityResolver {
    public:
        virtual Entity* get( const String& name )
        {
            return 0;
        }
    } anEntityResolver;

    scripting::ExpressionCompiler ec(
        anErrorReporter, aPropertyAccess,
        anEntityResolver, aVarRefResolver );

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + a") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.a );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + a + A.Value") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::LOAD_REAL, &aPropertyAccess.a );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::OBJECT_METHOD_REAL, (
                scripting::RealObjectMethodProxy::createConst<
                    Variable, &Variable::getValue >( 0 ) ) );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }

    try {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("1 + D.value") );
        BOOST_FAIL("The preceeding expression unexpectedly succeeded");
    } catch (const String& type) {
        BOOST_CHECK_EQUAL(type, "NotFound");
    }
}

BOOST_AUTO_TEST_CASE(testFunctionCall)
{
    class ErrorReporter: public scripting::ErrorReporter {
    public:
        ErrorReporter() {}

        virtual void error( const String& type, const String& msg ) const {
            throw type;
        }
    } anErrorReporter;

    class PropertyAccess: public scripting::PropertyAccess {
    public:
        PropertyAccess()
            : a(0), b(0), c(0)
        {
        }

        virtual Real* get( const String& name ) {
            if (name == "a") {
                return &a;
            } else if (name == "b") {
                return &b;
            } else if (name == "c") {
                return &c;
            }
            return 0;
        }
    public:
        Real a, b, c;
    } aPropertyAccess;

    class VariableReferenceResolver: public scripting::VariableReferenceResolver {
    public:
        virtual const VariableReference* get(
                const String& name ) const
        {
            if (name == "A") {
                return &a;
            } else if (name == "B") {
                return &b;
            } else if (name == "C") {
                return &c;
            }
            return 0;
        }
    public:
        VariableReference a, b, c;
    } aVarRefResolver;


    class EntityResolver: public scripting::EntityResolver {
    public:
        virtual Entity* get( const String& name )
        {
            return 0;
        }
    } anEntityResolver;

    scripting::ExpressionCompiler ec(
        anErrorReporter, aPropertyAccess,
        anEntityResolver, aVarRefResolver );

    {
        std::auto_ptr<const scripting::Code> code(
             ec.compileExpression("sqrt( A.Value ) + 1.0") );

        const unsigned char* pc = code->data();
        const unsigned char* eoc = &*code->end();
        CHECK_INSTRUCTION( pc, scripting::OBJECT_METHOD_REAL, (
                scripting::RealObjectMethodProxy::createConst<
                    Variable, &Variable::getValue >( 0 ) ) );
        CHECK_INSTRUCTION( pc, scripting::CALL_FUNC1,
                static_cast< double(*)( double ) >( &std::sqrt ) );
        CHECK_INSTRUCTION( pc, scripting::PUSH_REAL, 1 );
        CHECK_INSTRUCTION( pc, scripting::ADD, scripting::NoOperand() );
        CHECK_INSTRUCTION( pc, scripting::RET, scripting::NoOperand() );
        BOOST_CHECK_EQUAL(eoc, pc);
    }
}
