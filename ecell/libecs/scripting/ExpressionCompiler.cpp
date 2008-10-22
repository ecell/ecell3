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

#include <boost/assert.hpp>

#include "Util.hpp"
#include "scripting/ExpressionCompiler.hpp"
#include "scripting/Assembler.hpp"

namespace libecs { namespace scripting {

using namespace boost::spirit;
using namespace libecs;

typedef boost::spirit::tree_match<const char*> TreeMatch;
typedef TreeMatch::tree_iterator TreeIterator;

//  VariableReferenceMethodProxy;

typedef void (*InstructionAppender)( CodeRef );
    
DECLARE_ASSOCVECTOR(
    libecs::String,
    libecs::Real(*)(libecs::Real),
    std::less<const libecs::String>,
    FunctionMap1
);

DECLARE_ASSOCVECTOR(
    libecs::String,
    libecs::Real(*)(libecs::Real, libecs::Real),
    std::less<const libecs::String>,
    FunctionMap2
);

DECLARE_ASSOCVECTOR(
    libecs::String,
    libecs::Real,
    std::less<const libecs::String>,
    ConstantMap
);

class Tokens
{
public:
    static const int GROUP           =  1;
    static const int INTEGER         =  2;
    static const int FLOAT           =  3;
    static const int NEGATIVE        =  4;
    static const int EXPONENT        =  5;
    static const int FACTOR          =  6;
    static const int POWER           =  7;
    static const int TERM            =  8;
    static const int EXPRESSION      =  9;
    static const int VARIABLE        = 10;
    static const int CALL_FUNC       = 11;
    static const int SYSTEM_FUNC     = 12;
    static const int SYSTEM_PROPERTY = 13;
    static const int IDENTIFIER      = 14;
    static const int CONSTANT        = 15;
};

class CompileGrammar: public grammar<CompileGrammar>, public Tokens
{
public:
    template <typename ScannerT>
    struct definition {
#define leafNode( str ) lexeme_d[leaf_node_d[str]]
#define rootNode( str ) lexeme_d[root_node_d[str]]

        definition( CompileGrammar const& /*self*/ ) {
            integer     =   leafNode( +digit_p );
            floating    =   leafNode( +digit_p >> ch_p('.') >> +digit_p );

            exponent    =   ( floating | integer ) >>
                            rootNode( ch_p('e') | ch_p('E') ) >>
                            ( ch_p('-') >> integer |
                              discard_node_d[ ch_p('+') ] >> integer |
                              integer );

            negative    = rootNode( ch_p('-') ) >> factor;

            identifier  =   leafNode( alpha_p >> *( alnum_p | ch_p('_') ) );

            variable    =   identifier >> rootNode( ch_p('.') ) >> identifier;

            system_func = identifier >> system_property >> rootNode( ch_p('.') ) >> identifier;

            system_property = +( rootNode( ch_p('.') ) >>
                                 leafNode( +( alpha_p | ch_p('_') ) ) >>
                                 discard_node_d[ ch_p('(') ] >>
                                 discard_node_d[ ch_p(')') ] );


            // XXX: The syntax is dirty due to the bug of Spirit
            //call_func = rootNode( +alpha_p ) >>
            call_func = (   rootNode( str_p("eq") )
                            | rootNode( str_p("neq") )
                            | rootNode( str_p("gt") )
                            | rootNode( str_p("lt") )
                            | rootNode( str_p("geq") )
                            | rootNode( str_p("leq") )
                            | rootNode( str_p("and") )
                            | rootNode( str_p("or") )
                            | rootNode( str_p("xor") )
                            | rootNode( str_p("not") )
                            | rootNode( str_p("abs") )
                            | rootNode( str_p("sqrt") )
                            | rootNode( str_p("pow") )
                            | rootNode( str_p("exp") )
                            | rootNode( str_p("log10") )
                            | rootNode( str_p("log") )
                            | rootNode( str_p("floor") )
                            | rootNode( str_p("ceil") )
                            | rootNode( str_p("sin") )
                            | rootNode( str_p("cos") )
                            | rootNode( str_p("tan") )
                            | rootNode( str_p("sinh") )
                            | rootNode( str_p("cosh") )
                            | rootNode( str_p("tanh") )
                            | rootNode( str_p("asin") )
                            | rootNode( str_p("acos") )
                            | rootNode( str_p("atan") )
                            | rootNode( str_p("fact") )
                            | rootNode( str_p("asinh") )
                            | rootNode( str_p("acosh") )
                            | rootNode( str_p("atanh") )
                            | rootNode( str_p("asech") )
                            | rootNode( str_p("acsch") )
                            | rootNode( str_p("acoth") )
                            | rootNode( str_p("sech") )
                            | rootNode( str_p("csch") )
                            | rootNode( str_p("coth") )
                            | rootNode( str_p("asec") )
                            | rootNode( str_p("acsc") )
                            | rootNode( str_p("acot") )
                            | rootNode( str_p("sec") )
                            | rootNode( str_p("csc") )
                            | rootNode( str_p("cot") )
                        ) >>
                        inner_node_d[ ch_p('(') >>
                                      ( expression >>
                                        *( discard_node_d[ ch_p(',') ] >>
                                           expression ) ) >>
                                      ch_p(')') ];


            group       =   inner_node_d[ ch_p('(') >> expression >> ch_p(')')];

            constant    =   exponent | floating | integer;

            factor      =   call_func
                            |   system_func
                            |   variable
                            |   constant
                            |   group
                            |   identifier
                            |   negative;

            power = factor >> *( rootNode( ch_p('^') ) >> factor );

            term        =  power >>
                           *( ( rootNode( ch_p('*') ) >> power )
                              |  ( rootNode( ch_p('/') ) >> power ) );
            //|  ( rootNode( ch_p('^') ) >> power ) );


            expression  =  term >>
                           *( (rootNode( ch_p('+') ) >> term)
                              |  (rootNode( ch_p('-') ) >> term) );
        }

        rule<ScannerT, PARSER_CONTEXT, parser_tag<VARIABLE> >     variable;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<CALL_FUNC> >    call_func;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<EXPRESSION> >   expression;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<TERM> >         term;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<POWER> >        power;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<FACTOR> >       factor;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<FLOAT> >        floating;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<EXPONENT> >     exponent;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<INTEGER> >      integer;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<NEGATIVE> >     negative;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<GROUP> >        group;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<IDENTIFIER> >   identifier;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<CONSTANT> >     constant;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<SYSTEM_FUNC> >  system_func;
        rule<ScannerT, PARSER_CONTEXT, parser_tag<SYSTEM_PROPERTY> >
        system_property;

        rule<ScannerT, PARSER_CONTEXT, parser_tag<EXPRESSION> > const&
        start() const {
            return expression;
        }
    };

#undef leafNode
#undef rootNode
};

struct CompilerConfig
{
    ConstantMap  theConstantMap;
    FunctionMap1 theFunctionMap1;
    FunctionMap2 theFunctionMap2;
    CompileGrammar theGrammar;

    CompilerConfig()
    {
        // set ConstantMap
        theConstantMap["true"]  = 1.0;
        theConstantMap["false"] = 0.0;
        theConstantMap["pi"]    = M_PI;
        theConstantMap["NaN"]   = std::numeric_limits<Real>::quiet_NaN();
        theConstantMap["INF"]   = std::numeric_limits<Real>::infinity();
        theConstantMap["N_A"]   = N_A;
        theConstantMap["exp"]   = M_E;

        // set ExpressionCompiler::FunctionMap1
        theFunctionMap1["abs"]   = std::fabs;
        theFunctionMap1["sqrt"]  = std::sqrt;
        theFunctionMap1["exp"]   = std::exp;
        theFunctionMap1["log10"] = std::log10;
        theFunctionMap1["log"]   = std::log;
        theFunctionMap1["floor"] = std::floor;
        theFunctionMap1["ceil"]  = std::ceil;
        theFunctionMap1["sin"]   = std::sin;
        theFunctionMap1["cos"]   = std::cos;
        theFunctionMap1["tan"]   = std::tan;
        theFunctionMap1["sinh"]  = std::sinh;
        theFunctionMap1["cosh"]  = std::cosh;
        theFunctionMap1["tanh"]  = std::tanh;
        theFunctionMap1["asin"]  = std::asin;
        theFunctionMap1["acos"]  = std::acos;
        theFunctionMap1["atan"]  = std::atan;
        theFunctionMap1["fact"]  = fact;
        theFunctionMap1["asinh"] = asinh;
        theFunctionMap1["acosh"] = acosh;
        theFunctionMap1["atanh"] = atanh;
        theFunctionMap1["asech"] = asech;
        theFunctionMap1["acsch"] = acsch;
        theFunctionMap1["acoth"] = acoth;
        theFunctionMap1["sech"]  = sech;
        theFunctionMap1["csch"]  = csch;
        theFunctionMap1["coth"]  = coth;
        theFunctionMap1["asec"]  = asec;
        theFunctionMap1["acsc"]  = acsc;
        theFunctionMap1["acot"]  = acot;
        theFunctionMap1["sec"]   = sec;
        theFunctionMap1["csc"]   = csc;
        theFunctionMap1["cot"]   = cot;
        theFunctionMap1["not"]   = libecs::real_not;


        // set ExpressionCompiler::FunctionMap2
        theFunctionMap2["pow"]   = pow;
        theFunctionMap2["and"]   = libecs::real_and;
        theFunctionMap2["or"]    = libecs::real_or;
        theFunctionMap2["xor"]   = libecs::real_xor;
        theFunctionMap2["eq"]    = libecs::real_eq;
        theFunctionMap2["neq"]   = libecs::real_neq;
        theFunctionMap2["gt"]    = libecs::real_gt;
        theFunctionMap2["lt"]    = libecs::real_lt;
        theFunctionMap2["geq"]   = libecs::real_geq;
        theFunctionMap2["leq"]   = libecs::real_leq;
    }
};

template <class CLASS,typename RESULT>
struct ObjectMethodOperand {
    typedef RESULT (CLASS::* MethodPtr)( void ) const;

    const CLASS* theOperand1;
    MethodPtr theOperand2;
};

typedef ObjectMethodOperand<libecs::Process, libecs::Real> ProcessMethod;
typedef ObjectMethodOperand<libecs::System, libecs::Real>  SystemMethod;

// {{{ CompilerHelper
template<typename Tconfig_>
class CompilerHelper: public Tokens
{
public:
    typedef Tconfig_ configuration_type;

public:
    CompilerHelper(
        StringCref anExpression,
        Assembler& anAssembler,
        ErrorReporter& anErrorReporter,
        PropertyAccess& aPropertyAccess,
        EntityResolver& anEntityResolver,
        VariableReferenceResolver& aVarRefResolver)
        : theExpression( anExpression ),
          theAssembler( anAssembler ),
          theErrorReporter( anErrorReporter ),
          thePropertyAccess( aPropertyAccess ),
          theEntityResolver( anEntityResolver ),
          theVarRefResolver( aVarRefResolver )
    {
    }

    void compile();

protected:
    void compileTree( TreeIterator const& aTreeIterator );

    void compileSystemProperty(
        TreeIterator const& aTreeIterator,
        SystemPtr aSystemPtr, const String aMethodName );

private:
    StringCref theExpression;
    PropertyAccess& thePropertyAccess;
    EntityResolver& theEntityResolver;
    VariableReferenceResolver& theVarRefResolver;
    ErrorReporter& theErrorReporter;
    Assembler& theAssembler;
    static configuration_type theConfig;
};

template<typename Tconfig_>
Tconfig_ CompilerHelper<Tconfig_>::theConfig;

template<typename Tconfig_>
void CompilerHelper<Tconfig_>::compile()
{
    if ( theExpression.length() == 0 ) {
        theErrorReporter(
            "UnexpectedError",
             "Expression is empty"
        );
    }

    tree_parse_info<> info(
        ast_parse( theExpression.c_str(), theConfig.theGrammar, space_p ) );

    if ( !info.full ) {
        theErrorReporter(
            "UnexpectedError",
             String( "Parse error: " ) + theExpression 
        );
        return;
    }

    compileTree( info.trees.begin() );
    // place RET at the tail.
    theAssembler.appendInstruction( Instruction<RET>() );
}

/**
   This function is ExpressionCompiler subclass member function.
   This member function evaluates AST tree and makes binary codes.
*/
template<typename Tconfig_> void
CompilerHelper<Tconfig_>::compileTree( TreeIterator const& aTreeIterator )
{
    switch ( aTreeIterator->value.id().to_long() ) {
    case FLOAT:
        {
            BOOST_ASSERT( aTreeIterator->children.size() == 0 );

            const String aFloatString( aTreeIterator->value.begin(),
                                       aTreeIterator->value.end() );

            const Real aFloatValue = stringCast<Real>( aFloatString );

            theAssembler.appendInstruction( Instruction<PUSH_REAL>( aFloatValue ) );
        }
        break;

    case INTEGER:
        {
            BOOST_ASSERT( aTreeIterator->children.size() == 0 );

            const String anIntegerString( aTreeIterator->value.begin(),
                                          aTreeIterator->value.end() );

            const Real anIntegerValue = stringCast<Real>( anIntegerString );

            theAssembler.appendInstruction( Instruction<PUSH_REAL>( anIntegerValue ) );
        }
        break;

    case EXPONENT:
        {
            BOOST_ASSERT( aTreeIterator->value.begin()[ 0 ] == 'E' ||
                    aTreeIterator->value.begin()[ 0 ] == 'e' );

            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );

            const String aBaseString( aChildTreeIterator->value.begin(),
                                      aChildTreeIterator->value.end() );

            const String
            anExponentString( ( aChildTreeIterator + 1 )->value.begin(),
                              ( aChildTreeIterator + 1 )->value.end() );

            const Real aBaseValue = stringCast<Real>( aBaseString );

            if ( anExponentString != "-") {
                const Real
                anExponentValue = stringCast<Real>( anExponentString );

                theAssembler.appendInstruction( Instruction<PUSH_REAL>
                  ( aBaseValue * pow( 10, anExponentValue ) ) );
            } else {
                const String
                anExponentString1( ( aChildTreeIterator + 2 )->value.begin(),
                                   ( aChildTreeIterator + 2 )->value.end() );

                const Real
                anExponentValue = stringCast<Real>( anExponentString1 );

                theAssembler.appendInstruction(
                  Instruction<PUSH_REAL>
                  ( aBaseValue * pow( 10, -anExponentValue ) ) );
            }
        }
        break;

    case CALL_FUNC:
        {
            TreeMatch::container_t::size_type
            aChildTreeSize( aTreeIterator->children.size() );

            const String aFunctionString( aTreeIterator->value.begin(),
                                          aTreeIterator->value.end() );


            BOOST_ASSERT( aChildTreeSize != 0 );

            FunctionMap1Iterator aFunctionMap1Iterator;
            FunctionMap2Iterator aFunctionMap2Iterator;


            if ( aChildTreeSize == 1 ) {
                aFunctionMap1Iterator =
                    theConfig.theFunctionMap1.find( aFunctionString );

                TreeIterator const&
                aChildTreeIterator( aTreeIterator->children.begin() );


                if ( aChildTreeIterator->value.id() == INTEGER ||
                        aChildTreeIterator->value.id() == FLOAT ) {
                    const String
                    anArgumentString( aChildTreeIterator->value.begin(),
                                      aChildTreeIterator->value.end() );

                    const Real
                    anArgumentValue = stringCast<Real>( anArgumentString );

                    if ( aFunctionMap1Iterator != theConfig.theFunctionMap1.end() ) {
                        theAssembler.appendInstruction( Instruction<PUSH_REAL>
                          ( (*aFunctionMap1Iterator->second)
                            ( anArgumentValue ) ) );
                    } else {
                        aFunctionMap2Iterator =
                            theConfig.theFunctionMap2.find( aFunctionString );

                        if ( aFunctionMap2Iterator != theConfig.theFunctionMap2.end() ) {
                            theErrorReporter(
                                "UnexpectedError",
                                String( "Too few arguments for function: " )
                                + aFunctionString
                            );
                        } else {
                            theErrorReporter(
                                "NotFound",
                                String( "No such function: " )
                                + aFunctionString
                            );
                        }
                    }
                } else {
                    compileTree( aChildTreeIterator );

                    if ( aFunctionMap1Iterator != theConfig.theFunctionMap1.end() ) {
                        theAssembler.appendInstruction(
                            Instruction<CALL_FUNC1>(
                                aFunctionMap1Iterator->second ) );
                    } else {
                        aFunctionMap2Iterator =
                            theConfig.theFunctionMap2.find( aFunctionString );

                        if ( aFunctionMap2Iterator != theConfig.theFunctionMap2.end() ) {
                            theErrorReporter(
                                "UnexpectedError",
                                String( "Too few arguments for function: " )
                                + aFunctionString
                            );
                        } else {
                            theErrorReporter(
                                "NotFound",
                                String( "No such function: " )
                                + aFunctionString
                            );
                        }
                    }
                }
            } else if ( aChildTreeSize == 2 ) {
                TreeIterator const&
                aChildTreeIterator( aTreeIterator->children.begin() );

                compileTree( aChildTreeIterator );
                compileTree( aChildTreeIterator + 1 );

                aFunctionMap2Iterator =
                    theConfig.theFunctionMap2.find( aFunctionString );

                if ( aFunctionMap2Iterator != theConfig.theFunctionMap2.end() ) {
                    theAssembler.appendInstruction(
                        Instruction<CALL_FUNC2>(
                            aFunctionMap2Iterator->second ) );
                } else {
                    aFunctionMap1Iterator =
                        theConfig.theFunctionMap1.find( aFunctionString );

                    if ( aFunctionMap1Iterator != theConfig.theFunctionMap1.end() ) {
                        theErrorReporter(
                            "UnexpectedError",
                            String( "Too many arguments for function: " )
                            + aFunctionString
                        );
                    } else {
                        theErrorReporter(
                            "NotFound",
                            String( "No such function: " )
                            + aFunctionString
                        );
                    }
                }
            } else {
                theErrorReporter(
                    "UnexpectedError",
                    String( "Too many arguments for function: " )
                    + aFunctionString
                );
            }

        }
        break;

    case SYSTEM_FUNC:
        {
            BOOST_ASSERT( aTreeIterator->children.size() >= 3 );
            TreeMatch::container_t::size_type
            aChildTreeSize( aTreeIterator->children.size() );

            TreeIterator const& aChildTreeIterator(
                aTreeIterator->children.begin() );

            const String aClassString( aChildTreeIterator->value.begin(),
                                       aChildTreeIterator->value.end() );

            BOOST_ASSERT( aTreeIterator->value.begin()[ 0 ] == '.' );

            EntityPtr anEntityPtr( theEntityResolver[ aClassString ] );
            if ( !anEntityPtr ) {
                theErrorReporter(
                    "NotFound",
                    String( "No such variable reference: " ) + aClassString
                );
                return;
            }

            SystemPtr const aSystemPtr( anEntityPtr->getSuperSystem() );

            const String aMethodName(
                ( aChildTreeIterator+aChildTreeSize - 1 )->value.begin(),
                ( aChildTreeIterator+aChildTreeSize - 1 )->value.end() );

            compileSystemProperty( aChildTreeIterator + 1,
                                   aSystemPtr,
                                   aMethodName );
        }
        break;

    case VARIABLE:
        {
            BOOST_ASSERT( aTreeIterator->value.begin()[ 0 ] == '.' );

            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );

            const String aVariableReferenceString(
                aChildTreeIterator->value.begin(),
                aChildTreeIterator->value.end()
            );

            const String aVariableReferenceMethodString(
                ( aChildTreeIterator+1 )->value.begin(),
                ( aChildTreeIterator+1 )->value.end()
            );

            const VariableReference* aVariableReferencePtr(
                theVarRefResolver[ aVariableReferenceString ]);
            if ( !aVariableReferencePtr ) {
                theErrorReporter(
                    "NotFound",
                    String( "No such variable reference: " )
                    + aVariableReferenceString
                );
                return;
            }

            try {
                theAssembler.appendVariableReferenceMethodInstruction(
                    const_cast<VariableReference*>( aVariableReferencePtr ),
                    aVariableReferenceMethodString );
            } catch ( const NotFound& e ) {
                theErrorReporter(
                    "NotFound",
                    e.what()
                );
            }
        }
        break;

    case IDENTIFIER:
        {
            BOOST_ASSERT( aTreeIterator->children.size() == 0 );

            const String anIdentifierString( aTreeIterator->value.begin(),
                                             aTreeIterator->value.end() );

            do {
                ConstantMapIterator aConstantMapIterator(
                    theConfig.theConstantMap.find( anIdentifierString ) );
                if ( aConstantMapIterator != theConfig.theConstantMap.end() ) {
                    theAssembler.appendInstruction(
                        Instruction<PUSH_REAL>( aConstantMapIterator->second ) );
                    break;
                }

                const Real* prop( thePropertyAccess[ anIdentifierString ]);
                if (prop) {
                    theAssembler.appendInstruction( Instruction<LOAD_REAL>(prop) );
                    break;
                }

                theErrorReporter(
                    "NoSlot",
                    String( "No such property slot: " )
                    + anIdentifierString
                );
                return;
            } while (0);
        }
        break;

    case NEGATIVE:
        {
            BOOST_ASSERT( aTreeIterator->value.begin()[ 0 ] == '-' );

            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );


            if ( aChildTreeIterator->value.id() == INTEGER ||
                    aChildTreeIterator->value.id() == FLOAT ) {
                const String
                aValueString( aChildTreeIterator->value.begin(),
                              aChildTreeIterator->value.end() );

                const Real
                value = stringCast<Real>( aValueString );

                theAssembler.appendInstruction( Instruction<PUSH_REAL>( -value ) );
            } else {
                compileTree( aChildTreeIterator );

                theAssembler.appendInstruction( Instruction<NEG>() );
            }
        }
        break;

    case POWER:
        {
            BOOST_ASSERT(aTreeIterator->children.size() == 2);

            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );


            if ( ( aChildTreeIterator->value.id() == INTEGER ||
                    aChildTreeIterator->value.id() == FLOAT ) &&
                    ( (aChildTreeIterator+1)->value.id() ==INTEGER ||
                      (aChildTreeIterator+1)->value.id() == FLOAT ) ) {

                const String
                anArgumentString1( aChildTreeIterator->value.begin(),
                                   aChildTreeIterator->value.end() );

                const String
                anArgumentString2( ( aChildTreeIterator+1 )->value.begin(),
                                   ( aChildTreeIterator+1 )->value.end() );

                const Real
                anArgumentValue1 = stringCast<Real>( anArgumentString1 );
                const Real
                anArgumentValue2 = stringCast<Real>( anArgumentString2 );


                if ( aTreeIterator->value.begin()[ 0 ] == '^' ) {
                    theAssembler.appendInstruction(
                        Instruction<PUSH_REAL>(
                            pow( anArgumentValue1, anArgumentValue2 ) ) );
                }
                theErrorReporter(
                    "UnexpectedError",
                    String( "Invalid operation: " )
                    + String( aTreeIterator->value.begin(),
                        aTreeIterator->value.end() )
                );
            } else {
                compileTree( aTreeIterator->children.begin() );
                compileTree( aTreeIterator->children.begin() + 1 );

                if ( aTreeIterator->value.begin()[ 0 ] == '^' ) {
                    RealFunc2 aPowFunc( theConfig.theFunctionMap2.find( "pow" )->second );
                    theAssembler.appendInstruction( Instruction<CALL_FUNC2>( aPowFunc ) );
                }
                theErrorReporter(
                    "UnexpectedError",
                    String( "Invalid operation: " )
                    + String( aTreeIterator->value.begin(),
                        aTreeIterator->value.end() )
                );
            }
        }
        break;

    case TERM:
        {
            BOOST_ASSERT( aTreeIterator->children.size() == 2 );

            TreeIterator const& aChildTreeIterator(
                    aTreeIterator->children.begin() );

            if ( ( aChildTreeIterator->value.id() == INTEGER ||
                    aChildTreeIterator->value.id() == FLOAT ) &&
                    ( (aChildTreeIterator+1)->value.id() ==INTEGER ||
                      (aChildTreeIterator+1)->value.id() == FLOAT ) ) {

                const String aTerm1String( aChildTreeIterator->value.begin(),
                                           aChildTreeIterator->value.end() );

                const String
                aTerm2String( ( aChildTreeIterator+1 )->value.begin(),
                              ( aChildTreeIterator+1 )->value.end() );

                const Real aTerm1Value = stringCast<Real>( aTerm1String );
                const Real aTerm2Value = stringCast<Real>( aTerm2String );


                if ( aTreeIterator->value.begin()[ 0 ] == '*' ) {
                    theAssembler.appendInstruction(
                        Instruction<PUSH_REAL>( aTerm1Value * aTerm2Value ) );
                }
                else if ( aTreeIterator->value.begin()[ 0 ] == '/' ) {
                    theAssembler.appendInstruction(
                        Instruction<PUSH_REAL>( aTerm1Value / aTerm2Value ) );
                }
                else {
                    theErrorReporter(
                        "UnexpectedError",
                        String( "Invalid operation: " )
                        + String( aTreeIterator->value.begin(),
                            aTreeIterator->value.end() )
                    );
                }
            } else {
                compileTree( aChildTreeIterator );
                compileTree( aChildTreeIterator + 1 );

                if ( aTreeIterator->value.begin()[ 0 ] == '*' ) {
                    theAssembler.appendInstruction( Instruction<MUL>() );
                }
                else if ( aTreeIterator->value.begin()[ 0 ] == '/' ) {
                    theAssembler.appendInstruction( Instruction<DIV>() );
                }
                else {
                    theErrorReporter(
                        "UnexpectedError",
                        String( "Invalid operation: " )
                        + String( aTreeIterator->value.begin(),
                            aTreeIterator->value.end() )
                    );
                }
            }
        }
        break;

    case EXPRESSION:
        {
            BOOST_ASSERT( aTreeIterator->children.size() == 2 );

            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );


            if ( ( aChildTreeIterator->value.id() == INTEGER ||
                    aChildTreeIterator->value.id() == FLOAT ) &&
                    ( (aChildTreeIterator+1)->value.id() ==INTEGER ||
                      (aChildTreeIterator+1)->value.id() == FLOAT ) ) {
                const String aTerm1String( aChildTreeIterator->value.begin(),
                                           aChildTreeIterator->value.end() );

                const String aTerm2String(
                    ( aChildTreeIterator+1 )->value.begin(),
                    ( aChildTreeIterator+1 )->value.end() );

                const Real aTerm1Value = stringCast<Real>( aTerm1String );
                const Real aTerm2Value = stringCast<Real>( aTerm2String );

                if (aTreeIterator->value.begin()[ 0 ] == '+') {
                    theAssembler.appendInstruction(
                        Instruction<PUSH_REAL>( aTerm1Value + aTerm2Value ) );
                }
                else if (aTreeIterator->value.begin()[ 0 ] == '-') {
                    theAssembler.appendInstruction(
                        Instruction<PUSH_REAL>( aTerm1Value - aTerm2Value ) );
                }
                else {
                    theErrorReporter(
                        "UnexpectedError",
                        String( "Invalid operation: " )
                        + String( aTreeIterator->value.begin(),
                            aTreeIterator->value.end() )
                    );
                }
            } else {
                compileTree( aChildTreeIterator );
                compileTree( aChildTreeIterator + 1 );


                if (aTreeIterator->value.begin()[0] == '+') {
                    theAssembler.appendInstruction( Instruction<ADD>() );
                }
                else if (aTreeIterator->value.begin()[0] == '-') {
                    theAssembler.appendInstruction( Instruction<SUB>() );
                }
                else {
                    theErrorReporter(
                        "UnexpectedError",
                        String( "Invalid operation: " )
                        + String( aTreeIterator->value.begin(),
                            aTreeIterator->value.end() )
                    );
                }
            }
        }
        break;

    default:
        theErrorReporter(
            "UnexpectedError",
            "syntax error."
        );
    }
}

template<typename Tconfig_> void
CompilerHelper<Tconfig_>::compileSystemProperty(
    TreeIterator const& aTreeIterator,
    SystemPtr aSystemPtr, const String aMethodName )
{
    TreeIterator const&
    aChildTreeIterator( aTreeIterator->children.begin() );

    const String aChildString( aChildTreeIterator->value.begin(),
                               aChildTreeIterator->value.end() );

    BOOST_ASSERT( aTreeIterator->value.begin()[ 0 ] == '.' );

    if ( aChildString == "getSuperSystem" ) {
        theAssembler.appendSystemMethodInstruction( aSystemPtr, aMethodName );
    } else if ( aChildString == "." ) {
        SystemPtr theSystemPtr( aSystemPtr->getSuperSystem() );

        compileSystemProperty( aChildTreeIterator, theSystemPtr, aMethodName );
    } else {
        theErrorReporter(
            "UnexpectedError",
            String( "Could not parse system function: " )
            + aChildString
        );
    }
}
// }}}

const Code*
ExpressionCompiler::compileExpression( StringCref anExpression )
{
    Code* code = new Code();
    Assembler anAssembler( code );
    CompilerHelper<CompilerConfig> helper(
            anExpression, anAssembler, theErrorReporter, thePropertyAccess,
            theEntityResolver, theVarRefResolver );
    try {
        helper.compile();
    } catch (const std::exception& e ) {
        delete code;
        throw e;
    }
    return code;
}

} } // namespace libecs::scripting
