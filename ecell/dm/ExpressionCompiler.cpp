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

#include "ExpressionCompiler.hpp"

namespace scripting
{

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

class CompileGrammar: public grammar<CompileGrammar>
{
public:
    enum GrammarType {
        GROUP = 1,
        INTEGER,
        FLOAT,
        NEGATIVE,
        EXPONENT,
        FACTOR,
        POWER,
        TERM,
        EXPRESSION,
        VARIABLE,
        CALL_FUNC,
        SYSTEM_FUNC,
        SYSTEM_PROPERTY,
        IDENTIFIER,
        CONSTANT,
    };

    template <typename ScannerT>
    struct definition {
#define leafNode( str ) leaf_node_d[lexeme_d[str]]
#define rootNode( str ) root_node_d[lexeme_d[str]]

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

            /**system_func = identifier >> discard_node_d[ ch_p('.') ] >>
              +( rootNode( +( alpha_p | ch_p('_') ) ) >>
                 discard_node_d[ ch_p('(') ] >>
                 discard_node_d[ ch_p(')') ] >>
                 discard_node_d[ ch_p('.') ] ) >>
                 identifier;*/

            system_func = identifier >> system_property >> rootNode( ch_p('.') ) >> identifier;

            system_property = +( rootNode( ch_p('.') ) >>
                                 leafNode( +( alpha_p | ch_p('_') ) ) >>
                                 discard_node_d[ ch_p('(') ] >>
                                 discard_node_d[ ch_p(')') ] );


            ///////////////////////////////////////////////////
            //                                               //
            //      This syntax is made such dirty syntax    //
            //      by the bug of Spirit                     //
            //                                               //
            ///////////////////////////////////////////////////

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


static ConstantMap  theConstantMap;
static FunctionMap1 theFunctionMap1;
static FunctionMap2 theFunctionMap2;
static CompileGrammar theGrammer;

template <class CLASS,typename RESULT>
struct ObjectMethodOperand {
    //typedef boost::mem_fn< RESULT, CLASS > MethodType;
    typedef RESULT (CLASS::* MethodPtr)( void ) const;

    const CLASS* theOperand1;
    MethodPtr theOperand2;
};

typedef ObjectMethodOperand<libecs::Process, libecs::Real> ProcessMethod;
typedef ObjectMethodOperand<libecs::System, libecs::Real>  SystemMethod;

// {{{ CompilerHelper
class CompilerHelper
{
public:
    inline CompilerHelper( ProcessCref aProcess, PropertyMapRef aPropertyMap,
        StringCref anExpression )
        : theProcess( aProcess ), thePropertyMap( aPropertyMap ),
          theExpression( anExpression ), theCode( 0 )
    {
    }

    void compile();

    Code* getResult()
    {
        return theCode;
    }

protected:
    template < class INSTRUCTION >
    void appendInstruction( const INSTRUCTION& anInstruction )
    {
        Code::size_type aCodeSize( theCode->size() );
        theCode->resize( aCodeSize + sizeof( INSTRUCTION ) );
        // XXX: hackish!!!
        new (&(*theCode)[aCodeSize]) INSTRUCTION( anInstruction );
    }

    void
    appendVariableReferenceMethodInstruction(
            libecs::VariableReferencePtr aVariableReference,
            libecs::StringCref aMethodName );

    void
    appendSystemMethodInstruction( libecs::SystemPtr aSystemPtr,
                                   libecs::StringCref aMethodName );

    void compileTree( TreeIterator const& aTreeIterator );

    void compileSystemProperty(
        TreeIterator const& aTreeIterator,
        SystemPtr aSystemPtr, const String aMethodName );

    void throw_exception( String anExceptionType,
                                         String anExceptionString )
    {
        if ( anExceptionType == "UnexpeptedError" ) {
            THROW_EXCEPTION( UnexpectedError, anExceptionString );
        } else if ( anExceptionType == "NoSlot" ) {
            THROW_EXCEPTION( NoSlot, anExceptionString );
        } else if ( anExceptionType == "NotFound" ) {
            THROW_EXCEPTION( NotFound, anExceptionString );
        } else {
            THROW_EXCEPTION( UnexpectedError, anExceptionString );
        }
    }

private:
    StringCref theExpression;
    ProcessCref theProcess;
    PropertyMapRef thePropertyMap;
    Code* theCode;
};

void CompilerHelper::compile()
{
    if ( theExpression.length() == 0 ) {
        THROW_EXCEPTION( UnexpectedError,
                         "Expression is empty\nClass : " +
                         String( theProcess.getClassName() ) +
                         "\nProcessID : " + String( theProcess.getID() ) );
    }

    tree_parse_info<> info(
        ast_parse( theExpression.c_str(), theGrammer, space_p ) );

    if ( !info.full ) {
        THROW_EXCEPTION( UnexpectedError,
                         "Parse error in the expression.\nExpression : "
                         + theExpression + "\nClass : "
                         + String( theProcess.getClassName() )
                         + "\nProcessID : "
                         + String( theProcess.getID() ) );
    }

    theCode = new Code();
    try {
        compileTree( info.trees.begin() );
        // place RET at the tail.
        appendInstruction( Instruction<RET>() );
    } catch (const libecs::Exception& e) {
        delete theCode;
        theCode = 0;
        throw e;
    }
}


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
CompilerHelper::appendVariableReferenceMethodInstruction(
        VariableReferencePtr aVariableReference, StringCref aMethodName )
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
    }

    /**else if( str_child2 == "Fixed" ){
       aCode.push_back(
       new OBJECT_METHOD_REAL( aVariableReference,
       &libecs::VariableReference::isFixed ) );
       }*/

    else {
        THROW_EXCEPTION
        ( NotFound,
          "VariableReference attribute [" +
          aMethodName + "] not found." );
    }


}

void
CompilerHelper::
appendSystemMethodInstruction( SystemPtr aSystemPtr,
                               StringCref aMethodName )
{
    if ( aMethodName == "Size" ) {
        APPEND_OBJECT_METHOD_REAL( aSystemPtr, System, getSize );
    } else if ( aMethodName == "SizeN_A" ) {
        APPEND_OBJECT_METHOD_REAL( aSystemPtr, System, getSizeN_A );
    } else {
        THROW_EXCEPTION
        ( NotFound,
          "System attribute [" +
          aMethodName + "] not found." );
    }

}

#undef APPEND_OBJECT_METHOD_REAL
#undef APPEND_OBJECT_METHOD_INTEGER

/**
   This function is ExpressionCompiler subclass member function.
   This member function evaluates AST tree and makes binary codes.
*/

void
CompilerHelper::compileTree( TreeIterator const& aTreeIterator )
{
    /**
       compile AST
    */

    switch ( aTreeIterator->value.id().to_long() ) {
        /**
        Floating Grammar compile
        */

    case CompileGrammar::FLOAT : {
        assert( aTreeIterator->children.size() == 0 );

        const String aFloatString( aTreeIterator->value.begin(),
                                   aTreeIterator->value.end() );

        const Real aFloatValue = stringCast<Real>( aFloatString );

        appendInstruction( Instruction<PUSH_REAL>( aFloatValue ) );

        return;
    }

    /**
       Integer Grammar compile
    */

    case CompileGrammar::INTEGER : {
        assert( aTreeIterator->children.size() == 0 );

        const String anIntegerString( aTreeIterator->value.begin(),
                                      aTreeIterator->value.end() );

        const Real anIntegerValue = stringCast<Real>( anIntegerString );

        appendInstruction( Instruction<PUSH_REAL>( anIntegerValue ) );

        return;

    }

    /**
       Grammar compile
    */

    case CompileGrammar::EXPONENT: {
        assert( *aTreeIterator->value.begin() == 'E' ||
                *aTreeIterator->value.begin() == 'e' );

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

            appendInstruction( Instruction<PUSH_REAL>
              ( aBaseValue * pow( 10, anExponentValue ) ) );
        } else {
            const String
            anExponentString1( ( aChildTreeIterator + 2 )->value.begin(),
                               ( aChildTreeIterator + 2 )->value.end() );

            const Real
            anExponentValue = stringCast<Real>( anExponentString1 );

            appendInstruction(
              Instruction<PUSH_REAL>
              ( aBaseValue * pow( 10, -anExponentValue ) ) );
        }

        return;
    }



    /**
    Call_Func Grammar compile
    */

    case CompileGrammar::CALL_FUNC : {
        TreeMatch::container_t::size_type
        aChildTreeSize( aTreeIterator->children.size() );

        const String aFunctionString( aTreeIterator->value.begin(),
                                      aTreeIterator->value.end() );


        assert( aChildTreeSize != 0 );

        FunctionMap1Iterator aFunctionMap1Iterator;
        FunctionMap2Iterator aFunctionMap2Iterator;


        if ( aChildTreeSize == 1 ) {
            aFunctionMap1Iterator =
                theFunctionMap1.find( aFunctionString );

            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );


            if ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
                    aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) {
                const String
                anArgumentString( aChildTreeIterator->value.begin(),
                                  aChildTreeIterator->value.end() );

                const Real
                anArgumentValue = stringCast<Real>( anArgumentString );

                if ( aFunctionMap1Iterator != theFunctionMap1.end() ) {
                    appendInstruction( Instruction<PUSH_REAL>
                      ( (*aFunctionMap1Iterator->second)
                        ( anArgumentValue ) ) );
                } else {
                    aFunctionMap2Iterator =
                        theFunctionMap2.find( aFunctionString );

                    if ( aFunctionMap2Iterator != theFunctionMap2.end() ) {
                        throw_exception( "UnexpectedError",
                          "[ " + aFunctionString +
                          " ] function. Too few arguments\nProcessID : "
                          + theProcess.getID() );
                    } else {
                        throw_exception( "NoSlot",
                          "[ " + aFunctionString +
                          String( " ] : No such function." ) +
                          "\nProcessID : " + theProcess.getID() );
                    }
                }
            } else {
                compileTree( aChildTreeIterator );

                if ( aFunctionMap1Iterator != theFunctionMap1.end() ) {
                    appendInstruction(
                        Instruction<CALL_FUNC1>(
                            aFunctionMap1Iterator->second ) );
                } else {
                    aFunctionMap2Iterator =
                        theFunctionMap2.find( aFunctionString );

                    if ( aFunctionMap2Iterator != theFunctionMap2.end() ) {
                        throw_exception( "UnexpectedError",
                          "[ " + aFunctionString +
                          " ] function. Too few arguments\nProcessID : "
                          + theProcess.getID() );
                    } else {
                        throw_exception( "NoSlot",
                          "[ " + aFunctionString +
                          String( " ] : No such function." ) +
                          "\nProcessID : " + theProcess.getID() );
                    }
                }
            }
        }

        else if ( aChildTreeSize == 2 ) {
            TreeIterator const&
            aChildTreeIterator( aTreeIterator->children.begin() );

            compileTree( aChildTreeIterator );
            compileTree( aChildTreeIterator + 1 );

            aFunctionMap2Iterator =
                theFunctionMap2.find( aFunctionString );

            if ( aFunctionMap2Iterator != theFunctionMap2.end() ) {
                appendInstruction(
                    Instruction<CALL_FUNC2>(
                        aFunctionMap2Iterator->second ) );
            } else {
                aFunctionMap1Iterator =
                    theFunctionMap1.find( aFunctionString );

                if ( aFunctionMap1Iterator != theFunctionMap1.end() ) {
                    throw_exception( "UnexpectedError",
                      "[ " + aFunctionString +
                      " ] function. Too many arguments\nProcessID : " +
                      theProcess.getID() );
                } else {
                    throw_exception( "NotFound",
                      "[ " + aFunctionString +
                      String( " ] : No such function." ) +
                      "\nProcessID : " +
                      theProcess.getID() );
                }
            }
        }

        else {
            throw_exception( "UnexpectedError",
              " : Too many arguments\nProcessID : " +
              theProcess.getID() );
        }

        return;
    }


    /**
       System_Func Grammar compile
    */

    case CompileGrammar::SYSTEM_FUNC : {
        assert( aTreeIterator->children.size() >= 3 );
        TreeMatch::container_t::size_type
        aChildTreeSize( aTreeIterator->children.size() );

        TreeIterator const&
        aChildTreeIterator( aTreeIterator->children.begin() );

        const String aClassString( aChildTreeIterator->value.begin(),
                                   aChildTreeIterator->value.end() );

        assert( *aTreeIterator->value.begin() == '.' );

        if ( aClassString == "self" ) { // Process Class
            SystemPtr aSystemPtr( theProcess.getSuperSystem() );

            const String aMethodName
            ( ( aChildTreeIterator+aChildTreeSize-1 )->value.begin(),
              ( aChildTreeIterator+aChildTreeSize-1 )->value.end() );

            compileSystemProperty( aChildTreeIterator+1,
                                   aSystemPtr,
                                   aMethodName );
        }

        else { // VariableReference Class
            VariableReferenceCref aVariableReference( theProcess.
                                getVariableReference( aClassString ) );

            SystemPtr const aSystemPtr( aVariableReference.getSuperSystem() );

            const String aMethodName
            ( ( aChildTreeIterator+aChildTreeSize-1 )->value.begin(),
              ( aChildTreeIterator+aChildTreeSize-1 )->value.end() );

            compileSystemProperty( aChildTreeIterator+1,
                                   aSystemPtr,
                                   aMethodName );
        }
        return;
    }


    /**
       Variable Grammar compile
    */

    case CompileGrammar::VARIABLE : {
        assert( *aTreeIterator->value.begin() == '.' );

        TreeIterator const&
        aChildTreeIterator( aTreeIterator->children.begin() );

        const String
        aVariableReferenceString( aChildTreeIterator->value.begin(),
                                  aChildTreeIterator->value.end() );

        const String
        aVariableReferenceMethodString
        ( ( aChildTreeIterator+1 )->value.begin(),
          ( aChildTreeIterator+1 )->value.end() );

        VariableReferenceCref
        aVariableReference
        ( theProcess.
          getVariableReference( aVariableReferenceString ) );

        appendVariableReferenceMethodInstruction(
            const_cast<VariableReference*>( &aVariableReference ),
            aVariableReferenceMethodString );

        return;

    }



    /**
       Identifier Grammar compile
    */

    case CompileGrammar::IDENTIFIER : {
        assert( aTreeIterator->children.size() == 0 );

        const String anIdentifierString( aTreeIterator->value.begin(),
                                         aTreeIterator->value.end() );

        ConstantMapIterator aConstantMapIterator;
        PropertyMapIterator aPropertyMapIterator;

        aConstantMapIterator =
            theConstantMap.find( anIdentifierString );
        aPropertyMapIterator =
            thePropertyMap.find( anIdentifierString );


        if ( aConstantMapIterator != theConstantMap.end() ) {
            appendInstruction(
              Instruction<PUSH_REAL>( aConstantMapIterator->second ) );
        }

        else if ( aPropertyMapIterator != thePropertyMap.end() ) {
            appendInstruction( Instruction<LOAD_REAL>
              ( &(aPropertyMapIterator->second) ) );
        }

        else {
            throw_exception( "NoSlot",
              "[ " + anIdentifierString +
              " ] No such Property slot.\nProcessID : "
              + theProcess.getID() );
        }

        return;
    }



    /**
       Negative Grammar compile
    */

    case CompileGrammar::NEGATIVE : {
        assert( *aTreeIterator->value.begin() == '-' );

        TreeIterator const&
        aChildTreeIterator( aTreeIterator->children.begin() );


        if ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
                aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) {
            const String
            aValueString( aChildTreeIterator->value.begin(),
                          aChildTreeIterator->value.end() );

            const Real
            value = stringCast<Real>( aValueString );

            appendInstruction( Instruction<PUSH_REAL>( -value ) );
        } else {
            compileTree( aChildTreeIterator );

            appendInstruction( Instruction<NEG>() );
        }

        return;

    }



    /**
       Power Grammar compile
    */

    case CompileGrammar::POWER : {
        assert(aTreeIterator->children.size() == 2);

        TreeIterator const&
        aChildTreeIterator( aTreeIterator->children.begin() );


        if ( ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
                aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) &&
                ( (aChildTreeIterator+1)->value.id() ==CompileGrammar::INTEGER ||
                  (aChildTreeIterator+1)->value.id() == CompileGrammar::FLOAT ) ) {

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


            if ( *aTreeIterator->value.begin() == '^' ) {
                appendInstruction(
                    Instruction<PUSH_REAL>(
                        pow( anArgumentValue1, anArgumentValue2 ) ) );
            }

            else {
                throw_exception( "UnexpectedError",
                  String( "Invalid operation" ) +
                  "\nProcessID : " + theProcess.getID() );
            }

            return;
        } else {
            compileTree( aTreeIterator->children.begin() );
            compileTree( aTreeIterator->children.begin() + 1 );

            if ( *aTreeIterator->value.begin() == '^' ) {
                RealFunc2 aPowFunc( theFunctionMap2.find( "pow" )->second );
                appendInstruction( Instruction<CALL_FUNC2>( aPowFunc ) );
            }

            else {
                throw_exception( "UnexpectedError",
                  String( "Invalud operation" ) +
                  "\nProcessID : " + theProcess.getID() );
            }

            return;
        }

        return;

    }



    /**
       Term Grammar compile
    */

    case CompileGrammar::TERM : {

        assert(aTreeIterator->children.size() == 2);


        TreeIterator const&
        aChildTreeIterator( aTreeIterator->children.begin() );


        if ( ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
                aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) &&
                ( (aChildTreeIterator+1)->value.id() ==CompileGrammar::INTEGER ||
                  (aChildTreeIterator+1)->value.id() == CompileGrammar::FLOAT ) ) {

            const String aTerm1String( aChildTreeIterator->value.begin(),
                                       aChildTreeIterator->value.end() );

            const String
            aTerm2String( ( aChildTreeIterator+1 )->value.begin(),
                          ( aChildTreeIterator+1 )->value.end() );

            const Real aTerm1Value = stringCast<Real>( aTerm1String );
            const Real aTerm2Value = stringCast<Real>( aTerm2String );


            if (*aTreeIterator->value.begin() == '*') {
                appendInstruction(
                    Instruction<PUSH_REAL>( aTerm1Value * aTerm2Value ) );
            }

            else if (*aTreeIterator->value.begin() == '/') {
                appendInstruction(
                    Instruction<PUSH_REAL>( aTerm1Value / aTerm2Value ) );
            }

            else {
                throw_exception( "UnexpectedError",
                  String( "Invalid operation" ) +
                  "\nProcessID : " + theProcess.getID() );
            }

            return;
        } else {
            compileTree( aChildTreeIterator );
            compileTree( aChildTreeIterator + 1 );

            if (*aTreeIterator->value.begin() == '*') {
                appendInstruction( Instruction<MUL>() );
            }

            else if (*aTreeIterator->value.begin() == '/') {
                appendInstruction( Instruction<DIV>() );
            }

            else {
                throw_exception( "UnexpectedError",
                  String( "Invalid operation" ) +
                  "\nProcessID : " + theProcess.getID() );
            }

            return;
        }

        return;

    }



    /**
       Expression Grammar compile
    */

    case CompileGrammar::EXPRESSION : {

        assert(aTreeIterator->children.size() == 2);

        TreeIterator const&
        aChildTreeIterator( aTreeIterator->children.begin() );


        if ( ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
                aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) &&
                ( (aChildTreeIterator+1)->value.id() ==CompileGrammar::INTEGER ||
                  (aChildTreeIterator+1)->value.id() == CompileGrammar::FLOAT ) ) {
            const String aTerm1String( aChildTreeIterator->value.begin(),
                                       aChildTreeIterator->value.end() );

            const String
            aTerm2String( ( aChildTreeIterator+1 )->value.begin(),
                          ( aChildTreeIterator+1 )->value.end() );

            const Real aTerm1Value = stringCast<Real>( aTerm1String );
            const Real aTerm2Value = stringCast<Real>( aTerm2String );


            if (*aTreeIterator->value.begin() == '+') {
                appendInstruction(
                    Instruction<PUSH_REAL>( aTerm1Value + aTerm2Value ) );
            }

            else if (*aTreeIterator->value.begin() == '-') {
                appendInstruction(
                    Instruction<PUSH_REAL>( aTerm1Value - aTerm2Value ) );
            }

            else {
                throw_exception( "UnexpectedError",
                  String( "Invalid operation" ) +
                  "\nProcessID : " + theProcess.getID() );
            }
        } else {
            compileTree( aChildTreeIterator );
            compileTree( aChildTreeIterator + 1 );


            if (*aTreeIterator->value.begin() == '+') {
                appendInstruction( Instruction<ADD>() );
            }

            else if (*aTreeIterator->value.begin() == '-') {
                appendInstruction( Instruction<SUB>() );
            }

            else {
                throw_exception( "UnexpectedError",
                  String( "Invalid operation" ) +
                  "\nProcessID : " + theProcess.getID() );
            }
        }

        return;

    }

    default : {
        throw_exception( "UnexpectedError",
          "syntax error.\nProcessID : " + theProcess.getID() );

        return;
    }
    }
}


void CompilerHelper::compileSystemProperty(
    TreeIterator const& aTreeIterator,
    SystemPtr aSystemPtr, const String aMethodName )
{
    TreeIterator const&
    aChildTreeIterator( aTreeIterator->children.begin() );

    const String aChildString( aChildTreeIterator->value.begin(),
                               aChildTreeIterator->value.end() );

    assert( *aTreeIterator->value.begin() == '.' );

    if ( aChildString == "getSuperSystem" ) {
        appendSystemMethodInstruction( aSystemPtr, aMethodName );
    } else if ( aChildString == "." ) {
        SystemPtr theSystemPtr( aSystemPtr->getSuperSystem() );

        compileSystemProperty( aChildTreeIterator, theSystemPtr, aMethodName );
    } else {
        throw_exception( "UnexpectedError",
          String( "System function parse error" ) +
          "\nProcessID : " + theProcess.getID() );
    }
}
// }}}

const Code*
ExpressionCompiler::compileExpression( StringCref anExpression )
{
    CompilerHelper helper( theProcess, thePropertyMap, anExpression );
    helper.compile();
    return helper.getResult();
}


void ExpressionCompiler::populateMap()
{
    if ( theConstantMap.empty() ) {
        // set ConstantMap
        theConstantMap["true"]  = 1.0;
        theConstantMap["false"] = 0.0;
        theConstantMap["pi"]    = M_PI;
        theConstantMap["NaN"]   = std::numeric_limits<Real>::quiet_NaN();
        theConstantMap["INF"]   = std::numeric_limits<Real>::infinity();
        theConstantMap["N_A"]   = N_A;
        theConstantMap["exp"]   = M_E;
    }

    if ( theFunctionMap1.empty() ) {
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
}



void
ExpressionCompiler::throw_exception( String anExceptionType,
                                     String anExceptionString )
{
    if ( anExceptionType == "UnexpeptedError" ) {
        THROW_EXCEPTION( UnexpectedError, anExceptionString );
    } else if ( anExceptionType == "NoSlot" ) {
        THROW_EXCEPTION( NoSlot, anExceptionString );
    } else if ( anExceptionType == "NotFound" ) {
        THROW_EXCEPTION( NotFound, anExceptionString );
    } else {
        THROW_EXCEPTION( UnexpectedError, anExceptionString );
    }
}


} // namespace scripting
