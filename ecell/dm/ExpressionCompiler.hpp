//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2004 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// authors:
//     Kouichi Takahashi
//     Tatsuya Ishida
//
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
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
#include "libecs/ObjectMethodProxy.hpp"

using namespace boost::spirit;

USE_LIBECS;


DECLARE_ASSOCVECTOR
( String, Real(*)(Real), std::less<const String>, FunctionMap1 );
DECLARE_ASSOCVECTOR
( String, Real(*)(Real,Real), std::less<const String>, FunctionMap2 );
DECLARE_ASSOCVECTOR
( String, Real, std::less<const String>, ConstantMap );
DECLARE_ASSOCVECTOR
( String, Real, std::less<const String>, PropertyMap );


class ExpressionCompiler
{
public:

  DECLARE_VECTOR( unsigned char, Code );

  DECLARE_VECTOR( char, CharVector );

  // possible operand types:
  typedef libecs::Real    Real;
  typedef Real*           RealPtr;
  typedef libecs::Integer Integer;
  typedef void*           Pointer;

  typedef SystemPtr 
  (libecs::VariableReference::* VariableReferenceSystemMethodPtr)() const;
  typedef SystemPtr  (libecs::Process::* ProcessMethodPtr)() const;
  typedef const Real (libecs::System::* SystemMethodPtr)() const;

  typedef Real (*RealFunc0)();
  typedef Real (*RealFunc1)( Real );
  typedef Real (*RealFunc2)( Real, Real );


  typedef ObjectMethodProxy<Real> RealObjectMethodProxy;
  typedef ObjectMethodProxy<Integer> IntegerObjectMethodProxy;
  //  VariableReferenceMethodProxy;

  typedef void (*InstructionAppender)( CodeRef );


  enum Opcode// the order of items is optimized. don't change.
    {
      ADD = 0  // no arg
      , SUB    // no arg
      , MUL    // no arg
      , DIV    // no arg
      , CALL_FUNC2 // RealFunc2
      // Those instructions above are candidates of stack operations folding
      // in the stack machine, and grouped here to optimize the switch(). 

      , CALL_FUNC1 // RealFunc1
      //, CALL_FUNC0 // RealFunc0
      , NEG    // no arg
      //      , PUSH_POINTER // Pointer
      , PUSH_REAL   // Real
      , LOAD_REAL  // Real*
      , OBJECT_METHOD_REAL //VariableReferencePtr, VariableReferenceMethodPtr
      , OBJECT_METHOD_INTEGER // VariableReferenceIntegerMethodPtr

      , RET   // no arg
      , END = RET
    };




  class NoOperand {}; // Null type.

  template <Opcode OPCODE>
  class Opcode2Operand
  {
  public:
    typedef NoOperand     type;
  };


  template <Opcode OPCODE>
  class Opcode2Instruction;


  class InstructionHead
  {
  public:

    InstructionHead( Opcode anOpcode )
      :
      theOpcode( anOpcode )
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

  template < class OPERAND >
  class InstructionBase
    :
    public InstructionHead
  {
  public:

    DECLARE_TYPE( OPERAND, Operand );
      
    InstructionBase( Opcode anOpcode, OperandCref anOperand )
      :
      InstructionHead( anOpcode ),
      theOperand( anOperand )
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


  /**
     Instruction Class
  */

  template < Opcode OPCODE >
  class Instruction
    :
    public InstructionBase<typename Opcode2Operand<OPCODE>::type>
  {
    
  public:

    typedef typename Opcode2Operand<OPCODE>::type Operand_;
    DECLARE_TYPE( Operand_, Operand );
      
    Instruction( OperandCref anOperand )
      :
      InstructionBase<Operand>( OPCODE, anOperand )
    {
      ; // do nothing
    }

    Instruction()
      :
      InstructionBase<Operand>( OPCODE )
    {
      ; // do nothing
    }

  };


  template <class CLASS,typename RESULT>
  struct ObjectMethodOperand
  {
    //typedef boost::mem_fn< RESULT, CLASS > MethodType;
    typedef RESULT (CLASS::* MethodPtr)( void ) const;

    const CLASS* theOperand1;
    MethodPtr theOperand2;
  };

  typedef ObjectMethodOperand<Process,Real> ProcessMethod;
  typedef ObjectMethodOperand<System,Real>  SystemMethod;


private:
    
  class CompileGrammar 
    : 
    public grammar<CompileGrammar>
  {
  public:
    enum GrammarType
      {
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
	IDENTIFIER,
	CONSTANT,
      };

    template <typename ScannerT>
    struct definition
    {
#define leafNode( str ) leaf_node_d[lexeme_d[str]]
#define rootNode( str ) root_node_d[lexeme_d[str]]
	
      definition( CompileGrammar const& /*self*/ )
      {
	integer     =   leafNode( +digit_p );
	floating    =   leafNode( +digit_p >> ch_p('.') >> +digit_p );

	exponent    =   ( floating | integer ) >>
	  rootNode( ch_p('e') | ch_p('E') ) >>
	  ( ch_p('-') >> integer | 
	    discard_node_d[ ch_p('+') ] >> integer |
	    integer );

	negative    =	  rootNode( ch_p('-') ) >> factor; 

	identifier  =   leafNode( alpha_p >> *( alnum_p | ch_p('_') ) );

	variable    =   identifier >> rootNode( ch_p('.') ) >> identifier;
	
	system_func = identifier >> rootNode( ch_p('.') ) >>
	  +( leafNode( +( alpha_p | ch_p('_') ) ) >>
	     discard_node_d[ ch_p('(') ] >>
	     discard_node_d[ ch_p(')') ] >>
	     discard_node_d[ ch_p('.') ] ) >>
	  identifier;

	///////////////////////////////////////////////////
	//                                               //
	//      This syntax is made such dirty syntax    //
	//      by the bug of Spirit                     //
	//                                               //
	///////////////////////////////////////////////////
            
	//call_func = rootNode( +alpha_p ) >>

	call_func = (	  rootNode( str_p("eq") )
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
	     |  ( rootNode( ch_p('/') ) >> power )
	     |  ( rootNode( ch_p('^') ) >> power ) );
	

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
      rule<ScannerT, PARSER_CONTEXT, parser_tag<FLOAT> >     floating;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<EXPONENT> >     exponent;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<INTEGER> >      integer;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<NEGATIVE> >     negative;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<GROUP> >        group;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<IDENTIFIER> >   identifier;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<CONSTANT> >     constant;
      rule<ScannerT, PARSER_CONTEXT, parser_tag<SYSTEM_FUNC> > system_func;

      rule<ScannerT, PARSER_CONTEXT, parser_tag<EXPRESSION> > const&
      start() const { return expression; }
    };
      
#undef leafNode
#undef rootNode
      
  };
    
public:
    
  ExpressionCompiler( ProcessPtr aProcess, PropertyMapPtr aPropertyMap )
    :
    theProcessPtr( aProcess ),
    thePropertyMapPtr( aPropertyMap )
  {
    if( theConstantMap.empty() == true ||
	theFunctionMap1.empty() == true )
      {
	fillMap();
      }
  }
    

  ~ExpressionCompiler()
  {
    ; // do nothing
  }
    
  typedef char const*         iterator_t;
  typedef tree_match<iterator_t> parse_tree_match_t;
  typedef parse_tree_match_t::tree_iterator TreeIterator;
  //DECLARE_CLASS( TreeIterator );
    
  const Code compileExpression( StringCref anExpression );
    
protected:

  template < class INSTRUCTION >
  static void appendInstruction( Code& aCode, 
				 const INSTRUCTION& anInstruction )
  {
    Code::size_type aCodeSize( aCode.size() );
    aCode.resize( aCodeSize + sizeof( INSTRUCTION ) );
    new (&aCode[aCodeSize]) INSTRUCTION( anInstruction );
  }

  /**template < Opcode OPCODE >
     static void appendSimpleInstruction( Code& aCode )
     {
     appendInstruction( aCode, Instruction<OPCODE>() );
     }*/

  static void 
  appendVariableReferenceMethodInstruction( Code& aCode,
					    VariableReferencePtr
					    aVariableReference,
					    StringCref aMethodName );

  static void 
  appendSystemMethodInstruction( Code& aCode,
				 SystemPtr aSystemPtr,
				 StringCref aMethodName );

private:
    
  static void fillMap();

  void compileTree( TreeIterator const&  i, CodeRef aCode );  

private:
    
  ProcessPtr      theProcessPtr;
  PropertyMapPtr  thePropertyMapPtr;
    
  static ConstantMap        theConstantMap;
  static FunctionMap1       theFunctionMap1;
  static FunctionMap2       theFunctionMap2;

}; // ExpressionCompiler





template <> 
class ExpressionCompiler::InstructionBase<ExpressionCompiler::NoOperand>
  :
  public ExpressionCompiler::InstructionHead
{
public:
    
  InstructionBase( Opcode anOpcode )
    :
    InstructionHead( anOpcode )
  {
    ; // do nothing
  }
    
  InstructionBase( Opcode, const NoOperand& );
    
};




#define SPECIALIZE_OPCODE2OPERAND( OP, OPE )\
  template<> class ExpressionCompiler::Opcode2Operand<ExpressionCompiler::OP>\
  {\
  public:\
    typedef ExpressionCompiler::OPE type;\
  };


    
SPECIALIZE_OPCODE2OPERAND( PUSH_REAL,                Real );
//SPECIALIZE_OPCODE2OPERAND( PUSH_POINTER,             Pointer );
SPECIALIZE_OPCODE2OPERAND( LOAD_REAL,                RealPtr const );
//SPECIALIZE_OPCODE2OPERAND( CALL_FUNC0,           RealFunc0 );
SPECIALIZE_OPCODE2OPERAND( CALL_FUNC1,               RealFunc1 );
SPECIALIZE_OPCODE2OPERAND( CALL_FUNC2,               RealFunc2 );
SPECIALIZE_OPCODE2OPERAND( OBJECT_METHOD_REAL, 
			   RealObjectMethodProxy );
SPECIALIZE_OPCODE2OPERAND( OBJECT_METHOD_INTEGER, 
			   IntegerObjectMethodProxy );

  
#define DEFINE_OPCODE2INSTRUCTION( CODE )\
  template<> class\
    ExpressionCompiler::Opcode2Instruction<ExpressionCompiler::CODE>\
  {\
  public:\
    typedef ExpressionCompiler::Instruction<ExpressionCompiler::CODE> type;\
    typedef type::Operand operandtype;\
  }

      
DEFINE_OPCODE2INSTRUCTION( PUSH_REAL );
//DEFINE_OPCODE2INSTRUCTION( PUSH_POINTER );
DEFINE_OPCODE2INSTRUCTION( NEG );
DEFINE_OPCODE2INSTRUCTION( ADD );
DEFINE_OPCODE2INSTRUCTION( SUB );
DEFINE_OPCODE2INSTRUCTION( MUL );
DEFINE_OPCODE2INSTRUCTION( DIV );
//DEFINE_OPCODE2INSTRUCTION( POW );
DEFINE_OPCODE2INSTRUCTION( LOAD_REAL );
//DEFINE_OPCODE2INSTRUCTION( CALL_FUNC0 );
DEFINE_OPCODE2INSTRUCTION( CALL_FUNC1 );
DEFINE_OPCODE2INSTRUCTION( CALL_FUNC2 );
DEFINE_OPCODE2INSTRUCTION( OBJECT_METHOD_INTEGER );
DEFINE_OPCODE2INSTRUCTION( OBJECT_METHOD_REAL );
DEFINE_OPCODE2INSTRUCTION( RET );



const ExpressionCompiler::Code 
ExpressionCompiler::compileExpression( StringCref anExpression )
{
  Code aCode;
  CompileGrammar aGrammer;
	        
  tree_parse_info<> 
    info( ast_parse( anExpression.c_str(), aGrammer, space_p ) );

  if( info.full )
    {
      compileTree( info.trees.begin(), aCode );
	  
      // place RET at the tail.
      appendInstruction( aCode, Instruction<RET>() );
    }
  else
    {
      THROW_EXCEPTION( UnexpectedError, 
		       "Parse error in the expression. Expression : " +
		       anExpression );
    }
      
  return aCode;
}


void ExpressionCompiler::fillMap()
{

  // set ConstantMap
  theConstantMap["true"]  = 1.0;
  theConstantMap["false"] = 0.0;
  theConstantMap["pi"]    = M_PI;
  theConstantMap["NaN"]   = std::numeric_limits<Real>::quiet_NaN();
  theConstantMap["INF"]   = std::numeric_limits<Real>::infinity();
  theConstantMap["N_A"]   = N_A;
  theConstantMap["exp"]   = M_E;


  // set FunctionMap1
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


  // set FunctionMap2
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

#define APPEND_OBJECT_METHOD_REAL( OBJECT, CLASSNAME, METHODNAME )\
	appendInstruction\
	  ( aCode, \
	    Instruction<OBJECT_METHOD_REAL>\
	    ( RealObjectMethodProxy::\
	      create< CLASSNAME, & CLASSNAME::METHODNAME >\
	      ( OBJECT ) ) ) // \

#define APPEND_OBJECT_METHOD_INTEGER( OBJECT, CLASSNAME, METHODNAME )\
	appendInstruction\
	  ( aCode, \
	    Instruction<OBJECT_METHOD_INTEGER>\
	    ( IntegerObjectMethodProxy::\
	      create< CLASSNAME, & CLASSNAME::METHODNAME >\
	      ( OBJECT ) ) ) // \


void 
ExpressionCompiler::
appendVariableReferenceMethodInstruction( Code& aCode,
					  VariableReferencePtr 
					  aVariableReference,
					  StringCref aMethodName )
{


  if( aMethodName == "MolarConc" )
    {
      APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
				 getMolarConc );
    }
  else if( aMethodName == "NumberConc" )
    {
      APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
				 getNumberConc );
    }
  else if( aMethodName == "Value" )
    {
      APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
				 getValue );
    }
  else if( aMethodName == "Velocity" )
    {
      APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
				 getVelocity );
    }
  else if( aMethodName == "TotalVelocity" )
    {
      APPEND_OBJECT_METHOD_REAL( aVariableReference, VariableReference,
				 getTotalVelocity );
    }
  else if( aMethodName == "Coefficient" )
    {
      APPEND_OBJECT_METHOD_INTEGER( aVariableReference, VariableReference,
				    getCoefficient );
    }

  /**else if( str_child2 == "Fixed" ){
     aCode.push_back(
     new OBJECT_METHOD_REAL( aVariableReference,
     &libecs::VariableReference::isFixed ) );
     }*/

  else
    {
      THROW_EXCEPTION
	( NotFound,
	  "ExpressionCompiler: VariableReference attribute [" +
	  aMethodName + "] not found." ); 
    }


}

void 
ExpressionCompiler::
appendSystemMethodInstruction( Code& aCode,
			       SystemPtr aSystemPtr,
			       StringCref aMethodName )
{
  if( aMethodName == "Size" )
    {
      APPEND_OBJECT_METHOD_REAL( aSystemPtr, System, getSize );
    }
  else if( aMethodName == "SizeN_A" )
    {
      APPEND_OBJECT_METHOD_REAL( aSystemPtr, System, getSizeN_A );
    }
  else
    {
      THROW_EXCEPTION
	( NotFound,
	  "ExpressionCompiler: System attribute [" +
	  aMethodName + "] not found." ); 
    }

}

#undef APPEND_OBJECT_METHOD_REAL
#undef APPEND_OBJECT_METHOD_INTEGER





/**
   This function is ExpressionCompiler subclass member function.
   This member function evaluates AST tree and makes binary codes.
*/

void ExpressionCompiler::compileTree
( TreeIterator const& aTreeIterator, CodeRef aCode )
{
  /**
     compile AST
  */

  switch ( aTreeIterator->value.id().to_long() )
    {
      /**
	 Floating Grammar compile
      */

    case CompileGrammar::FLOAT :
      {
	assert( aTreeIterator->children.size() == 0 );

	const String aFloatString( aTreeIterator->value.begin(),
				   aTreeIterator->value.end() );
	  
	const Real aFloatValue = stringCast<Real>( aFloatString );
	  
	appendInstruction( aCode, Instruction<PUSH_REAL>( aFloatValue ) );
	  
	return;
      }

    /**
       Integer Grammar compile
    */
	
    case CompileGrammar::INTEGER :
      {	
	assert( aTreeIterator->children.size() == 0 );
	  
	const String anIntegerString( aTreeIterator->value.begin(),
				      aTreeIterator->value.end() );

	const Real anIntegerValue = stringCast<Real>( anIntegerString );
	  
	appendInstruction( aCode, Instruction<PUSH_REAL>( anIntegerValue ) );

	return; 
	
      }
      
    /**
       Grammar compile
    */

    case CompileGrammar::EXPONENT:
      {
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
	  
	if( anExponentString != "-")
	  {
	    const Real 
	      anExponentValue = stringCast<Real>( anExponentString );
	      
	    appendInstruction
	      ( aCode, Instruction<PUSH_REAL>
		( aBaseValue * pow( 10, anExponentValue ) ) );
	  }
	else
	  {
	    const String 
	      anExponentString1( ( aChildTreeIterator + 2 )->value.begin(),
				 ( aChildTreeIterator + 2 )->value.end() );
	      
	    const Real 
	      anExponentValue = stringCast<Real>( anExponentString1 );
	      
	    appendInstruction
	      ( aCode, 
		Instruction<PUSH_REAL>
		( aBaseValue * pow( 10, -anExponentValue ) ) );
	  }
	  
	return; 
      }



      /**
	 Call_Func Grammar compile
      */

    case CompileGrammar::CALL_FUNC :
      {

	Integer aChildTreeSize( aTreeIterator->children.size() );
	  
	const String aFunctionString( aTreeIterator->value.begin(),
				      aTreeIterator->value.end() );


	assert( aChildTreeSize != 0 );

	if( aChildTreeSize == 1 )
	  {
	    FunctionMap1Iterator aFunctionMap1Iterator;

	    aFunctionMap1Iterator = 
	      theFunctionMap1.find( aFunctionString );
		
	    TreeIterator const& 
	      aChildTreeIterator( aTreeIterator->children.begin() );


	    if( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
		aChildTreeIterator->value.id() == CompileGrammar::FLOAT )
	      {
		const String 
		  anArgumentString( aChildTreeIterator->value.begin(),
				    aChildTreeIterator->value.end() );
		    
		const Real 
		  anArgumentValue = stringCast<Real>( anArgumentString );
		
		if( aFunctionMap1Iterator != theFunctionMap1.end() )
		  {
		    appendInstruction
		      ( aCode, Instruction<PUSH_REAL>
			( (*aFunctionMap1Iterator->second)
			  ( anArgumentValue ) ) );
		  }
		else
		  {
		    THROW_EXCEPTION( NoSlot, 
				     aFunctionString +
				     String( " : No such function." ) );
		  }
	      }
	    else
	      {
		compileTree( aChildTreeIterator, aCode );
		    
		if( aFunctionMap1Iterator != theFunctionMap1.end() )
		  {
		    appendInstruction
		      ( aCode, Instruction<CALL_FUNC1>
			( aFunctionMap1Iterator->second ) );
		  }
		else
		  {
		    THROW_EXCEPTION( NoSlot, 
				     aFunctionString +
				     String( " : No such function." ) );
		  }
	      }
	  }

	else if( aChildTreeSize == 2 )
	  {

	    TreeIterator const& 
	      aChildTreeIterator( aTreeIterator->children.begin() );

	    compileTree( aChildTreeIterator, aCode );
	    compileTree( aChildTreeIterator+1, aCode );
		
	      
	    FunctionMap2Iterator aFunctionMap2Iterator;

	    aFunctionMap2Iterator =
	      theFunctionMap2.find( aFunctionString );
		
	    if( aFunctionMap2Iterator != theFunctionMap2.end() )
	      {
		appendInstruction
		  ( aCode, Instruction<CALL_FUNC2>
		    ( aFunctionMap2Iterator->second ) );
	      }
	    else
	      {
		THROW_EXCEPTION( NoSlot, 
				 aFunctionString +
				 String( " : No such function." ) );
	      }
	  }

	else
	  {
	    THROW_EXCEPTION( UnexpectedError,
			     " : Too many arguments" );
	  }

	return;
      }	


    /**
       System_Func Grammar compile
    */

    case CompileGrammar::SYSTEM_FUNC :
      {
	TreeIterator const& 
	  aChildTreeIterator( aTreeIterator->children.begin() );

	const String aClassString( aChildTreeIterator->value.begin(),
				   aChildTreeIterator->value.end() );

	const String
	  aClassMethodString( ( aChildTreeIterator+1 )->value.begin(),
			      ( aChildTreeIterator+1 )->value.end() );
	  

	assert( *aTreeIterator->value.begin() == '.' );
	

	if( aClassString == "self" )  // Process Class
	  {
	    if( aClassMethodString != "getSuperSystem" )
	      {
		THROW_EXCEPTION( NoSlot,
				 aClassMethodString + 
				 String
				 ( " : No such Process method." ) );
	      }

	    SystemPtr aSystemPtr( theProcessPtr->getSuperSystem() );

	    const String aMethodName( ( aChildTreeIterator+2 )->value.begin(),
				      ( aChildTreeIterator+2 )->value.end() );
	    
	    appendSystemMethodInstruction( aCode, aSystemPtr, aMethodName );
	  }		  		
	else // VariableReference Class
	  {
	    VariableReferenceCref
	      aVariableReference( theProcessPtr->
				  getVariableReference( aClassString ) );
	      
	    if( aClassMethodString != "getSuperSystem" )
	      {
		THROW_EXCEPTION( NoSlot,
				 aClassMethodString + 
				 String( " : No such Process method." ) );
	      }

	    SystemPtr const aSystemPtr( aVariableReference.getSuperSystem() );

	    const String aMethodName( ( aChildTreeIterator+2 )->value.begin(),
				      ( aChildTreeIterator+2 )->value.end() );
	    
	    appendSystemMethodInstruction( aCode, aSystemPtr, aMethodName );

	  }
	return;
      }


    /**
       Variable Grammar compile
    */

    case CompileGrammar::VARIABLE :
      {
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
	  ( theProcessPtr->
	    getVariableReference( aVariableReferenceString ) );
	  
	appendVariableReferenceMethodInstruction
	  ( aCode,
	    const_cast<VariableReference*>( &aVariableReference ),
	    aVariableReferenceMethodString );

	return;
	
      }



    /**
       Identifier Grammar compile
    */

    case CompileGrammar::IDENTIFIER :
      {
	assert( aTreeIterator->children.size() == 0 );
	
	const String anIdentifierString( aTreeIterator->value.begin(),
					 aTreeIterator->value.end() );

	ConstantMapIterator aConstantMapIterator;
	PropertyMapIterator aPropertyMapIterator;

	aConstantMapIterator = 
	  theConstantMap.find( anIdentifierString );
	aPropertyMapIterator =
	  thePropertyMapPtr->find( anIdentifierString );
	

	if( aConstantMapIterator != theConstantMap.end() )
	  {
	    appendInstruction
	      ( aCode,
		Instruction<PUSH_REAL>( aConstantMapIterator->second ) );
	  }

	else if( aPropertyMapIterator != thePropertyMapPtr->end() )
	  {
	    appendInstruction
	      ( aCode, Instruction<LOAD_REAL>
		( &(aPropertyMapIterator->second) ) );
	  }

	else
	  {
	    THROW_EXCEPTION( NoSlot,
			     anIdentifierString +
			     String( " : No such Property slot." ) );
	  }
	
	return;      
      }



    /**
       Negative Grammar compile 
    */
    
    case CompileGrammar::NEGATIVE :
      {
	assert( *aTreeIterator->value.begin() == '-' );

	TreeIterator const& 
	  aChildTreeIterator( aTreeIterator->children.begin() );


	if( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
	    aChildTreeIterator->value.id() == CompileGrammar::FLOAT )
	  {
	    const String 
	      aValueString( aChildTreeIterator->value.begin(),
			    aChildTreeIterator->value.end() );

	    const Real 
	      value = stringCast<Real>( aValueString );

	    appendInstruction( aCode, Instruction<PUSH_REAL>( -value ) );
	  }
	else
	  {
	    compileTree( aChildTreeIterator, aCode );

	    appendInstruction( aCode, Instruction<NEG>() );
	  }
	
	return;
      
      }
    


    /**
       Power Grammar compile
    */

    case CompileGrammar::POWER :
      {
	assert(aTreeIterator->children.size() == 2);

	TreeIterator const& 
	  aChildTreeIterator( aTreeIterator->children.begin() );


	if( ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
	      aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) && 
	    ( (aChildTreeIterator+1)->value.id() ==CompileGrammar::INTEGER ||
	      (aChildTreeIterator+1)->value.id() == CompileGrammar::FLOAT ) )
	  {

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


	    if( *aTreeIterator->value.begin() == '^' )
	      {
		appendInstruction
		  ( aCode, Instruction<PUSH_REAL>
		    ( pow( anArgumentValue1, anArgumentValue2 ) ) );
	      }

	    else

	      THROW_EXCEPTION( UnexpectedError,
			       String( "Invalid operation" ) );

	    return;
	  }
	else
	  {
	    compileTree( aTreeIterator->children.begin(), aCode );
	    compileTree( aTreeIterator->children.begin()+1, aCode );
	    
	    if( *aTreeIterator->value.begin() == '^' )
	      {
		RealFunc2 aPowFunc( theFunctionMap2.find( "pow" )->second );
		appendInstruction( aCode, 
				   Instruction<CALL_FUNC2>( aPowFunc ) );
	      }

	    else

	      THROW_EXCEPTION( UnexpectedError,
			       String( "Invalud operation" ) );

	    return;
	  }

	return;
      
      }



    /**
       Term Grammar compile
    */

    case CompileGrammar::TERM :
      {

	assert(aTreeIterator->children.size() == 2);


	TreeIterator const& 
	  aChildTreeIterator( aTreeIterator->children.begin() );


	if( ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
	      aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) && 
	    ( (aChildTreeIterator+1)->value.id() ==CompileGrammar::INTEGER ||
	      (aChildTreeIterator+1)->value.id() == CompileGrammar::FLOAT ) )
	  {

	    const String aTerm1String( aChildTreeIterator->value.begin(),
				       aChildTreeIterator->value.end() );

	    const String 
	      aTerm2String( ( aChildTreeIterator+1 )->value.begin(),
			    ( aChildTreeIterator+1 )->value.end() );

	    const Real aTerm1Value = stringCast<Real>( aTerm1String );
	    const Real aTerm2Value = stringCast<Real>( aTerm2String );


	    if (*aTreeIterator->value.begin() == '*')
	      {
		appendInstruction
		  ( aCode, 
		    Instruction<PUSH_REAL>( aTerm1Value * aTerm2Value ) );
	      }	

	    else if (*aTreeIterator->value.begin() == '/')
	      {
		appendInstruction
		  ( aCode, 
		    Instruction<PUSH_REAL>( aTerm1Value / aTerm2Value ) );
	      }

	    else
	      THROW_EXCEPTION( UnexpectedError,
			       String( "Invalid operation" ) );

	    return;
	  }
	else
	  {
	    compileTree( aChildTreeIterator, aCode );
	    compileTree( aChildTreeIterator+1, aCode );
	    
	    if (*aTreeIterator->value.begin() == '*')
	      {
		appendInstruction( aCode, Instruction<MUL>() );
	      }
	    
	    else if (*aTreeIterator->value.begin() == '/')
	      {
		appendInstruction( aCode, Instruction<DIV>() );
	      }

	    else
	      THROW_EXCEPTION( UnexpectedError,
			       String( "Invalid operation" ) );

	    return;
	  }

	return;
      
      }

    

    /**
       Expression Grammar compile
    */

    case CompileGrammar::EXPRESSION :
      {

	assert(aTreeIterator->children.size() == 2);
	
	TreeIterator const& 
	  aChildTreeIterator( aTreeIterator->children.begin() );


	if( ( aChildTreeIterator->value.id() == CompileGrammar::INTEGER ||
	      aChildTreeIterator->value.id() == CompileGrammar::FLOAT ) &&
	    ( (aChildTreeIterator+1)->value.id() ==CompileGrammar::INTEGER ||
	      (aChildTreeIterator+1)->value.id() == CompileGrammar::FLOAT ) )
	  {
	    const String aTerm1String( aChildTreeIterator->value.begin(),
				       aChildTreeIterator->value.end() );
	      
	    const String 
	      aTerm2String( ( aChildTreeIterator+1 )->value.begin(),
			    ( aChildTreeIterator+1 )->value.end() );

	    const Real aTerm1Value = stringCast<Real>( aTerm1String );
	    const Real aTerm2Value = stringCast<Real>( aTerm2String );


	    if (*aTreeIterator->value.begin() == '+')
	      {
		appendInstruction
		  ( aCode, 
		    Instruction<PUSH_REAL>( aTerm1Value + aTerm2Value ) );
	      }	

	    else if (*aTreeIterator->value.begin() == '-')
	      {
		appendInstruction
		  ( aCode, 
		    Instruction<PUSH_REAL>( aTerm1Value - aTerm2Value ) );
	      }

	    else
	      THROW_EXCEPTION( UnexpectedError,
			       String( "Invalid operation" ) );
	  }
	else
	  {
	    compileTree( aChildTreeIterator, aCode );
	    compileTree( aChildTreeIterator+1, aCode );
		

	    if (*aTreeIterator->value.begin() == '+')
	      {
		appendInstruction( aCode, Instruction<ADD>() );
	      }

	    else if (*aTreeIterator->value.begin() == '-')
	      {
		appendInstruction( aCode, Instruction<SUB>() );
	      }

	    else
	      THROW_EXCEPTION( UnexpectedError,
			       String( "Invalid operation" ) );
	  }

	return;

      }
	
    default :
      THROW_EXCEPTION( UnexpectedError, String( "No such syntax" ) );
	
      return;
    }
}


// this should be moved to .cpp
  
ConstantMap ExpressionCompiler::theConstantMap;
FunctionMap1 ExpressionCompiler::theFunctionMap1;
FunctionMap2 ExpressionCompiler::theFunctionMap2;

#endif /* __EXPRESSIONCOMPILER_HPP */

