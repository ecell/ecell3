//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2004 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// authors:
//     Kouichi Takahashi
//     Tatsuya Ishida
//
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#ifndef __EXPRESSIONCOMPILER_HPP
#define __EXPRESSIONCOMPILER_HPP

#include <new>

#include "libecs.hpp"
#include "Process.hpp"

#include <boost/spirit/core.hpp>
#include <boost/spirit/tree/ast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
//#include "boost/variant.hpp"

#if SPIRIT_VERSION >= 0x1800
#define PARSER_CONTEXT parser_context<>
#else
#define PARSER_CONTEXT parser_context
#endif

using namespace boost::spirit;

USE_LIBECS;

namespace libecs
{

  DECLARE_ASSOCVECTOR
  ( String, Real(*)(Real), std::less<const String>, FunctionMap1 );
  DECLARE_ASSOCVECTOR
  ( String, Real(*)(Real,Real), std::less<const String>, FunctionMap2 );
  DECLARE_ASSOCVECTOR( String, Real, std::less<const String>, ConstantMap );
  DECLARE_ASSOCVECTOR
  ( String, Real, std::less<const String>, PropertyMap );


  class ExpressionCompiler
  {
  public:

    DECLARE_VECTOR( char, Code );

    // possible operand types:
    typedef libecs::Real    Real;
    typedef Real*           RealPtr;
    typedef libecs::Integer Integer;
    typedef void*           Pointer;
    typedef const Real 
    (libecs::VariableReference::* VariableReferenceMethodPtr)() const;
    typedef SystemPtr 
    (libecs::VariableReference::* VariableReferenceSystemMethodPtr)() const;
    typedef SystemPtr  (libecs::Process::* ProcessMethodPtr)() const;
    typedef const Real (libecs::System::* SystemMethodPtr)() const;
    typedef Pointer (*PointerFunc)();
    typedef Real (*RealFunc0)();
    typedef Real (*RealFunc1)( Real );
    typedef Real (*RealFunc2)( Real, Real );

    enum Opcode  // the order of items is optimized. don't change.
      {   
	ADD = 0  // no arg
	, SUB    // no arg
	, MUL    // no arg
	, DIV    // no arg
	, POW    // no arg
	, NEG    // no arg
	, HALT   // no arg
	// Those instructions above are candidates of stack operations folding
	// in the stack machine, and grouped here to optimize the switch().


	//, CALL_FUNC0 // RealFunc0
	, CALL_FUNC1 // RealFunc1
	, CALL_FUNC2 // RealFunc2
	, LOAD_REAL  // Real*
	, PUSH_REAL  // Real
	, VARREF_METHOD // VariableReferencePtr, VariableReferenceMethodPtr
	, PUSH_INTEGER // Integer
	, PUSH_POINTER // Pointer
	, EQ     // no arg
	, NEQ    // no arg
	, GT     // no arg
	, LT     // no arg
	, GEQ    // no arg
	, LEQ    // no arg
	, AND    // no arg
	, OR     // no arg
	, XOR    // no arg
	, NOT    // no arg
	, SYSTEM_FUNC  // 
	, PROCESS_METHOD // ProcessPtr, ProcessMethodPtr
	, SYSTEM_METHOD // SystemPtr, SystemMethodPtr
	, VARREF_TO_SYSTEM_METHOD  // VariableReferenceSystemMethodPtr
	, PROCESS_TO_SYSTEM_METHOD // ProcessMethodPtr
	, SYSTEM_TO_REAL_METHOD // SystemMethodPtr
	, NOP
	, END=NOP
	// , CALL_OBJECT???
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
      //    typedef typename OPERANDTYPE::type Operand_;
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
      
    protected:
      
      InstructionBase( Opcode );
      //	:
      //	InstructionHead( NOP )
      //      {}  // should be left undefined


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
    typedef ObjectMethodOperand<VariableReference,const Real>
    VariableReferenceMethod;



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
	  FLOATING,
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
	rule<ScannerT, PARSER_CONTEXT, parser_tag<FLOATING> >     floating;
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
    
    ExpressionCompiler()
    {
      if( theConstantMap.empty() == true )
	{
	  setConstantMap();
	}
      if( theFunctionMap1.empty() == true )
	{
	  setFunctionMap();
	}
    }
    
    ~ExpressionCompiler()
    {
      ; // do nothing
    }
    
    typedef char const*         iterator_t;
    typedef tree_match<iterator_t> parse_tree_match_t;
    typedef parse_tree_match_t::tree_iterator TreeIterator;
    //    DECLARE_CLASS( TreeIterator );
    
    void setProcessPtr( ProcessPtr aProcessPtr )
    {
      theProcessPtr = aProcessPtr;
    }

    const int getStackSize()
    {
      return theStackSize;
    }
    
    void setConstantMap();
    
    void setFunctionMap();

    void setPropertyMap( PropertyMapPtr aPropertyMapPtr )
    {
      thePropertyMapPtr = aPropertyMapPtr;
    }

    const Code compileExpression( StringCref anExpression );
    
  protected:

    template < class INSTRUCTION >
    void appendInstruction( Code& aCode, const INSTRUCTION& anInstruction )
    {
      Code::size_type aCodeSize( aCode.size() );
      aCode.resize( aCodeSize + sizeof( INSTRUCTION ) );
      new (&aCode[aCodeSize]) INSTRUCTION( anInstruction );
    }

  private:
    
    void compileTree( TreeIterator const&  i, CodeRef aCode );  

  private:
    
    int theStackSize;
    ProcessPtr theProcessPtr;
    
    ConstantMap theConstantMap;
    FunctionMap1 theFunctionMap1;
    FunctionMap2 theFunctionMap2;

    PropertyMapPtr thePropertyMapPtr;
  };

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


    
  SPECIALIZE_OPCODE2OPERAND( PUSH_REAL,      Real );
  SPECIALIZE_OPCODE2OPERAND( PUSH_INTEGER,   Integer );
  SPECIALIZE_OPCODE2OPERAND( PUSH_POINTER,   Pointer );
  SPECIALIZE_OPCODE2OPERAND( LOAD_REAL,      RealPtr );
  //SPECIALIZE_OPCODE2OPERAND( CALL_FUNC0,     RealFunc0 );
  SPECIALIZE_OPCODE2OPERAND( CALL_FUNC1,     RealFunc1 );
  SPECIALIZE_OPCODE2OPERAND( CALL_FUNC2,     RealFunc2 );
  SPECIALIZE_OPCODE2OPERAND( VARREF_METHOD,  VariableReferenceMethod );
  SPECIALIZE_OPCODE2OPERAND( PROCESS_METHOD, ProcessMethod );
  SPECIALIZE_OPCODE2OPERAND( SYSTEM_METHOD,  SystemMethod );
  SPECIALIZE_OPCODE2OPERAND( PROCESS_TO_SYSTEM_METHOD, ProcessMethodPtr );
  SPECIALIZE_OPCODE2OPERAND( SYSTEM_TO_REAL_METHOD, SystemMethodPtr );  
  SPECIALIZE_OPCODE2OPERAND( VARREF_TO_SYSTEM_METHOD,  
			     VariableReferenceSystemMethodPtr );

  
#define DEFINE_OPCODE2INSTRUCTION( CODE )\
  template<> class\
    ExpressionCompiler::Opcode2Instruction<ExpressionCompiler::CODE>\
  {\
  public:\
    typedef ExpressionCompiler::Instruction<ExpressionCompiler::CODE> type;\
    typedef type::Operand operandtype;\
  }

      
  DEFINE_OPCODE2INSTRUCTION( PUSH_REAL );
  DEFINE_OPCODE2INSTRUCTION( PUSH_INTEGER );
  DEFINE_OPCODE2INSTRUCTION( PUSH_POINTER );
  DEFINE_OPCODE2INSTRUCTION( NEG );
  DEFINE_OPCODE2INSTRUCTION( ADD );
  DEFINE_OPCODE2INSTRUCTION( SUB );
  DEFINE_OPCODE2INSTRUCTION( MUL );
  DEFINE_OPCODE2INSTRUCTION( DIV );
  DEFINE_OPCODE2INSTRUCTION( POW );
  DEFINE_OPCODE2INSTRUCTION( EQ );
  DEFINE_OPCODE2INSTRUCTION( NEQ );
  DEFINE_OPCODE2INSTRUCTION( GT );
  DEFINE_OPCODE2INSTRUCTION( LT );
  DEFINE_OPCODE2INSTRUCTION( GEQ );
  DEFINE_OPCODE2INSTRUCTION( LEQ );
  DEFINE_OPCODE2INSTRUCTION( AND );
  DEFINE_OPCODE2INSTRUCTION( OR );
  DEFINE_OPCODE2INSTRUCTION( XOR );
  DEFINE_OPCODE2INSTRUCTION( NOT );
  DEFINE_OPCODE2INSTRUCTION( LOAD_REAL );
  //DEFINE_OPCODE2INSTRUCTION( CALL_FUNC0 );
  DEFINE_OPCODE2INSTRUCTION( CALL_FUNC1 );
  DEFINE_OPCODE2INSTRUCTION( CALL_FUNC2 );
  DEFINE_OPCODE2INSTRUCTION( VARREF_METHOD );
  DEFINE_OPCODE2INSTRUCTION( PROCESS_METHOD );
  DEFINE_OPCODE2INSTRUCTION( SYSTEM_METHOD );
  DEFINE_OPCODE2INSTRUCTION( VARREF_TO_SYSTEM_METHOD );
  DEFINE_OPCODE2INSTRUCTION( PROCESS_TO_SYSTEM_METHOD );
  DEFINE_OPCODE2INSTRUCTION( SYSTEM_TO_REAL_METHOD );
  DEFINE_OPCODE2INSTRUCTION( HALT );
  DEFINE_OPCODE2INSTRUCTION( NOP );

  void ExpressionCompiler::setConstantMap()
  {
    theConstantMap[ "true" ] = 1.0;
    theConstantMap[ "false" ] = 0.0;
    theConstantMap[ "pi" ] = M_PI;
    theConstantMap[ "NaN" ] = std::numeric_limits<Real>::quiet_NaN();
    theConstantMap[ "INF"] = std::numeric_limits<Real>::infinity();
    theConstantMap[ "N_A" ] = N_A;
    theConstantMap[ "exp" ] = M_E;
  }

  const ExpressionCompiler::Code 
  ExpressionCompiler::compileExpression( StringCref anExpression )
  {
    Code aCode;
    CompileGrammar aGrammer;
	        
    theStackSize = 1;
      
    tree_parse_info<> 
      info( ast_parse( anExpression.c_str(), aGrammer, space_p ) );

    if( info.full )
      {
	compileTree( info.trees.begin(), aCode );
	  
	// place HALT at the tail.
	appendInstruction( aCode, Instruction<HALT>() );
      }
    else
      {
	THROW_EXCEPTION( UnexpectedError, 
			 "Parse error in the expression. Expression : " +
			 anExpression );
      }
      
    return aCode;
  }


  void libecs::ExpressionCompiler::setFunctionMap()
  {
    theFunctionMap1["abs"] = std::fabs;
    theFunctionMap1["sqrt"] = std::sqrt;
    theFunctionMap1["exp"] = std::exp;
    theFunctionMap1["log10"] = std::log10;
    theFunctionMap1["log"] = std::log;
    theFunctionMap1["floor"] = std::floor;
    theFunctionMap1["ceil"] = std::ceil;
    theFunctionMap1["sin"] = std::sin;
    theFunctionMap1["cos"] = std::cos;
    theFunctionMap1["tan"] = std::tan;
    theFunctionMap1["sinh"] = std::sinh;
    theFunctionMap1["cosh"] = std::cosh;
    theFunctionMap1["tanh"] = std::tanh;
    theFunctionMap1["asin"] = std::asin;
    theFunctionMap1["acos"] = std::acos;
    theFunctionMap1["atan"] = std::atan;
    theFunctionMap1["fact"] = fact;
    theFunctionMap1["asinh"] = asinh;
    theFunctionMap1["acosh"] = acosh;
    theFunctionMap1["atanh"] = atanh;
    theFunctionMap1["asech"] = asech;
    theFunctionMap1["acsch"] = acsch;
    theFunctionMap1["acoth"] = acoth;
    theFunctionMap1["sech"] = sech;
    theFunctionMap1["csch"] = csch;
    theFunctionMap1["coth"] = coth;
    theFunctionMap1["asec"] = asec;
    theFunctionMap1["acsc"] = acsc;
    theFunctionMap1["acot"] = acot;
    theFunctionMap1["sec"] = sec;
    theFunctionMap1["csc"] = csc;
    theFunctionMap1["cot"] = cot;

    theFunctionMap2["pow"] = pow;
  }



  /**
     This function is ExpressionCompiler subclass member function.
     This member function evaluates AST tree and makes binary codes.
  */

  void
  libecs::ExpressionCompiler::compileTree
  ( TreeIterator const& i, CodeRef aCode )
  {

    std::vector<char>::iterator CharIterator;

    /**
       compile AST
    */

    switch ( i->value.id().to_long() )
      {
	/**
	   Floating Grammar compile
	*/

      case CompileGrammar::FLOATING :
	{
	  Real value;
	  String aString;

	  assert(i->children.size() == 0);

	  for( CharIterator = i->value.begin();
	       CharIterator != i->value.end(); ++CharIterator )
	    {
	      aString += *CharIterator;
	    }
	  
	  value = stringCast<Real>( aString );
	  
	  ++theStackSize;

	  appendInstruction( aCode, Instruction<PUSH_REAL>( value ) );
	  
	  return;
	}

      /**
	 Integer Grammar compile
      */
	
      case CompileGrammar::INTEGER :
	{	
	  Real value;
	  String aString;

	  assert(i->children.size() == 0);
	  
	  for( CharIterator = i->value.begin();
	       CharIterator != i->value.end(); ++CharIterator )
	    {	  
	      aString += *CharIterator;
	    }
	  
	  value = stringCast<Real>( aString );
	  
	  ++theStackSize;

	  appendInstruction( aCode, Instruction<PUSH_REAL>( value ) );

	  return; 
	
	}
	
      /**
	 Grammar compile
      */

      case CompileGrammar::EXPONENT:
	{
	  Real value1, value2;
	  String aString1, aString2, aString3;

	  assert( *i->value.begin() == 'E' || *i->value.begin() == 'e' );
	  
	  for( CharIterator = i->children.begin()->value.begin();
	       CharIterator != i->children.begin()->value.end();
	       ++CharIterator )
	    {
	      aString1 += *CharIterator;
	    }
	  
	  for( CharIterator = ( i->children.begin()+1 )->value.begin();
	       CharIterator != ( i->children.begin()+1 )->value.end();
	       ++CharIterator )
	    {
	      aString2 += *CharIterator;
	    }
	  
	  value1 = stringCast<Real>( aString1 );
	  
	  ++theStackSize;
	  
	  if( aString2 != "-")
	    {
	      value2 = stringCast<Real>( aString2 );
	      
	      appendInstruction
		( aCode, Instruction<PUSH_REAL>( value1 * pow(10, value2) ) );
	    }
	  else
	    {
	      for( CharIterator = ( i->children.begin()+2 )->value.begin();
		   CharIterator != ( i->children.begin()+2 )->value.end();
		   ++CharIterator )
		{
		  aString3 += *CharIterator;
		}
	      
	      value2 = stringCast<Real>( aString3 );
	      
	      appendInstruction
		( aCode, Instruction<PUSH_REAL>( value1 * pow(10, -value2) ) );
	    }
	  
	  return; 
	}

	/**
	   Call_Func Grammar compile
	*/

      case CompileGrammar::CALL_FUNC :
	{
	  Real value1;
	  String aString1, aString2;
	  FunctionMap1Iterator theFunctionMap1Iterator;
	  FunctionMap2Iterator theFunctionMap2Iterator;
	    

	  assert( i->children.size() != 0 );
	
	  for( CharIterator = i->value.begin();
	       CharIterator != i->value.end(); ++CharIterator )
	    {
	      aString1 += *CharIterator;
	    }

	  ++theStackSize;
	  if( i->children.size() == 1 )
	    {
	      if( aString1 == "not" )
		{
		  compileTree( i->children.begin(), aCode );

		  appendInstruction( aCode, Instruction<NOT>() );
		}
	      else
		{
		  theFunctionMap1Iterator = theFunctionMap1.find( aString1 );
		
		  if( i->children.begin()->value.id() ==
		      CompileGrammar::INTEGER ||
		      i->children.begin()->value.id() ==
		      CompileGrammar::FLOATING  )
		    {
		      for( CharIterator =
			     i->children.begin()->value.begin();
			   CharIterator !=
			     i->children.begin()->value.end();
			   ++CharIterator )
			{
			  aString2 += *CharIterator;
			}
		    
		      value1 = stringCast<Real>( aString2 );
		
		      if( theFunctionMap1Iterator != theFunctionMap1.end() )
			{
			  appendInstruction
			    ( aCode, Instruction<PUSH_REAL>
			      ( (*theFunctionMap1Iterator->second)
				( value1 ) ) );
			}
		      else
			{
			  THROW_EXCEPTION( NoSlot, 
					   aString1 +
					   String( " : No such function." ) );
			}
		    }
		  else
		    {
		      compileTree( i->children.begin(), aCode );	  
		    
		      if( theFunctionMap1Iterator != theFunctionMap1.end() )
			{
			  appendInstruction
			    ( aCode, Instruction<CALL_FUNC1>
			      ( theFunctionMap1Iterator->second ) );
			}
		      else
			{
			  THROW_EXCEPTION( NoSlot, 
					   aString1 +
					   String( " : No such function." ) );
			}
		    }
		}
	    }
	  else if( i->children.size() == 2 )
	    {
	      compileTree( i->children.begin(), aCode );	  
	      compileTree( i->children.begin()+1, aCode );
		
	      if( aString1 == "eq" )
		{
		  appendInstruction( aCode, Instruction<EQ>() );
		}
	      else if( aString1 == "neq" )
		{
		  appendInstruction( aCode, Instruction<NEQ>() );
		}
	      else if( aString1 == "gt" )
		{
		  appendInstruction( aCode, Instruction<GT>() );
		}
	      else if( aString1 == "lt" )
		{
		  appendInstruction( aCode, Instruction<LT>() );
		}
	      else if( aString1 == "geq" )
		{
		  appendInstruction( aCode, Instruction<GEQ>() );
		}
	      else if( aString1 == "leq" )
		{
		  appendInstruction( aCode, Instruction<LEQ>() );
		}
	      else if( aString1 == "and" )
		{
		  appendInstruction( aCode, Instruction<AND>() );
		}
	      else if( aString1 == "or" )
		{
		  appendInstruction( aCode, Instruction<OR>() );
		}
	      else if( aString1 == "xor" )
		{
		  appendInstruction( aCode, Instruction<XOR>() );
		}
	      else
		{
		  theFunctionMap2Iterator = theFunctionMap2.find( aString1 );
		
		  if( theFunctionMap2Iterator != theFunctionMap2.end() )
		    {
		      appendInstruction
			( aCode, Instruction<CALL_FUNC2>
			  ( theFunctionMap2Iterator->second ) );
		    }
		  else
		    {
		      THROW_EXCEPTION( NoSlot, 
				       aString1 +
				       String( " : No such function." ) );
		    }
		}
	    }
	  else
	    {
	      THROW_EXCEPTION( NoSlot,
			       aString1 + 
			       String(" : No such function.") );
	    }

	  return;
	}	


      /**
	 System_Func Grammar compile
      */

      case CompileGrammar::SYSTEM_FUNC :
	{
	  String aString1, aString2, aString3;
	  FunctionMap2Iterator theFunctionMap2Iterator;
	  
	  ++theStackSize;
	
	  for( CharIterator = i->children.begin()->value.begin();
	       CharIterator != i->children.begin()->value.end();
	       ++CharIterator )
	    {
	      aString1 += *CharIterator;
	    }

	  for( CharIterator = ( i->children.begin()+1 )->value.begin();
	       CharIterator != ( i->children.begin()+1 )->value.end();
	       ++CharIterator )
	    {
	      aString2 += *CharIterator;
	    }
	
	  assert( *i->value.begin() == '.' );
	
	  if( aString1 == "self" )
	    {
	      appendInstruction
		( aCode, Instruction<PUSH_POINTER>( theProcessPtr ) );
	      

	      if( aString2 == "getSuperSystem" )
		{
		  appendInstruction
		    ( aCode, Instruction<PROCESS_TO_SYSTEM_METHOD>
		      ( &libecs::Process::getSuperSystem ) );		  
		  

		  for( CharIterator = ( i->children.begin()+2 )->value.begin();
		       CharIterator != ( i->children.begin()+2 )->value.end();
		       ++CharIterator )
		    {
		      aString3 += *CharIterator;
		    }
		  
		  if( aString3 == "Size" )
		    {
		      appendInstruction
			( aCode, Instruction<SYSTEM_TO_REAL_METHOD>
			  ( &libecs::System::getSize ) );
		    }

		  else if( aString3 == "SizeN_A" )
		    {
		      appendInstruction
			( aCode, Instruction<SYSTEM_TO_REAL_METHOD>
			  ( &libecs::System::getSizeN_A ) );
		    }
		  else
		    {
		      THROW_EXCEPTION( NoSlot,
				       aString3 + 
				       String
				       (" : No such System method.") );
		    }
		}
	      else
		{
		  THROW_EXCEPTION( NoSlot,
				   aString2 + 
				   String
				   ( " : No such Process method." ) );
		}
	    }		  		
	  else
	    {
	      VariableReferenceCref
		aVariableReference( theProcessPtr->libecs::Process::
				    getVariableReference( aString1 ) );
	      
	      appendInstruction
		( aCode, Instruction<PUSH_POINTER>
		  ( const_cast<VariableReference*>( &aVariableReference ) ) );
	      
	      if( aString2 == "getSuperSystem" )
		{
		  appendInstruction
		    ( aCode, Instruction<VARREF_TO_SYSTEM_METHOD>
		      ( &libecs::VariableReference::getSuperSystem ) );

		  for( CharIterator = ( i->children.begin()+2 )->value.begin();
		       CharIterator != ( i->children.begin()+2 )->value.end();
		       ++CharIterator )
		    {
		      aString3 += *CharIterator;
		    }
		  
		  
		  if( aString3 == "Size" )
		    {
		      appendInstruction
			( aCode, Instruction<SYSTEM_TO_REAL_METHOD>
			  ( &libecs::System::getSize ) );
		    }

		  else if( aString3 == "SizeN_A" )
		    {
		      appendInstruction
			( aCode, Instruction<SYSTEM_TO_REAL_METHOD>
			  ( &libecs::System::getSizeN_A ) );
		    }
		  else
		    {
		      THROW_EXCEPTION( NoSlot,
				       aString3 + 
				       String
				       (" : No such System method.") );
		    } 
		}
	      else
		{
		  THROW_EXCEPTION( NoSlot,
				   aString2 + 
				   String
				   ( " : No such Process method." ) );
		}
	    }
	  return;
	}


      /**
	 Variable Grammar compile
      */

      case CompileGrammar::VARIABLE :
	{
	  String aString1, aString2;
	  
	  assert( *i->value.begin() == '.' );

	  for( CharIterator = i->children.begin()->value.begin();
	       CharIterator != i->children.begin()->value.end();
	       ++CharIterator )
	    {
	      aString1 += *CharIterator;
	    }

	  for( CharIterator = ( i->children.begin()+1 )->value.begin();
	       CharIterator != ( i->children.begin()+1 )->value.end();
	       ++CharIterator )
	    {
	      aString2 += *CharIterator;
	    }
      	
	  VariableReferenceCref
	    aVariableReference( theProcessPtr->libecs::Process::
				getVariableReference( aString1 ) );
	  
	  ++theStackSize;

	  if( aString2 == "MolarConc" )
	    {
	      VariableReferenceMethod aVariableReferenceMethod;

	      aVariableReferenceMethod.theOperand1 = &aVariableReference;
	      aVariableReferenceMethod.theOperand2 = 
		&libecs::VariableReference::getMolarConc;

	      appendInstruction
		( aCode, 
		  Instruction<VARREF_METHOD>( aVariableReferenceMethod ) );
	    }

	  else if( aString2 == "NumberConc" )
	    {
	      VariableReferenceMethod aVariableReferenceMethod;

	      aVariableReferenceMethod.theOperand1 = &aVariableReference;
	      aVariableReferenceMethod.theOperand2 = 
		&libecs::VariableReference::getNumberConc;

	      appendInstruction
		( aCode,
		  Instruction<VARREF_METHOD>( aVariableReferenceMethod ) );
	    }

	  else if( aString2 == "Value" )
	    {
	      VariableReferenceMethod aVariableReferenceMethod;

	      aVariableReferenceMethod.theOperand1 = &aVariableReference;
	      aVariableReferenceMethod.theOperand2 = 
		&libecs::VariableReference::getValue;

	      appendInstruction
		( aCode,
		  Instruction<VARREF_METHOD>( aVariableReferenceMethod ) );
	    }
	  /**else if( str_child2 == "Coefficient" )
	    {
	      VariableReferenceMethod aVariableReferenceMethod;

	      aVariableReferenceMethod.theOperand1 = &aVariableReference;
	      aVariableReferenceMethod.theOperand2 = 
		&libecs::VariableReference::getCoefficient;
	      
	      appendInstruction
		( aCode, Instruction<VARREF_METHOD>( aVariableReferenceMethod ) );
	      
		}*/ 
			/**else if( str_child2 == "Fixed" ){
			aCode.push_back(
			new VARREF_METHOD( aVariableReference,
			&libecs::VariableReference::isFixed ) );
			}*/

	  else if( aString2 == "Velocity" )
	    {
	      VariableReferenceMethod aVariableReferenceMethod;

	      aVariableReferenceMethod.theOperand1 = &aVariableReference;
	      aVariableReferenceMethod.theOperand2 = 
		&libecs::VariableReference::getVelocity;

	      appendInstruction
		( aCode,
		  Instruction<VARREF_METHOD>( aVariableReferenceMethod ) );
	    }

	  else if( aString2 == "TotalVelocity" )
	    {
	      VariableReferenceMethod aVariableReferenceMethod;

	      aVariableReferenceMethod.theOperand1 = &aVariableReference;
	      aVariableReferenceMethod.theOperand2 = 
		&libecs::VariableReference::getTotalVelocity;

	      appendInstruction
		( aCode,
		  Instruction<VARREF_METHOD>( aVariableReferenceMethod ) );
	    }
	  else
	    {
	      THROW_EXCEPTION
		( NoSlot,
		  aString2 + 
		  String
		  ( " : No such VariableReference attribute." ) ); 
	    }
	  return;
	
	}



      /**
	 Identifier Grammar compile
      */

      case CompileGrammar::IDENTIFIER :
	{
	  String aString1;
	  ConstantMapIterator theConstantMapIterator;
	  PropertyMapIterator thePropertyMapIterator;
    
	  assert( i->children.size() == 0 );
	
	  ++theStackSize;

	  for( CharIterator = i->value.begin();
	       CharIterator != i->value.end(); ++CharIterator )
	    {
	      aString1 += *CharIterator;
	    }

	  theConstantMapIterator = theConstantMap.find( aString1 );
	  thePropertyMapIterator = (*thePropertyMapPtr).find( aString1 );
	
	  if( theConstantMapIterator != theConstantMap.end() )
	    {
	      appendInstruction
		( aCode, Instruction<PUSH_REAL>( theConstantMapIterator->second ) );
	    }

	  else if( thePropertyMapIterator != (*thePropertyMapPtr).end() )
	    {
	      appendInstruction
		( aCode, Instruction<LOAD_REAL>
		  ( &(thePropertyMapIterator->second) ) );
	    }

	  else
	    {
	      THROW_EXCEPTION( NoSlot,
			       aString1 +
			       String( " : No such Property slot." ) );
	    }
	
	  return;      
	}



      /**
	 Negative Grammar compile 
      */
    
      case CompileGrammar::NEGATIVE :
	{
	  Real value;
	  String aString;

	  assert( *i->value.begin() == '-' );

	  for( CharIterator = i->children.begin()->value.begin();
	       CharIterator != i->children.begin()->value.end();
	       ++CharIterator )
	    {
	      aString += *CharIterator;
	    }

	  if( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
	      i->children.begin()->value.id() == CompileGrammar::FLOATING  )
	    {
	      value = stringCast<Real>( aString );

	      ++theStackSize;

	      appendInstruction( aCode, Instruction<PUSH_REAL>( -value ) );
	    }
	  else
	    {
	      compileTree(i->children.begin(), aCode );

	      appendInstruction( aCode, Instruction<NEG>() );
	    }
	
	  return;
      
	}
    


      /**
	 Power Grammar compile
      */

      case CompileGrammar::POWER :
	{
	  Real value1, value2;
	  String aString1, aString2;

	  assert(i->children.size() == 2);

	  if( ( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
		i->children.begin()->value.id() == CompileGrammar::FLOATING ) && 
	      ( ( i->children.begin()+1 )->value.id() == CompileGrammar::INTEGER
		||
		( i->children.begin()+1 )->value.id() == CompileGrammar::FLOATING
		) )
	    {
	      for( CharIterator = i->children.begin()->value.begin();
		   CharIterator != i->children.begin()->value.end();
		   ++CharIterator )
		{
		  aString1 += *CharIterator;
		}

	      for( CharIterator = ( i->children.begin()+1 )->value.begin();
		   CharIterator != ( i->children.begin()+1 )->value.end();
		   ++CharIterator )
		{
		  aString2 += *CharIterator;
		}

	      value1 = stringCast<Real>( aString1 );
	      value2 = stringCast<Real>( aString2 );	  

	      ++theStackSize;

	      if( *i->value.begin() == '^' )
		{
		  appendInstruction
		    ( aCode, Instruction<PUSH_REAL>( pow( value1, value2 ) ) );
		}

	      else
		THROW_EXCEPTION( UnexpectedError,
				 String( "Invalid operation" ) );

	      return;
	    }
	  else
	    {
	      compileTree( i->children.begin(), aCode );
	      compileTree( i->children.begin()+1, aCode );
	    
	      if( *i->value.begin() == '^' )
		{
		  appendInstruction( aCode, Instruction<POW>() );
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
	  Real value1, value2;
	  String aString1, aString2;

	  assert(i->children.size() == 2);

	  if( ( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
		i->children.begin()->value.id() == CompileGrammar::FLOATING ) && 
	      ( ( i->children.begin()+1 )->value.id() == CompileGrammar::INTEGER
		||
		( i->children.begin()+1 )->value.id() == CompileGrammar::FLOATING
		) )
	    {
	      for( CharIterator = i->children.begin()->value.begin();
		   CharIterator != i->children.begin()->value.end();
		   ++CharIterator )
		{
		  aString1 += *CharIterator;
		}

	      for( CharIterator = ( i->children.begin()+1 )->value.begin();
		   CharIterator != ( i->children.begin()+1 )->value.end();
		   ++CharIterator )
		{
		  aString2 += *CharIterator;
		}

	      value1 = stringCast<Real>( aString1 );
	      value2 = stringCast<Real>( aString2 );	  

	      ++theStackSize;

	      if (*i->value.begin() == '*')
		{
		  appendInstruction
		    ( aCode, Instruction<PUSH_REAL>( value1 * value2 ) );
		}	
	      else if (*i->value.begin() == '/')
		{
		  appendInstruction
		    ( aCode, Instruction<PUSH_REAL>( value1 / value2 ) );
		}
	      else
		THROW_EXCEPTION( UnexpectedError,
				 String( "Invalid operation" ) );

	      return;
	    }
	  else
	    {
	      compileTree( i->children.begin(), aCode );
	      compileTree( i->children.begin()+1, aCode );
	    
	      if (*i->value.begin() == '*')
		{
		  appendInstruction( aCode, Instruction<MUL>() );
		}
	    
	      else if (*i->value.begin() == '/')
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
	  Real value1, value2;
	  String aString1, aString2;

	  assert(i->children.size() == 2);
	
	  if( ( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
		i->children.begin()->value.id() == CompileGrammar::FLOATING ) &&
	      ( ( i->children.begin()+1 )->value.id() == CompileGrammar::INTEGER
		||
		( i->children.begin()+1 )->value.id() == CompileGrammar::FLOATING
		) )
	    {
	      for( CharIterator = i->children.begin()->value.begin();
		   CharIterator != i->children.begin()->value.end();
		   ++CharIterator )
		{
		  aString1 += *CharIterator;
		}

	      for( CharIterator = ( i->children.begin()+1 )->value.begin();
		   CharIterator != ( i->children.begin()+1 )->value.end();
		   ++CharIterator )
		{
		  aString2 += *CharIterator;
		}

	      value1 = stringCast<Real>( aString1 );
	      value2 = stringCast<Real>( aString2 );	  

	      ++theStackSize;

	      if (*i->value.begin() == '+')
		{
		  appendInstruction
		    ( aCode, Instruction<PUSH_REAL>( value1 + value2 ) );
		}	
	      else if (*i->value.begin() == '-')
		{
		  appendInstruction
		    ( aCode, Instruction<PUSH_REAL>( value1 - value2 ) );
		}
	      else
		THROW_EXCEPTION( UnexpectedError,
				 String( "Invalid operation" ) );
	    }
	  else
	    {
	      compileTree( i->children.begin(), aCode );
	      compileTree( i->children.begin()+1, aCode );
		
	      if (*i->value.begin() == '+')
		{
		  appendInstruction( aCode, Instruction<ADD>() );
		}
	      else if (*i->value.begin() == '-')
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


} // namespace libecs

#endif /* __EXPRESSIONCOMPILER_HPP */

