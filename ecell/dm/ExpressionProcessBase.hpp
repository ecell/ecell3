//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 2003 Keio University
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
//   Tatsuya Ishida
//
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//

#ifndef __EXPRESSIONPROCESSBASE_HPP
#define __EXPRESSIONPROCESSBASE_HPP


#include <cassert>
#include <limits>

#include "boost/spirit/core.hpp"
#include "boost/spirit/tree/ast.hpp"
#include "boost/shared_ptr.hpp"

#include "Process.hpp"

using namespace boost::spirit;

USE_LIBECS;

namespace libecs
{

  LIBECS_DM_CLASS( ExpressionProcessBase, Process )
  {

  protected:
    
    class StackMachine;
    class Compiler;

    DECLARE_CLASS( Instruction );
    DECLARE_CLASS( Code );
    DECLARE_VECTOR( Int, IntVector );
    DECLARE_VECTOR( Real, RealVector );
    DECLARE_VECTOR( InstructionPtr, InstructionVector );

    DECLARE_ASSOCVECTOR( String, Real, std::less<const String>, ConstantMap);
    DECLARE_ASSOCVECTOR( String, Real, std::less<const String>, PropertyMap);
    DECLARE_ASSOCVECTOR( String, Real(*)(Real), std::less<const String>, FunctionMap);

    typedef const Real (libecs::VariableReference::* VariableReferenceMethodPtr)() const;
    typedef SystemPtr (libecs::Process::* System_Func)() const;
    typedef const Real (libecs::System::* System_Attribute)() const;


    class Instruction
    {
    public:
      Instruction() {}
      virtual ~Instruction() {}
      
      virtual void initialize() = 0;
      virtual void execute( StackMachine& aStackMachine ) = 0;
    };

    class PUSH
      :
      public Instruction
    {
    public:
      PUSH( Real aValue ) 
	:
	theValue( aValue )
      { 
	; // do nothing
      }
      virtual ~PUSH() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );

    private:
      Real theValue;
    };
  
    class NEG
      :
      public Instruction
    {
    public:
      NEG() {}
      virtual ~NEG() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );    
    };
  
    class ADD
      :
      public Instruction
    {
    public:
      ADD() {}
      virtual ~ADD() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );    
    };
  
    class SUB
      :
      public Instruction
    {
    public:
      SUB() {}
      virtual ~SUB() {}
    
      virtual void initialize()
      {
	 //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );    
    };
  
    class MUL
      :
      public Instruction
    {
    public:
      MUL() {}
      virtual ~MUL() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );    
    };
  
    class DIV
      :
      public Instruction
    {
    public:
      DIV() {}
      virtual ~DIV() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );
    };
  
    class POW
      :
      public Instruction
    {
    public:
      POW() {}
      virtual ~POW() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );    
    };
  

    class CALL_FUNC
      :
      public Instruction
    {
    public:
      CALL_FUNC( Real (*aFuncPtr)(Real) )
		:
	theFuncPtr( aFuncPtr )
      {
	; // do nothing
      }
      virtual ~CALL_FUNC() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );

    private:
      Real (*theFuncPtr)(Real);
    };



    class CALL_PROPERTY
      :
      public Instruction
    {
    public:
      CALL_PROPERTY( PropertyMap &aPropertyMap, String aName )
      {
	thePropertyMapIterator = ( aPropertyMap ).find( aName ) ;
      }
      virtual ~CALL_PROPERTY() {}
    
      virtual void initialize()
      {
	theValue = thePropertyMapIterator->second;
      }

      virtual void execute( StackMachine& aStackMachine );

    private:
      Real theValue;
      PropertyMapIterator thePropertyMapIterator;
    };


    class VARREF_FUNC
      :
      public Instruction
    {
    public:
      VARREF_FUNC() {}
      VARREF_FUNC( VariableReference tmpVariableReference,
	    VariableReferenceMethodPtr aFuncPtr )
	:
	theVariableReference( tmpVariableReference ),
	theFuncPtr( aFuncPtr )
      {
	; // do nothing
      }
      virtual ~VARREF_FUNC() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );
    
    private:
      VariableReference theVariableReference;
      VariableReferenceMethodPtr theFuncPtr;
    };
    


    class SYSTEM_FUNC
      :
      public Instruction
    {
    public:
      SYSTEM_FUNC() {}
      SYSTEM_FUNC( ProcessPtr aProcessPtr,
		     System_Func aFuncPtr,
		     System_Attribute aAttributePtr )
	:
	theProcessPtr( aProcessPtr ), 
	theFuncPtr( aFuncPtr ),
	theAttributePtr( aAttributePtr )
      {
	; // do nothing
      }
      virtual ~SYSTEM_FUNC() {}
    
      virtual void initialize()
      {
	; //do nothing
      }

      virtual void execute( StackMachine& aStackMachine );
    
    private:
      ProcessPtr theProcessPtr;
      System_Func theFuncPtr;
      System_Attribute theAttributePtr;
    };

        
    class Code
      : 
      private InstructionVector
    {
    public: 
      Code()
      {
	; // do nothing;
      }
      
      ~Code() {}

      iterator begin() { return theCode.InstructionVector::begin(); }
      const_iterator begin() const { return theCode.InstructionVector::begin(); }
      iterator end() { return theCode.InstructionVector::end(); }
      const_iterator end() const { return theCode.InstructionVector::end(); }

      bool empty() const { return theCode.InstructionVector::empty(); }
      size_type size() const { return theCode.InstructionVector::size(); }
      size_type max_size() { return theCode.InstructionVector::max_size(); }

      void clear()
      {
	for( InstructionVectorConstIterator 
	       i( theCode.InstructionVector::begin() );
	     i != theCode.InstructionVector::end(); i++ )
	  {
	    delete *i;
	  }

	theCode.InstructionVector::clear();
      }

      void push_back( InstructionPtr anInstruction )
      {
	theCode.InstructionVector::push_back( anInstruction );
      }

    protected:
      InstructionVector theCode;
    };
    


    class StackMachine
    {
    public:
    
      StackMachine()
      {
	; // do nothing
      }
    
      ~StackMachine() {}
    
      void resize( IntVector::size_type aSize )
      {
	theStack.resize( aSize );
      }
    
      void reset()
      {
	theStackPtr = &theStack[0];
	theStack[0] = 0.0;
      }
    
      RealPtr& getStackPtr()
      {
	return theStackPtr;
      }
    
      const Real execute( CodeCref aCode )
      {
	reset();
 	for( InstructionVectorConstIterator i( aCode.begin() );
	     i != aCode.end(); ++i )
	  {
	    (*i)->execute( *this );
	  }

	return *theStackPtr;
      }
    
    protected:
    
      RealVector theStack;
    
      RealPtr theStackPtr;
    };
  
  
  
    class Compiler
    {
    public:
    
      Compiler()
      {
	if( theConstantMap.empty() == 1 )
	  {
	    fillConstantMap();
	  }
	if( theFunctionMap.empty() == 1 )
	  {
	    setFunctionMap();
	  }
      }
    
      ~Compiler()
      {
	;
      }
    
      typedef char const*         iterator_t;
      typedef tree_match<iterator_t> parse_tree_match_t;
      typedef parse_tree_match_t::tree_iterator TreeIterator;
      //    DECLARE_CLASS( TreeIterator );

      void setProcessPtr( ProcessPtr aProcessPtr )
      {
	theProcessPtr = aProcessPtr;
      }

      void setExpressionProcessBasePtr( ExpressionProcessBasePtr 
					aExpressionProcessBasePtr )
      {
	theExpressionProcessBasePtr = aExpressionProcessBasePtr;
      }

      const Int getStackSize()
      {
	return theStackSize;
      }
      
      void fillConstantMap()
      {
	theConstantMap[ "true" ] = 1.0;
	theConstantMap[ "false" ] = 0.0;
	theConstantMap[ "pi" ] = M_PI;
	theConstantMap[ "NaN" ] = std::numeric_limits<Real>::quiet_NaN();
	theConstantMap[ "INF"] = std::numeric_limits<Real>::infinity();
	theConstantMap[ "N_A" ] = N_A;
	theConstantMap[ "exp" ] = M_E;
      }
    
      void setFunctionMap();

      const Code compileExpression( StringCref anExpression )
      {
	Code aCode;
	CompileGrammar calc;
	
	theStackSize = 1;
	
	tree_parse_info<> 
	  info( ast_parse( anExpression.c_str(), calc, space_p ) );
	
	if( info.full )
	  {
	    compileTree( info.trees.begin(), aCode );
	  }
	else
	  {
	    THROW_EXCEPTION( UnexpectedError, "Input string parse error" );
	  }
	
	return aCode;
      }
      
    private:
      
      void compileTree( TreeIterator const&  i, CodeRef aCode );  

      class CompileGrammar;

    private:

      Int theStackSize;
      ProcessPtr theProcessPtr;
      ExpressionProcessBasePtr theExpressionProcessBasePtr;

      ConstantMap theConstantMap;
      FunctionMap theFunctionMap;
    };


    class Compiler::CompileGrammar : public grammar<CompileGrammar>
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
	   TERM,
	   EXPRESSION,
	   VARIABLE,
	   CALL_FUNC,
	   SYSTEM_FUNC,
	   ARGUMENT,
	   IDENTIFIER,
	   CONSTANT,
	};

      template <typename ScannerT>
      struct definition
      {
#define leafNode( str ) leaf_node_d[lexeme_d[str]]
#define rootNode( str ) root_node_d[lexeme_d[str]]
    
	definition(CompileGrammar const& /*self*/)
	{
	  integer     =   leafNode( +digit_p );
	  floating    =   leafNode( +digit_p >> ch_p('.') >> +digit_p );

	  exponent    =   ( integer | floating ) >>
	                  rootNode( ch_p('e') | ch_p('E') ) >>
	                  ( factor | discard_first_node_d[ ch_p('+') >> factor ] );
	  negative    =	  rootNode( ch_p('-') ) >> factor; 

	  identifier  =   leafNode( alpha_p >> *( alnum_p | ch_p('_') ) );

	  argument    =   inner_node_d[ ch_p('(') >> 
					infix_node_d[ *expression >>
						      *( ch_p(',') >>
							 expression ) ] >>
					ch_p(')') ];
	  
	  variable    =   identifier >>
	                  rootNode( ch_p('.') ) >>
	                  identifier;
	
 	  system_func = identifier >>
	                rootNode( ch_p('.') ) >>
	                +( leafNode( +( alpha_p | ch_p('_') ) ) >>
			  discard_node_d[ argument ] >>
			  discard_node_d[ ch_p('.') ] ) >>
	                  identifier;

	  ///////////////////////////////////////////////////
	  //                                               //
	  //      This syntax is made such dirty syntax    //
	  //      by the bug of Spirit                     //
          //                                               //
	  ///////////////////////////////////////////////////
            
	    call_func = (   rootNode( str_p("abs") )
			    | rootNode( str_p("sqrt") )
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
#ifndef __MINGW32__
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
#endif  			    
			    ) >> argument;

	  group       =   inner_node_d[ ch_p('(') >> expression >> ch_p(')')];
	
	  constant    =   exponent
	              |   floating
	              |   integer;

	  factor      =   call_func
	              |   system_func
	              |   group
	              |   variable 
	              |   constant
	              |   identifier
	              |   negative;
	
	  term        =  factor >>
	              *( ( rootNode( ch_p('*') ) >> factor )
	              |  ( rootNode( ch_p('/') ) >> factor )
	              |  ( rootNode( ch_p('^') ) >> factor ) );
	

	  expression  =  term >>
	              *( (rootNode( ch_p('+') ) >> term)
	              |  (rootNode( ch_p('-') ) >> term) );
	}
      
	rule<ScannerT, parser_context, parser_tag<VARIABLE> >     variable;
	rule<ScannerT, parser_context, parser_tag<CALL_FUNC> >    call_func;
	rule<ScannerT, parser_context, parser_tag<EXPRESSION> >   expression;
	rule<ScannerT, parser_context, parser_tag<TERM> >         term;
	rule<ScannerT, parser_context, parser_tag<FACTOR> >       factor;
	rule<ScannerT, parser_context, parser_tag<FLOATING> >     floating;
	rule<ScannerT, parser_context, parser_tag<EXPONENT> >     exponent;
	rule<ScannerT, parser_context, parser_tag<INTEGER> >      integer;
	rule<ScannerT, parser_context, parser_tag<NEGATIVE> >     negative;
	rule<ScannerT, parser_context, parser_tag<GROUP> >        group;
	rule<ScannerT, parser_context, parser_tag<ARGUMENT> >     argument;
	rule<ScannerT, parser_context, parser_tag<IDENTIFIER> >   identifier;
	rule<ScannerT, parser_context, parser_tag<CONSTANT> >     constant;
	rule<ScannerT, parser_context, parser_tag<SYSTEM_FUNC> >  system_func;

	rule<ScannerT, parser_context, parser_tag<EXPRESSION> > const&
	start() const { return expression; }
      };

#undef leafNode
#undef rootNode
 
    };
  


  public:

    LIBECS_DM_OBJECT_ABSTRACT( ExpressionProcessBase )
      {
	INHERIT_PROPERTIES( Process );

	PROPERTYSLOT_SET_GET( String, Expression );
      }


    ExpressionProcessBase()
      {
	;// do nothing
      }

    virtual ~ExpressionProcessBase()
      {
	;  // do nothing
      }

    SET_METHOD( String, Expression )
      {
	theExpression = value;
      }

    GET_METHOD( String, Expression )
      {
	return theExpression;
      }

    void defaultSetProperty( StringCref aPropertyName,
			     PolymorphCref aValue)
      {
	if( getClassName() == "ExpressionFluxProcess")
	  {
	    thePropertyMap[ aPropertyName ] = aValue.asReal();
	    
	    for( InstructionVectorConstIterator i( theCompiledCode.begin() );
		 i != theCompiledCode.end(); ++i )
	      {
		(*i)->initialize();
	      }
	    //	    ( static_cast<PropertiedClass*>( this ) )->setProperty( aPropertyName, aValue );
	  }
	else
	  THROW_EXCEPTION( NoSlot,
			   getClassName() +
			   String( ": No Property slot found by name [" )
			   + aPropertyName + "].  Set property failed." );
      } 
    
    virtual void initialize()
      {
	Compiler theCompiler;

	Process::initialize();

	theCompiler.setProcessPtr( static_cast<Process*>( this ) );
	theCompiler.setExpressionProcessBasePtr( this );

	theCompiledCode.clear();
	theCompiledCode = theCompiler.compileExpression( theExpression );

	theStackMachine.resize( theCompiler.getStackSize() );

	for( InstructionVectorConstIterator i( theCompiledCode.begin() );
	     i != theCompiledCode.end(); ++i )
	  {
	    (*i)->initialize();
	  }
      }

  protected:

    String    theExpression;
      
    Code theCompiledCode;
    StackMachine theStackMachine;

    PropertyMap thePropertyMap;
  };

  
  
  void libecs::ExpressionProcessBase::Compiler::setFunctionMap()
  {
    theFunctionMap["abs"] = fabs;
    theFunctionMap["sqrt"] = sqrt;
    theFunctionMap["exp"] = exp;
    theFunctionMap["log10"] = log10;
    theFunctionMap["log"] = log;
    theFunctionMap["floor"] = floor;
    theFunctionMap["ceil"] = ceil;
    theFunctionMap["sin"] = sin;
    theFunctionMap["cos"] = cos;
    theFunctionMap["tan"] = tan;
    theFunctionMap["sinh"] = sinh;
    theFunctionMap["cosh"] = cosh;
    theFunctionMap["tanh"] = tanh;
    theFunctionMap["asin"] = asin;
    theFunctionMap["acos"] = acos;
    theFunctionMap["atan"] = atan;
    #ifndef __MINGW32__
    theFunctionMap["fact"] = fact;
    theFunctionMap["asinh"] = asinh;
    theFunctionMap["acosh"] = acosh;
    theFunctionMap["atanh"] = atanh;
    theFunctionMap["asech"] = asech;
    theFunctionMap["acsch"] = acsch;
    theFunctionMap["acoth"] = acoth;
    theFunctionMap["sech"] = sech;
    theFunctionMap["csch"] = csch;
    theFunctionMap["coth"] = coth;
    theFunctionMap["asec"] = asec;
    theFunctionMap["acsc"] = acsc;
    theFunctionMap["acot"] = acot;
    theFunctionMap["sec"] = sec;
    theFunctionMap["csc"] = csc;
    theFunctionMap["cot"] = cot;
    #endif   
  }


  /**
     This function is Compiler subclass member function.
     This member function evaluates AST tree and makes binary codes.
  */
  
  void
  libecs::ExpressionProcessBase::Compiler::compileTree( TreeIterator const& i, CodeRef theCode )
  {
    /**
        std::cout << "In compileExpression. i->value = " <<
      String(i->value.begin(), i->value.end()) <<
      " i->children.size() = " << i->children.size() << std::endl;
    */

    Real n,n1,n2;
    String str, str_child1, str_child2, str_child3;
    VariableReference aVariableReference;

    ConstantMapIterator theConstantMapIterator;
    PropertyMapIterator thePropertyMapIterator;
    FunctionMapIterator theFunctionMapIterator;
	    
    std::vector<char>::iterator container_iterator;


    switch ( i->value.id().to_long() )
      {
	/**
	   Floating Grammar compile
	*/

      case CompileGrammar::FLOATING :
	
	assert(i->children.size() == 0);
	
	for( container_iterator = i->value.begin();
	     container_iterator != i->value.end(); container_iterator++ )
	  {
	    str += *container_iterator;
	  }
	
	n = stringTo<Real>( str.c_str() );
    
	theStackSize++;
	theCode.push_back( new PUSH( n ) );

	return;
      
    

	/**
	   Integer Grammar compile
	*/

      case CompileGrammar::INTEGER :
	
	assert(i->children.size() == 0);

	for( container_iterator = i->value.begin();
	     container_iterator != i->value.end(); container_iterator++ )
	  {	  
	    str += *container_iterator;
	  }

	n = stringTo<Real>( str.c_str() );
	
	theStackSize++;
	theCode.push_back( new PUSH( n ) );
	  
	return; 
	
	
	/**
	   Grammar compile
	*/

      case CompileGrammar::EXPONENT:
	
	assert( *i->value.begin() == 'E' || *i->value.begin() == 'e' );
	
	for( container_iterator = i->children.begin()->value.begin();
	     container_iterator != i->children.begin()->value.end();
	     container_iterator++ )
	  {
	    str_child1 += *container_iterator;
	  }

	for( container_iterator = ( i->children.begin()+1 )->value.begin();
	     container_iterator != ( i->children.begin()+1 )->value.end();
	     container_iterator++ )
	  {
	    str_child2 += *container_iterator;
	  }
	
	n1 = stringTo<Real>( str_child1.c_str() );
	n2 = stringTo<Real>( str_child2.c_str() );
	
	theStackSize++;
	theCode.push_back( new PUSH( n1 * pow(10, n2) ) );
	
	return; 
	
    

	/**
	   Call_Func Grammar compile
	*/

      case CompileGrammar::CALL_FUNC :
	
	assert( i->children.size() != 0 );
	
	theStackSize++;
	if( i->children.size() == 1 )
	  {
	    for( container_iterator = i->value.begin();
		 container_iterator != i->value.end(); container_iterator++ )
	      {
		str += *container_iterator;
	      }

	    theFunctionMapIterator = theFunctionMap.find( str );
	    
	    if( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
		i->children.begin()->value.id() == CompileGrammar::FLOATING  )
	      {
		for( container_iterator = i->children.begin()->value.begin();
		     container_iterator != i->children.begin()->value.end();
		     container_iterator++ )
		  {
		    str_child1 += *container_iterator;
		  }

		n = stringTo<Real>( str_child1.c_str() );
		
		if( theFunctionMapIterator != theFunctionMap.end() )
		  {
		    theCode.push_back( new PUSH( ( *theFunctionMapIterator->second )( n ) ) );
		  }
		else
		  {
		    THROW_EXCEPTION( NoSlot, str + String( " : No Function " ) );
		  }
	      }
	    else
	      {
		compileTree( i->children.begin(), theCode );	  
		
		if( theFunctionMapIterator != theFunctionMap.end() )
		  {
		    theCode.push_back( new CALL_FUNC( theFunctionMapIterator->second ) );
		  }
		else
		  {
		    THROW_EXCEPTION( NoSlot, str + String( " : No Function " ) );
		  }
	      }
	  }
	
	else if( i->children.size() >= 2 )
	  {
	    THROW_EXCEPTION( NoSlot, str + String( " : No Function or isn't mounted " ) );
	    return;
	  }
	
	return;
	
	

	/**
	   System_Func Grammar compile
	*/

      case CompileGrammar::SYSTEM_FUNC :
      
	theStackSize++;
	
	for( container_iterator = i->children.begin()->value.begin();
	     container_iterator != i->children.begin()->value.end();
	     container_iterator++ )
	  {
	    str_child1 += *container_iterator;
	  }

	for( container_iterator = ( i->children.begin()+1 )->value.begin();
	     container_iterator != ( i->children.begin()+1 )->value.end();
	     container_iterator++ )
	  {
	    str_child2 += *container_iterator;
	  }
	
	assert( str_child1 == "self" );
	assert( *i->value.begin() == '.' );
	
	if( str_child2 == "getSuperSystem" )
	  {
	    for( container_iterator = ( i->children.begin()+2 )->value.begin();
		 container_iterator != ( i->children.begin()+2 )->value.end();
		 container_iterator++ )
	      {
		str_child3 += *container_iterator;
	      }

	    if( str_child3 == "Size" )
	      theCode.push_back(   
		  new SYSTEM_FUNC( theProcessPtr,
                                    &libecs::Process::getSuperSystem,
				    &libecs::System::getSize ) );
	    else if( str_child3 == "SizeN_A" )
	      theCode.push_back( 
		  new SYSTEM_FUNC( theProcessPtr,
				    &libecs::Process::getSuperSystem,
				    &libecs::System::getSizeN_A ) );
	    else
	      THROW_EXCEPTION( NoSlot,
			       str_child3 + String( " : No System method or isn't mounted" ) );
	  }
	else
	  THROW_EXCEPTION( NoSlot,
			   str_child2 + String( " : No Process method or isn't mounted" ) );
	return;
	

	/**
	   Variable Grammar compile
	*/

      case CompileGrammar::VARIABLE :
	
	assert( *i->value.begin() == '.' );

	for( container_iterator = i->children.begin()->value.begin();
	     container_iterator != i->children.begin()->value.end();
	     container_iterator++ )
	  {
	    str_child1 += *container_iterator;
	  }

	for( container_iterator = ( i->children.begin()+1 )->value.begin();
	     container_iterator != ( i->children.begin()+1 )->value.end();
	     container_iterator++ )
	  {
	    str_child2 += *container_iterator;
	  }

	aVariableReference = theProcessPtr->libecs::Process::getVariableReference( str_child1 );
	
	if( str_child2 == "MolarConc" ){
	  theCode.push_back( 
	      new VARREF_FUNC( aVariableReference,
				&libecs::VariableReference::getMolarConc ) );
	  }
	else if( str_child2 == "NumberConc" ){
	  theCode.push_back(
	      new VARREF_FUNC( aVariableReference,
				&libecs::VariableReference::getNumberConc ) );
	  }
	else if( str_child2 == "Value" ){
	  theCode.push_back(
              new VARREF_FUNC( aVariableReference,
				&libecs::VariableReference::getValue ) );
	}
	/**       	else if( str_child2 == "Coefficient" ){
	  theCode.push_back(
              new VARREF_FUNC( aVariableReference,
	                        &libecs::VariableReference::getCoefficient ) );
	} 
	else if( str_child2 == "Fixed" ){
	  theCode.push_back(
	      new VARREF_FUNC( aVariableReference,
	                        &libecs::VariableReference::isFixed ) );
	  }*/
	else if( str_child2 == "Volocity" ){
	  theCode.push_back(
              new VARREF_FUNC( aVariableReference,
				&libecs::VariableReference::getVelocity ) );
	}
	else if( str_child2 == "TotalVelocity" ){
	  theCode.push_back(
              new VARREF_FUNC( aVariableReference,
				&libecs::VariableReference::getTotalVelocity ) );
	}
	else
	  THROW_EXCEPTION( NoSlot,
			   str_child2 + String( " : No VariableReferencePtr method or isn't mounted" ) );	  
	return;
      


	/**
	   Identifier Grammar compile
	*/

      case CompileGrammar::IDENTIFIER :
	
	assert( i->children.size() == 0 );
	
	theStackSize++;

	for( container_iterator = i->value.begin();
	     container_iterator != i->value.end(); container_iterator++ )
	  {
	    str += *container_iterator;
	  }

	theConstantMapIterator = theConstantMap.find( str );
	thePropertyMapIterator = ( theExpressionProcessBasePtr->thePropertyMap).find( str );
	
	if( theConstantMapIterator != theConstantMap.end() )
	  {
	    theCode.push_back( new PUSH( theConstantMapIterator->second ) );
	  }
	else if( thePropertyMapIterator != ( (theExpressionProcessBasePtr->thePropertyMap).end() ) )
	  {
	    theCode.push_back( new CALL_PROPERTY( theExpressionProcessBasePtr->thePropertyMap, str ) );
	    //	    theCode.push_back( InstructPtr ( new CALL_PROPERTY( thePropertyMapIterator->second ) ) );
	  }
	else
	  {
	    THROW_EXCEPTION( NoSlot,
			     str + String( " : No Property slot " ) );
	  }
	
	return;
      


	/**
	   Negative Grammar compile 
	*/
    
      case CompileGrammar::NEGATIVE :
			       
	assert( *i->value.begin() == '-' );

	for( container_iterator = i->children.begin()->value.begin();
	     container_iterator != i->children.begin()->value.end();
	     container_iterator++ )
	  {
	    str_child1 += *container_iterator;
	  }

	n = stringTo<Real>( str_child1.c_str() );

	if( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
	    i->children.begin()->value.id() == CompileGrammar::FLOATING  )
	  {
	    theStackSize++;
	    theCode.push_back( new PUSH( -n ) ); 
	  }
	else
	  {
	    compileTree(i->children.begin(), theCode );
	    theCode.push_back( new NEG() );
	  }
	return;
      

    
	/**
	   Term Grammar compile
	*/

      case CompileGrammar::TERM :
	
	assert(i->children.size() == 2);

	if( ( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
	      i->children.begin()->value.id() == CompileGrammar::FLOATING ) && 
	    ( ( i->children.begin()+1 )->value.id() == CompileGrammar::INTEGER ||
	      ( i->children.begin()+1 )->value.id() == CompileGrammar::FLOATING ) )
	  {
	    for( container_iterator = i->children.begin()->value.begin();
		 container_iterator != i->children.begin()->value.end();
		 container_iterator++ )
	      {
		str_child1 += *container_iterator;
	      }

	    for( container_iterator = ( i->children.begin()+1 )->value.begin();
		 container_iterator != ( i->children.begin()+1 )->value.end();
		 container_iterator++ )
	      {
		str_child2 += *container_iterator;
	      }

	    n1 = stringTo<Real>( str_child1.c_str() );
	    n2 = stringTo<Real>( str_child2.c_str() );	  

	    theStackSize++;

	    if (*i->value.begin() == '*')
	      {
		theCode.push_back( new PUSH( n1 * n2 ) );
	      }	
	    else if (*i->value.begin() == '/')
	      {
		theCode.push_back( new PUSH( n1 / n2 ) ); 
	      }
	    else if (*i->value.begin() == '^')
	      {
		theCode.push_back( new PUSH( pow( n1, n2 ) ) ); 
	      }
	    else if (*i->value.begin() == 'e' || 'E')
	      {
		theCode.push_back( new PUSH( n1 * pow( 10, n2 ) ) );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );

	    return;
	  }
	else
	  {
	    compileTree( i->children.begin(), theCode );
	    compileTree( i->children.begin()+1, theCode );
	    
	    if (*i->value.begin() == '*')
	      {
		theCode.push_back( new MUL() );
	      }
	    
	    else if (*i->value.begin() == '/')
	      {
		theCode.push_back( new DIV() );
	      }
	    else if (*i->value.begin() == '^')
	      {
		theCode.push_back( new POW() );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );

	    return;
	  }

	return;
      

    
	/**
	   Expression Grammar compile
	*/

      case CompileGrammar::EXPRESSION :
      
	assert(i->children.size() == 2);
	
	if( ( i->children.begin()->value.id() == CompileGrammar::INTEGER ||
	      i->children.begin()->value.id() == CompileGrammar::FLOATING ) &&
	    ( ( i->children.begin()+1 )->value.id() == CompileGrammar::INTEGER ||
	      ( i->children.begin()+1 )->value.id() == CompileGrammar::FLOATING ) )
	  {
	    for( container_iterator = i->children.begin()->value.begin();
		 container_iterator != i->children.begin()->value.end();
		 container_iterator++ )
	      {
		str_child1 += *container_iterator;
	      }

	    for( container_iterator = ( i->children.begin()+1 )->value.begin();
		 container_iterator != ( i->children.begin()+1 )->value.end();
		 container_iterator++ )
	      {
		str_child2 += *container_iterator;
	      }

	    n1 = stringTo<Real>( str_child1.c_str() );
	    n2 = stringTo<Real>( str_child2.c_str() );	  

	    theStackSize++;

	    if (*i->value.begin() == '+')
	      {
		theCode.push_back( new PUSH( n1 + n2 ) );
	      }	
	    else if (*i->value.begin() == '-')
	      {
		theCode.push_back( new PUSH( n1 - n2 ) );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );
	  }
	else
	  {
	    compileTree( i->children.begin(), theCode );
	    compileTree( i->children.begin()+1, theCode );
		
	    if (*i->value.begin() == '+')
	      {
		theCode.push_back( new ADD() );
	      }
	    else if (*i->value.begin() == '-')
	      {
		theCode.push_back( new SUB() );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );
	  }

	return;
	

      default :
	THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );
	
	return;
      }
  }

  /**
     Member function of the Instruction subclasses are defined here.
     This member function execute on the binary codes.
  */
  
  void ExpressionProcessBase::PUSH::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;

    *aStackMachine.getStackPtr() = theValue;
  }
  
  void ExpressionProcessBase::NEG::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr() ) = - *( aStackMachine.getStackPtr() );
  }
  
  void ExpressionProcessBase::ADD::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) += *( aStackMachine.getStackPtr() );

    aStackMachine.getStackPtr()--;
  }
  
  void ExpressionProcessBase::SUB::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) -= *( aStackMachine.getStackPtr() );

    aStackMachine.getStackPtr()--;
  }
  
  void ExpressionProcessBase::MUL::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) *= *( aStackMachine.getStackPtr() );

    aStackMachine.getStackPtr()--;
  }
  
  void ExpressionProcessBase::DIV::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) /= *( aStackMachine.getStackPtr() );

    aStackMachine.getStackPtr()--;
  }
  
  void ExpressionProcessBase::POW::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) = pow( *( aStackMachine.getStackPtr()-1 ), *( aStackMachine.getStackPtr() ) );

    aStackMachine.getStackPtr()--;
  }
  
  void ExpressionProcessBase::CALL_FUNC::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr() ) = ( *theFuncPtr )( *( aStackMachine.getStackPtr() ) );
  }

  void ExpressionProcessBase::CALL_PROPERTY::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;

    *aStackMachine.getStackPtr() = theValue;
  }
  
  void ExpressionProcessBase::VARREF_FUNC::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;

    *aStackMachine.getStackPtr() = ( theVariableReference.*theFuncPtr )();
  }

  void ExpressionProcessBase::SYSTEM_FUNC::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;
    
    *aStackMachine.getStackPtr() = ( ( theProcessPtr->*theFuncPtr )()->*theAttributePtr)();
  }
  
  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, ExpressionFluxProcess );
  
} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */
