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


#include <functional>
#include <string>
#include <cassert>
#include <limits>
#include <cmath>

#include "boost/spirit/core.hpp"
#include "boost/spirit/tree/ast.hpp"

#include "Process.hpp"

using namespace boost::spirit;

namespace libecs
{

  LIBECS_DM_CLASS( ExpressionProcessBase, Process )
  {

  protected:
    
    class StackMachine;
    class Compiler;
    

    DECLARE_CLASS( Instruction );
    DECLARE_VECTOR( InstructionPtr, InstructionVector );
    
    class Instruction
    {
    public:
      Instruction() {}
      virtual ~Instruction() {}
      
      virtual void execute( StackMachine& aStackMachine ) = 0;
    };

   
    class PUSH : public Instruction
    {
    public:
      PUSH() {}
      PUSH( Real n ) { value = n; }
      virtual ~PUSH() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    private:
      Real value;
    };
  
    class NEG : public Instruction
    {
    public:
      NEG() {}
      virtual ~NEG() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class ADD : public Instruction
    {
    public:
      ADD() {}
      virtual ~ADD() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class SUB : public Instruction
    {
    public:
      SUB() {}
      virtual ~SUB() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class MUL : public Instruction
    {
    public:
      MUL() {}
      virtual ~MUL() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class DIV : public Instruction
    {
    public:
      DIV() {}
      virtual ~DIV() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class POW : public Instruction
    {
    public:
      POW() {}
      virtual ~POW() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class EXP : public Instruction
    {
    public:
      EXP() {}
      virtual ~EXP() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class CALL : public Instruction
    {
    public:
      CALL() {}
      CALL( VariableReferenceRef tmpVariableReference,
	    const Real(libecs::VariableReference::*function)() const )
      {
	theVariableReference = tmpVariableReference;
	func = function;
      }
      virtual ~CALL() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    private:
      VariableReference theVariableReference;
      const Real(libecs::VariableReference::*func)() const;
    };
  
    class METHOD : public Instruction
    {
    public:
      METHOD() {}
      METHOD( SystemPtr (libecs::VariableReference::*function1)() const,
	      const Int(libecs::Logger::*function2)() const )
      {
	func1 = function1;
	func2 = function2;
      }
      virtual ~METHOD() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    private:
      SystemPtr (libecs::VariableReference::*func1)() const;
      const Int(libecs::Logger::*func2)() const;
    };

    
    class StackMachine
    {
    public:
    
      StackMachine()
      {
	; // do nothing
      }
    
      ~StackMachine() {}
    
      const Real top() const 
      { 
	return theStackPtr[-1]; 
      }
    
      void resize( std::vector<int>::size_type aSize )
      {
	if( theStack.size() < aSize )
	  theStack.resize( aSize );
	//  theStackPtr = &theStack[0];
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
    
      void execute( InstructionVectorCref aCode )
      {
	reset();
	for( InstructionVectorConstIterator i( aCode.begin() );
	     i != aCode.end(); ++i )
	  {
	    (*i)->execute( *this );
	  }
      }
    
    private:
    
      std::vector<Real> theStack;
    
      RealPtr theStackPtr;
    };
  
  
  
    class Compiler
    {
    public:
    
      Compiler()
      {
	;// do nothing
      }
    
      ~Compiler()
      {
	;
      }
    
      typedef char const*         iterator_t;
      typedef tree_match<iterator_t> parse_tree_match_t;
      typedef parse_tree_match_t::tree_iterator TreeIterator;
      //    DECLARE_CLASS( TreeIterator );
    
      const InstructionVector compileExpression( StringCref anExpression )
      {
	InstructionVector aCode;
	calculator calc;

	aStackSize = 0;

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

      void setPropertyMap( std::map<String, Real> thePropertyMap )
      {
	aPropertyMap = thePropertyMap;
      }

      void setVariableReferenceMap( std::map<String, VariableReference> theVariableReferenceMap )
      {
	aVariableReferenceMap = theVariableReferenceMap;
      }

      const int getStackSize()
      {
	return aStackSize;
      }

    protected:

      int aStackSize;

      void compileTree( TreeIterator const&  i,
			InstructionVectorRef aCode );  

      std::map<String, Real> aPropertyMap;
      std::map<String, VariableReference> aVariableReferenceMap;
    
      struct calculator;    
    };


    struct Compiler::calculator : public grammar<calculator>
    {
      static const int groupID = 1;
      static const int integerID = 2;
      static const int floatingID = 3;
      static const int factorID = 4;
      static const int termID = 5;
      static const int expressionID = 6;
      static const int variableID = 7;
      static const int math_methodID = 8;
      static const int system_methodID = 9;
      static const int argumentsID = 10;
      static const int propertyID = 11;
      static const int objectID = 12;
      static const int attributeID = 13;
      static const int functionID = 13;
    
      template <typename ScannerT>
      struct definition
      {
	definition(calculator const& /*self*/)
	{
	  integer     =   leaf_node_d[ lexeme_d[ +digit_p ] ];
	  floating    =   leaf_node_d[ lexeme_d[ +digit_p >> ch_p('.') >> +digit_p ] ];
	
	  property    =   leaf_node_d[ lexeme_d[ +( alnum_p | ch_p('_') ) ] ];
	  object      =   leaf_node_d[ lexeme_d[ alpha_p >> *( alnum_p | ch_p('_') ) ] ];
	  attribute   =   leaf_node_d[ lexeme_d[ +alpha_p ] ];

	  variable    =   object >> root_node_d[ lexeme_d[ ch_p('.') ] ] >> attribute;
	
	  //   This syntax is made such dirty syntax    //
	  //   by the bug of Spirit                     //
                      
	  function = leaf_node_d[ lexeme_d[ str_p("getSuperSystem") ] ] >> arguments >> leaf_node_d[ lexeme_d[ ch_p('.') ] ];

	  system_method = object >> root_node_d[ lexeme_d[ ch_p('.') ] ] >> +function >> attribute;

	  math_method = (   root_node_d[ lexeme_d[ str_p("abs")] ]
			    | root_node_d[ lexeme_d[ str_p("sqrt")] ]
			    | root_node_d[ lexeme_d[ str_p("exp")] ]
			    | root_node_d[ lexeme_d[ str_p("log10")] ]
			    | root_node_d[ lexeme_d[ str_p("log")] ]
			    | root_node_d[ lexeme_d[ str_p("floor")] ]
			    | root_node_d[ lexeme_d[ str_p("ceil")] ]
			    | root_node_d[ lexeme_d[ str_p("fact")] ]
			    | root_node_d[ lexeme_d[ str_p("asinh")] ]
			    | root_node_d[ lexeme_d[ str_p("acosh")] ]
			    | root_node_d[ lexeme_d[ str_p("atanh")] ]
			    | root_node_d[ lexeme_d[ str_p("asech")] ]
			    | root_node_d[ lexeme_d[ str_p("acsch")] ]
			    | root_node_d[ lexeme_d[ str_p("acoth")] ]
			    | root_node_d[ lexeme_d[ str_p("sinh")] ]
			    | root_node_d[ lexeme_d[ str_p("cosh")] ]
			    | root_node_d[ lexeme_d[ str_p("tanh")] ]
			    | root_node_d[ lexeme_d[ str_p("sech")] ]
			    | root_node_d[ lexeme_d[ str_p("csch")] ]
			    | root_node_d[ lexeme_d[ str_p("coth")] ]
			    | root_node_d[ lexeme_d[ str_p("asin")] ]
			    | root_node_d[ lexeme_d[ str_p("acos")] ]
			    | root_node_d[ lexeme_d[ str_p("atan")] ]
			    | root_node_d[ lexeme_d[ str_p("asec")] ]
			    | root_node_d[ lexeme_d[ str_p("acsc")] ]
			    | root_node_d[ lexeme_d[ str_p("acot")] ]
			    | root_node_d[ lexeme_d[ str_p("sin")] ]
			    | root_node_d[ lexeme_d[ str_p("cos")] ]
			    | root_node_d[ lexeme_d[ str_p("tan")] ]
			    | root_node_d[ lexeme_d[ str_p("sec")] ]
			    | root_node_d[ lexeme_d[ str_p("csc")] ]
			    | root_node_d[ lexeme_d[ str_p("abs")] ]
			    | root_node_d[ lexeme_d[ str_p("cot")] ] ) >> arguments;
	
	  group       =   inner_node_d[ch_p('(') >> expression >> ch_p(')')];
	  arguments   =   inner_node_d[ch_p('(') >> infix_node_d[ *term >> *( ch_p(',') >> term ) ] >> ch_p(')') ];
	
	  factor      =   floating
	    |   integer
	    |   math_method
	    |   group
	    |   variable
	    |   system_method
	    |   property        //  added Constant 
	    |   (root_node_d[ch_p('-')] >> factor);
	
	  term        =  factor >>
	    *( (root_node_d[ch_p('*')] >> factor)
	       |  (root_node_d[ch_p('/')] >> factor)
	       |  (root_node_d[ch_p('^')] >> factor)
	       |  (root_node_d[ch_p('e') | ch_p('E')] >> ( factor | discard_first_node_d[ch_p('+') >> factor] ) ) );
	
	
	  expression  =  term >>
	    *( (root_node_d[ch_p('+')] >> term)
	       | (root_node_d[ch_p('-')] >> term) );
	}
      
	rule<ScannerT, parser_context, parser_tag<variableID> >     variable;
	rule<ScannerT, parser_context, parser_tag<math_methodID> >  math_method;
	rule<ScannerT, parser_context, parser_tag<system_methodID> >  system_method;
	rule<ScannerT, parser_context, parser_tag<expressionID> >   expression;
	rule<ScannerT, parser_context, parser_tag<termID> >         term;
	rule<ScannerT, parser_context, parser_tag<factorID> >       factor;
	rule<ScannerT, parser_context, parser_tag<floatingID> >     floating;
	rule<ScannerT, parser_context, parser_tag<integerID> >      integer;
	rule<ScannerT, parser_context, parser_tag<groupID> >        group;
	rule<ScannerT, parser_context, parser_tag<argumentsID> >    arguments;
	rule<ScannerT, parser_context, parser_tag<propertyID> >     property;
	rule<ScannerT, parser_context, parser_tag<objectID> >       object;
	rule<ScannerT, parser_context, parser_tag<attributeID> >    attribute;
	rule<ScannerT, parser_context, parser_tag<functionID> >    function;

	rule<ScannerT, parser_context, parser_tag<expressionID> > const&
	start() const { return expression; }
      };
    };
  


  public:
  
    LIBECS_DM_OBJECT_ABSTRACT( ExpressionProcessBase )
      {
	INHERIT_PROPERTIES( Process );

	PROPERTYSLOT_SET_GET( String, Expression );
      }


    ExpressionProcessBase()
      {
	;  // do nothing
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
	    theCompiledCode = aCompiler.compileExpression( aValue.asString() );
	    theStackMachine.resize( aCompiler.getStackSize() );
	    theStackMachine.execute( theCompiledCode );
	    thePropertyMap[ aPropertyName ] = theStackMachine.top();
	    aCompiler.setPropertyMap( thePropertyMap );
	  }
	else
	  THROW_EXCEPTION( NoSlot,
			   getClassName() +
			   String( ": No Property slot found by name [" )
			   + aPropertyName + "].  Set property failed." );
      } 
    
    virtual void initialize()
      {
	Process::initialize();

	std::map<String, VariableReference> theVariableReferenceMap;
    
	for( VariableReferenceVectorConstIterator
	       i( getVariableReferenceVector().begin() );
	     i != getVariableReferenceVector().end(); ++i )
	  {
	    VariableReferenceCref aVariableReference( *i );
	    
	    theVariableReferenceMap[ aVariableReference.getName() ]
	      = aVariableReference;
	  }

	aCompiler.setVariableReferenceMap( theVariableReferenceMap );
	
	//	aCompiler.setProperty_VariableReferenceMap( thePropertyMap );
	theCompiledCode = aCompiler.compileExpression( theExpression );
	theStackMachine.resize( aCompiler.getStackSize() );

      }

  protected:

    Compiler  aCompiler;
    String    theExpression;
      
    InstructionVector theCompiledCode;
    StackMachine theStackMachine;

    std::map<String, Real> thePropertyMap;
  };



  /**
     This function is Compiler subclass member function.
     This member function evaluates AST tree and makes binary codes.
  */
  
  void
  libecs::ExpressionProcessBase::Compiler::compileTree( TreeIterator const& i, InstructionVectorRef aCode )
  {
    /**
        std::cout << "In compileExpression. i->value = " <<
      String(i->value.begin(), i->value.end()) <<
      " i->children.size() = " << i->children.size() << std::endl;
    */

    /**
       Floating Grammer evaluation
    */

    if ( i->value.id() == calculator::floatingID )
      {
	assert(i->children.size() == 0);

	String str(i->value.begin(), i->value.end());
	Real n = strtod(str.c_str(), 0);

	aStackSize++;
	aCode.push_back( new PUSH( n ) );

	return;
      }
    

    /**
       Integer Grammer evaluation
    */

    else if ( i->value.id() == calculator::integerID )
      {
	assert(i->children.size() == 0);
	
	String str(i->value.begin(), i->value.end());
	Real n = strtod(str.c_str(), 0);

	aStackSize++;
	aCode.push_back( new PUSH( n ) );
	  
	return; 
      }
    

    /**
       Math_Method Grammer evaluation
    */

    else if ( i->value.id() == calculator::math_methodID )
      {
	String str(i->value.begin(), i->value.end() );
	//	 assert(i->children.size() == 1);

	if(i->children.size() == 1)
	  {
	    if( i->children.begin()->value.id() == calculator::integerID ||
		i->children.begin()->value.id() == calculator::floatingID  )
	      {
		aStackSize++;
		aCode.push_back( new PUSH() );
		return;
	      }
	    else
	      {
		compileTree(i->children.begin(), aCode);	  
		aCode.push_back( new METHOD() );
		
		return;
	      }
	  }
	else if(i->children.size() == 2)
	  {
	    ;
	  }
	
      }
    

    /**
       System_Method Grammer evaluation
    */

    else if (i->value.id() == calculator::system_methodID)
      {
	assert(*i->value.begin() == '.');

	String str1( (i->children.begin()+1)->value.begin(), (i->children.begin()+1)->value.end() );
	String str2( (i->children.begin()+3)->value.begin(), (i->children.begin()+3)->value.end() );
	
	if( str1 == "getSuperSystem" )
	  {
	    if( str2 == "Size" )
	      aCode.push_back( new METHOD( &libecs::VariableReference::getSuperSystem, &libecs::Logger::getSize ) );
	  }
	return;
      }


    /**
       Variable Grammer evaluation
    */

    else if (i->value.id() == calculator::variableID)
      {
	assert(*i->value.begin() == '.');	

	String str1( i->children.begin()->value.begin(), i->children.begin()->value.end() );
	String str2( ( i->children.begin()+1 )->value.begin(), ( i->children.begin()+1 )->value.end() );

	if(str2 == "MolarConc"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getMolarConc ) );
	}
	else if(str2 == "NumberConc"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getNumberConc ) );
	}
	else if(str2 == "Value"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getValue ) );
	}
	/**       	else if(str2 == "Coefficient"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getCoefficient ) );
	} 
	else if(str2 == "Fixed"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::isFixed ) );
	  }*/
	else if(str2 == "Volocity"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getVelocity ) );
	}
	else if(str2 == "TotalVelocity"){
	  aCode.push_back( new CALL( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getTotalVelocity ) );
	}
	return;
      }


    /**
       Property Grammer evaluation
    */

    else if (i->value.id() == calculator::propertyID)
      {
	assert(i->children.size() == 0);
	
	String str(i->value.begin(), i->value.end() );
	
	aStackSize++;

	if( str == "true")
	  aCode.push_back( new PUSH( 1.0 ) );
	else if( str == "false")
	  aCode.push_back( new PUSH( 0.0 ) );
	else if( str == "NaN")
	  aCode.push_back( new PUSH( std::numeric_limits<Real>::quiet_NaN() ) );
	else if( str == "pi")
	  aCode.push_back( new PUSH( M_PI ) );
	else if( str == "INF")
	  aCode.push_back( new PUSH( std::numeric_limits<Real>::infinity() ) );
	else if( str == "N_A")
	  aCode.push_back( new PUSH( N_A ) );
	else if( str == "exponential")
	  aCode.push_back( new PUSH( M_E ) );
	else
	  { 
	    std::map<String, Real>::iterator thePropertyMapIterator;
    
 	    thePropertyMapIterator = aPropertyMap.find( str );

	    if( thePropertyMapIterator != aPropertyMap.end() )
	      {
		aCode.push_back( new PUSH( thePropertyMapIterator->second ) );
	      }
	    else
	      {
		THROW_EXCEPTION( NoSlot,
	 			 str + String( " : No Property slot " ) );
	      }
	  }

	return;
      }


    /**
       Factor Grammer evaluation 
    */
    
    else if (i->value.id() == calculator::factorID)
      {
	assert(*i->value.begin() == '-');

	String str(i->children.begin()->value.begin(), i->children.begin()->value.end());
	Real n = strtod(str.c_str(), 0);

	if( i->children.begin()->value.id() == calculator::integerID ||
	    i->children.begin()->value.id() == calculator::floatingID  )
	  {
	    i->value.id() = calculator::floatingID;

	    aStackSize++;

	    aCode.push_back( new PUSH( -n ) ); 
	    return;
	  }
	else
	  {
	    compileTree(i->children.begin(), aCode);
	    aCode.push_back( new NEG() );
	      
	    return;
	  }
      }

    
    /**
       Term Grammer evaluation
    */

    else if (i->value.id() == calculator::termID)
      {
	assert(i->children.size() == 2);

	if( ( i->children.begin()->value.id() == calculator::integerID ||
	      i->children.begin()->value.id() == calculator::floatingID ||
	      i->children.begin()->value.id() == calculator::propertyID ) && 
	    ( ( i->children.begin()+1 )->value.id() == calculator::integerID ||
	      ( i->children.begin()+1 )->value.id() == calculator::floatingID ||
	      ( i->children.begin()+1 )->value.id() == calculator::propertyID ) )
	  {
	    i->value.id() = calculator::floatingID;

	    String str1(i->children.begin()->value.begin(), i->children.begin()->value.end());
	    String str2((i->children.begin()+1)->value.begin(), (i->children.begin()+1)->value.end());
	   
	    Real n1, n2;
	    int flg1=0; int flg2=0;

	    if(	i->children.begin()->value.id() == calculator::propertyID )
	      {
		flg1 = 1;
		
		if( str1 == "true") n1 = 1.0;
		else if( str1 == "false") n1 = 0.0;
		else if( str1 == "NaN")
		  n1 = std::numeric_limits<Real>::quiet_NaN();
		else if( str1 == "pi") n1 = M_PI;
		else if( str1 == "INF") n1 = std::numeric_limits<Real>::infinity();
		else if( str1 == "N_A") n1 = N_A;
		else if( str1 == "exponential") n1 = M_E;
		else
		  { 
		    std::map<String, Real>::iterator thePropertyMapIterator;
		    
		    thePropertyMapIterator = aPropertyMap.find( str1 );
		    if( thePropertyMapIterator != aPropertyMap.end() )
		      {
			n1 = thePropertyMapIterator->second;
		      }
		    else
		      {
			THROW_EXCEPTION( NoSlot,
					 str1 + String( " : No Property slot " ) );
		      }
		  }
	      }
	    
	    if( (i->children.begin()+1 )->value.id() == calculator::propertyID )
	      {
		flg2 = 1;

		if( str2 == "true") n2 = 1.0;
		else if( str2 == "false") n2 = 0.0;
		else if( str2 == "NaN")
		  n2 = std::numeric_limits<Real>::quiet_NaN();
		else if( str2 == "pi") n2 = M_PI;
		else if( str2 == "INF") n2 = std::numeric_limits<Real>::infinity();
		else if( str2 == "N_A") n2 = N_A;
		else if( str2 == "exponential") n2 = M_E;
		else
		  { 
		    std::map<String, Real>::iterator thePropertyMapIterator;
		    
		    thePropertyMapIterator = aPropertyMap.find( str2 );
		    if( thePropertyMapIterator != aPropertyMap.end() )
		      {
			n2 = thePropertyMapIterator->second;
		      }
		    else
		      {
			THROW_EXCEPTION( NoSlot,
					 str2 + String( " : No Property slot " ) );
		      }
		  }
	      }

	    if( flg1 == 0 && flg2 == 0){
	      n1 = strtod(str1.c_str(), 0);
	      n2 = strtod(str2.c_str(), 0);	  
	    }
	    else if( flg1 == 1 && flg2 == 0){
	      n2 = strtod(str2.c_str(), 0);
	    }
	    else if( flg1 == 0 && flg2 == 1){
	      n1 = strtod(str1.c_str(), 0);
	    }

	    aStackSize++;

	    if (*i->value.begin() == '*')
	      {
		aCode.push_back( new PUSH( n1 * n2 ) );
		return;
	      }	
	    else if (*i->value.begin() == '/')
	      {
		aCode.push_back( new PUSH( n1 / n2 ) ); 
		return;	      
	      }
	    else if (*i->value.begin() == '^')
	      {
		aCode.push_back( new PUSH( pow( n1, n2 ) ) ); 
		return;	      
	      }
	    else if (*i->value.begin() == 'e' || 'E')
	      {
		aCode.push_back( new PUSH( n1 * pow( 10, n2 ) ) );
		return;	      
	      }
	  }
	else
	  {
	    compileTree( i->children.begin(), aCode );
	    compileTree( ( i->children.begin()+1 ), aCode );
	    
	    if (*i->value.begin() == '*')
	      {
		aCode.push_back( new MUL() );
		return;
	      }
	    
	    else if (*i->value.begin() == '/')
	      {
		aCode.push_back( new DIV() );
		return;
	      }
	    else if (*i->value.begin() == '^')
	      {
		aCode.push_back( new POW() );
		return;
	      }
	    
	    else if (*i->value.begin() == 'e' || 'E')
	      {
		aCode.push_back( new EXP() );
		return;
	      }
	  }
      }

    
    /**
       Expression Grammer evaluation
    */

    else if (i->value.id() == calculator::expressionID)
      {
	assert(i->children.size() == 2);
	
	if( ( i->children.begin()->value.id() == calculator::integerID ||
	      i->children.begin()->value.id() == calculator::floatingID ||
	      i->children.begin()->value.id() == calculator::propertyID ) &&
	    ( ( i->children.begin()+1 )->value.id() == calculator::integerID ||
	      ( i->children.begin()+1 )->value.id() == calculator::floatingID ||
	      ( i->children.begin()+1 )->value.id() == calculator::propertyID ) )
	  {
	    i->value.id() = calculator::floatingID;

	    String str1(i->children.begin()->value.begin(), i->children.begin()->value.end());
	    String str2( ( i->children.begin()+1 )->value.begin(), ( i->children.begin()+1 )->value.end() );
	   
	    Real n1, n2;
	    int flg1=0; int flg2=0;

	    if(	i->children.begin()->value.id() == calculator::propertyID )
	      {
		flg1 = 1;
		
		if( str1 == "true") n1 = 1.0;
		else if( str1 == "false") n1 = 0.0;
		else if( str1 == "NaN")
		  n1 = std::numeric_limits<Real>::quiet_NaN();
		else if( str1 == "pi") n1 = M_PI;
		else if( str1 == "INF") n1 = std::numeric_limits<Real>::infinity();
		else if( str1 == "N_A") n1 = N_A;
		else if( str1 == "exponential") n1 = M_E;
		else
		  { 
		    std::map<String, Real>::iterator thePropertyMapIterator;
		    
		    thePropertyMapIterator = aPropertyMap.find( str1 );
		    if( thePropertyMapIterator != aPropertyMap.end() )
		      {
			n1 = thePropertyMapIterator->second;
		      }
		    else
		      {
			THROW_EXCEPTION( NoSlot,
					 str1 + String( " : No Property slot " ) );
		      }
		  }
	      }
	    
	    if( (i->children.begin()+1 )->value.id() == calculator::propertyID )
	      {
		flg2 = 1;
		
		if( str2 == "true") n2 = 1.0;
		else if( str2 == "false") n2 = 0.0;
		else if( str2 == "NaN")
		  n2 = std::numeric_limits<Real>::quiet_NaN();
		else if( str2 == "pi") n2 = M_PI;
		else if( str2 == "INF") n2 = std::numeric_limits<Real>::infinity();
		else if( str2 == "N_A") n2 = N_A;
		else if( str2 == "exponential") n2 = M_E;
		else
		  { 
		    std::map<String, Real>::iterator thePropertyMapIterator;
		    
		    thePropertyMapIterator = aPropertyMap.find( str2 );
		    if( thePropertyMapIterator != aPropertyMap.end() )
		      {
			n2 = thePropertyMapIterator->second;
		      }
		    else
		      {
			THROW_EXCEPTION( NoSlot,
					 str2 + String( " : No Property slot " ) );
		      }
		  }
	      }
	    
	    if( flg1 == 0 && flg2 == 0){
	      n1 = strtod(str1.c_str(), 0);
	      n2 = strtod(str2.c_str(), 0);	  
	    }
	    else if( flg1 == 1 && flg2 == 0){
	      n2 = strtod(str2.c_str(), 0);
	    }
	    else if( flg1 == 0 && flg2 == 1){
	      n1 = strtod(str1.c_str(), 0);
	    }

	    aStackSize++;

	    if (*i->value.begin() == '+')
	      {
		aCode.push_back( new PUSH( n1 + n2 ) );
		return;
	      }	
	    else if (*i->value.begin() == '-')
	      {
		aCode.push_back( new PUSH( n1 - n2 ) );
		return;	      
	      }
	  }
	else
	  {
	    compileTree(i->children.begin(), aCode);
	    compileTree( ( i->children.begin()+1 ), aCode );
		
	    if (*i->value.begin() == '+')
	      {
		aCode.push_back( new ADD() );
		return;
	      }
	    else if (*i->value.begin() == '-')
	      {
		aCode.push_back( new SUB() );
		return;
	      }
	  }
      }
    else
      {
	assert(0); // error
      }
    
    return;
  }
  


  /**
     Member function of the Instruction subclasses are defined here.
     This member function execute on the binary codes.
  */
  
  void ExpressionProcessBase::PUSH::execute( StackMachine& aStackMachine )
  {
    *aStackMachine.getStackPtr() = value;

    aStackMachine.getStackPtr()++;
  }
  
  void ExpressionProcessBase::NEG::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) = -( *( aStackMachine.getStackPtr()-1 ) );
  }
  
  void ExpressionProcessBase::ADD::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()--;
    
    *( aStackMachine.getStackPtr()-1 ) += *( aStackMachine.getStackPtr() );
  }
  
  void ExpressionProcessBase::SUB::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()--;
    
    *( aStackMachine.getStackPtr()-1 ) -= *( aStackMachine.getStackPtr() );
  }
  
  void ExpressionProcessBase::MUL::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()--;
    
    *( aStackMachine.getStackPtr()-1 ) *= *( aStackMachine.getStackPtr() );
  }
  
  void ExpressionProcessBase::DIV::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()--;
    
    *( aStackMachine.getStackPtr()-1 ) /= *( aStackMachine.getStackPtr() );
  }
  
  void ExpressionProcessBase::POW::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()--;
    
    *( aStackMachine.getStackPtr()-1 ) = pow( *( aStackMachine.getStackPtr()-1 ), *( aStackMachine.getStackPtr() ) );
  }
  
  void ExpressionProcessBase::EXP::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()--;
    
    *( aStackMachine.getStackPtr()-1 ) = *( aStackMachine.getStackPtr()-1 ) * pow(10, *( aStackMachine.getStackPtr() ) );
  }
  
  void ExpressionProcessBase::CALL::execute( StackMachine& aStackMachine )
  {
    *aStackMachine.getStackPtr() = (theVariableReference.*func)();
    aStackMachine.getStackPtr()++;
  }

  void ExpressionProcessBase::METHOD::execute( StackMachine& aStackMachine )
  {
    //    *aStackMachine.getStackPtr() = ( ( func1() ).*func2 )();
    aStackMachine.getStackPtr()++;
  }

  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, ExpressionFluxProcess );
  
} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */
