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


//#include <functional>
//#include <string>
#include <cassert>
#include <limits>
//#include <cmath>

#include "boost/spirit/core.hpp"
#include "boost/spirit/tree/ast.hpp"

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
    DECLARE_VECTOR( InstructionPtr, InstructionVector );

    typedef const Real (libecs::VariableReference::* theFunc)() const;
    typedef SystemPtr (libecs::Process::* System_Func)() const;
    typedef const Real (libecs::System::* System_Attribute)() const;


    class Instruction
    {
    public:
      Instruction() {}
      virtual ~Instruction() {}
      
      virtual void execute( StackMachine& aStackMachine ) = 0;
    };

    class CODE
      :
      public Instruction
    {
    public:
      CODE() {}
      virtual ~CODE() {}
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
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class ADD
      :
      public Instruction
    {
    public:
      ADD() {}
      virtual ~ADD() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class SUB
      :
      public Instruction
    {
    public:
      SUB() {}
      virtual ~SUB() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class MUL
      :
      public Instruction
    {
    public:
      MUL() {}
      virtual ~MUL() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class DIV
      :
      public Instruction
    {
    public:
      DIV() {}
      virtual ~DIV() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class POW
      :
      public Instruction
    {
    public:
      POW() {}
      virtual ~POW() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    };
  
    class EXP
      :
      public Instruction
    {
    public:
      EXP() {}
      virtual ~EXP() {}
    
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
    
      virtual void execute( StackMachine& aStackMachine );

    private:
      Real (*theFuncPtr)(Real);
    };

    class VARREF_METHOD
      :
      public Instruction
    {
    public:
      VARREF_METHOD() {}
      VARREF_METHOD( VariableReferenceRef tmpVariableReference,
	    theFunc aFuncPtr )
	:
	theVariableReference( tmpVariableReference ),
	theFuncPtr( aFuncPtr )
      {
	; // do nothing
      }
      virtual ~VARREF_METHOD() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    private:
      VariableReference theVariableReference;
      theFunc theFuncPtr;
    };
  
    class SYSTEM_METHOD
      :
      public Instruction
    {
    public:
      SYSTEM_METHOD() {}
      SYSTEM_METHOD( ProcessPtr aProcessPtr,
	      System_Func aFuncPtr,
	      System_Attribute aAttributePtr )
	:
	theProcessPtr( aProcessPtr ), 
	theFuncPtr( aFuncPtr ),
	theAttributePtr( aAttributePtr )
      {
	; // do nothing
      }
      virtual ~SYSTEM_METHOD() {}
    
      virtual void execute( StackMachine& aStackMachine );
    
    private:
      ProcessPtr theProcessPtr;
      System_Func theFuncPtr;
      System_Attribute theAttributePtr;
    };

    
    class StackMachine
    {
    public:
    
      StackMachine()
      {
	; // do nothing
      }
    
      ~StackMachine() {}
    
      void resize( std::vector<Int>::size_type aSize )
      {
	//	if( theStack.size() < aSize )
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
    
      const Real execute( InstructionVectorCref aCode )
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
	CompileGrammer calc;

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

      void setPropertyMap( std::map<String, Real> thePropertyMap )
      {
	thePropertyMap[ "true" ] = 1.0;
	thePropertyMap[ "false" ] = 0.0;
	thePropertyMap[ "pi" ] = M_PI;
	thePropertyMap[ "NaN" ] = std::numeric_limits<Real>::quiet_NaN();
	thePropertyMap[ "INF"] = std::numeric_limits<Real>::infinity();
	thePropertyMap[ "N_A" ] = N_A;
	thePropertyMap[ "exp" ] = M_E;

	aPropertyMap = thePropertyMap;
      }

      void setVariableReferenceMap( std::map<String, VariableReference> theVariableReferenceMap )
      {
	aVariableReferenceMap = theVariableReferenceMap;
      }

      void setProcessPtr( ProcessPtr aProcessPtr )
      {
	theProcessPtr = aProcessPtr;
      }

      const Int getStackSize()
      {
	return theStackSize;
      }

      void setFunctionMap();

    private:

      void compileTree( TreeIterator const&  i,
			InstructionVectorRef aCode );  

      class CompileGrammer;

    private:

      Int theStackSize;
      ProcessPtr theProcessPtr;

      std::map<String, Real> aPropertyMap;
      std::map<String, Real(*)(Real)> aFunctionMap;
      std::map<String, VariableReference> aVariableReferenceMap;
    };


    class Compiler::CompileGrammer : public grammar<CompileGrammer>
    {
    public:
      enum GrammerType
	{
	   groupID = 1,
	   integerID,
	   floatingID,
	   factorID,
	   termID,
	   expressionID,
	   variableID,
	   call_funcID,
	   system_methodID,
	   argumentsID,
	   propertyID,
	   objectID,
	   attributeID,
	};

      template <typename ScannerT>
      struct definition
      {
	definition(CompileGrammer const& /*self*/)
	{
	  integer     =   leaf_node_d[ lexeme_d[ +digit_p ] ];
	  floating    =   leaf_node_d[ lexeme_d[ +digit_p >> ch_p('.') >> +digit_p ] ];
	
	  property    =   leaf_node_d[ lexeme_d[ +( alnum_p | ch_p('_') ) ] ];
	  object      =   leaf_node_d[ lexeme_d[ alpha_p >> *( alnum_p | ch_p('_') ) ] ];
	  attribute   =   leaf_node_d[ lexeme_d[ +( alpha_p | ch_p('_') ) ] ];

	  variable    =   object >> root_node_d[ lexeme_d[ ch_p('.') ] ] >> attribute;
	
	  //   This syntax is made such dirty syntax    //
	  //   by the bug of Spirit                     //
                      
 	  system_method = object >> root_node_d[ lexeme_d[ ch_p('.') ] ] >> +( leaf_node_d[ lexeme_d[ +( alpha_p | ch_p('_') ) ] ] >> discard_node_d[ arguments ] >> discard_node_d[ ch_p('.') ] ) >> attribute;

	    call_func = (   root_node_d[ lexeme_d[ str_p("abs")] ]
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
	
	  group       =   inner_node_d[ ch_p('(') >> expression >> ch_p(')')];
	  arguments   =   inner_node_d[ ch_p('(') >> infix_node_d[ *expression >> *( ch_p(',') >> expression ) ] >> ch_p(')') ];
	
	  factor      =   floating
	              |   integer
	              |   call_func
	              |   system_method
	              |   group
	              |   variable
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
	rule<ScannerT, parser_context, parser_tag<call_funcID> >  call_func;
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
	    thePropertyMap[ aPropertyName ] = aValue.asReal();
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

	for( VariableReferenceVectorConstIterator
	       i( getVariableReferenceVector().begin() );
	     i != getVariableReferenceVector().end(); ++i )
	  {
	    VariableReferenceCref aVariableReference( *i );
	    
	    theVariableReferenceMap[ aVariableReference.getName() ]
	      = aVariableReference;
	  }

	aCompiler.setVariableReferenceMap( theVariableReferenceMap );
	aCompiler.setProcessPtr( static_cast<Process*>(this) );

	theCompiledCode = aCompiler.compileExpression( theExpression );
	theStackMachine.resize( aCompiler.getStackSize() );
      }

  protected:

    Compiler  aCompiler;
    String    theExpression;
      
    InstructionVector theCompiledCode;
    StackMachine theStackMachine;

    std::map<String, Real> thePropertyMap;
    std::map<String, VariableReference> theVariableReferenceMap;
  };



  void libecs::ExpressionProcessBase::Compiler::setFunctionMap()
  {
    aFunctionMap["abs"] = fabs;      aFunctionMap["sqrt"] = sqrt;
    aFunctionMap["exp"] = exp;       aFunctionMap["log10"] = log10;
    aFunctionMap["log"] = log;       aFunctionMap["floor"] = floor;
    aFunctionMap["ceil"] = ceil;     //aFunctionMap["fact"] = fact;
    aFunctionMap["asinh"] = asinh;   aFunctionMap["acosh"] = acosh;
    aFunctionMap["atanh"] = atanh;   //aFunctionMap["asech"] = ?;
    //aFunctionMap["acsch"] = ?;     //aFunctionMap["acoth"] = ?;
    aFunctionMap["sinh"] = sinh;     aFunctionMap["cosh"] = cosh;
    aFunctionMap["tanh"] = tanh;     //aFunctionMap["sech"] = ?;
    //aFunctionMap["csch"] = ?;      //aFunctionMap["coth"] = ?;
    aFunctionMap["asin"] = asin;     aFunctionMap["acos"] = acos;
    aFunctionMap["atan"] = atan;     //aFunctionMap["asec"] = ?;
    //aFunctionMap["acsc"] = ?;      //aFunctionMap["acot"] = ?;
    aFunctionMap["sin"] = sin;       aFunctionMap["cos"] = cos;
    aFunctionMap["tan"] = tan;       //aFunctionMap["sec"] = ?;
    //aFunctionMap["csc"] = ?;       //aFunctionMap["cot"] = ?;
  }


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

    if ( i->value.id() == CompileGrammer::floatingID )
      {
	assert(i->children.size() == 0);

	String str(i->value.begin(), i->value.end());
	Real n = stringTo<Real>( str.c_str() );

	theStackSize++;
	aCode.push_back( new PUSH( n ) );

	return;
      }
    

    /**
       Integer Grammer evaluation
    */

    else if ( i->value.id() == CompileGrammer::integerID )
      {
	assert(i->children.size() == 0);

	String str(i->value.begin(), i->value.end());
	Real n = stringTo<Real>( str.c_str() );

	theStackSize++;
	aCode.push_back( new PUSH( n ) );
	  
	return; 
      }
    

    /**
       Call_Func Grammer evaluation
    */

    else if ( i->value.id() == CompileGrammer::call_funcID )
      {
	assert( i->children.size() == 0 );

	String str( i->value.begin(), i->value.end() );

	theStackSize++;
	if( i->children.size() == 1 )
	  {
	    setFunctionMap();

 	    std::map<String, Real(*)(Real)>::iterator theFunctionMapIterator;
	    theFunctionMapIterator = aFunctionMap.find( str );

	    if( i->children.begin()->value.id() == CompileGrammer::integerID ||
		i->children.begin()->value.id() == CompileGrammer::floatingID  )
	      {
		String str1( i->children.begin()->value.begin(), i->children.begin()->value.end() );
		Real n = stringTo<Real>( str1.c_str() );

		if( theFunctionMapIterator != aFunctionMap.end() )
		  {
		    aCode.push_back( new PUSH( ( *theFunctionMapIterator->second )(n) ) );
		  }
		else
		  {
		    THROW_EXCEPTION( NoSlot, str + String( " : No Function " ) );
		  }
	      }
	    else
	      {
		compileTree( i->children.begin(), aCode );	  

		if( theFunctionMapIterator != aFunctionMap.end() )
		  {
		    aCode.push_back( new CALL_FUNC( theFunctionMapIterator->second ) );
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
      }
    

    /**
       System_Method Grammer evaluation
    */

    else if (i->value.id() == CompileGrammer::system_methodID)
      {
	theStackSize++;

	String str( i->children.begin()->value.begin(), i->children.begin()->value.end() );
	String str1( (i->children.begin()+1)->value.begin(), (i->children.begin()+1)->value.end() );
	String str2( (i->children.begin()+2)->value.begin(), (i->children.begin()+2)->value.end() );

	assert( str == "self" );
	assert( *i->value.begin() == '.' );

	if( str1 == "getSuperSystem" )
	  {
	    if( str2 == "getSize" )
	      aCode.push_back( new SYSTEM_METHOD( theProcessPtr, &libecs::Process::getSuperSystem, &libecs::System::getSize ) );
	    else if( str2 == "getSizeN_A" )
	      aCode.push_back( new SYSTEM_METHOD( theProcessPtr, &libecs::Process::getSuperSystem, &libecs::System::getSizeN_A ) );
	    else
	      THROW_EXCEPTION( NoSlot,
			       str2 + String( " : No System method or isn't mounted" ) );
	  }
	else
	  THROW_EXCEPTION( NoSlot,
			   str1 + String( " : No Process method or isn't mounted" ) );
	return;
      }


    /**
       Variable Grammer evaluation
    */

    else if ( i->value.id() == CompileGrammer::variableID )
      {
	assert(*i->value.begin() == '.');

	String str1( i->children.begin()->value.begin(), i->children.begin()->value.end() );
	String str2( ( i->children.begin()+1 )->value.begin(), ( i->children.begin()+1 )->value.end() );

	if(str2 == "MolarConc"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getMolarConc ) );
	}
	else if(str2 == "NumberConc"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getNumberConc ) );
	}
	else if(str2 == "Value"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getValue ) );
	}
	/**       	else if(str2 == "Coefficient"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getCoefficient ) );
	} 
	else if(str2 == "Fixed"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::isFixed ) );
	  }*/
	else if(str2 == "Volocity"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getVelocity ) );
	}
	else if(str2 == "TotalVelocity"){
	  aCode.push_back( new VARREF_METHOD( aVariableReferenceMap[ str1 ], &libecs::VariableReference::getTotalVelocity ) );
	}
	else
	  THROW_EXCEPTION( NoSlot,
			   str2 + String( " : No VariableReference method or isn't mounted" ) );	  
	return;
      }


    /**
       Property Grammer evaluation
    */

    else if (i->value.id() == CompileGrammer::propertyID)
      {
	assert( i->children.size() == 0 );
	
	String str( i->value.begin(), i->value.end() );
	
	theStackSize++;

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
	
	return;
      }


    /**
       Factor Grammer evaluation 
    */
    
    else if (i->value.id() == CompileGrammer::factorID)
      {
	assert(*i->value.begin() == '-');

	String str(i->children.begin()->value.begin(), i->children.begin()->value.end());
	Real n = stringTo<Real>( str.c_str() );

	if( i->children.begin()->value.id() == CompileGrammer::integerID ||
	    i->children.begin()->value.id() == CompileGrammer::floatingID  )
	  {
	    theStackSize++;
	    aCode.push_back( new PUSH( -n ) ); 
	  }
	else
	  {
	    compileTree(i->children.begin(), aCode);
	    aCode.push_back( new NEG() );
	  }
	return;
      }

    
    /**
       Term Grammer evaluation
    */

    else if (i->value.id() == CompileGrammer::termID)
      {
	assert(i->children.size() == 2);

	if( ( i->children.begin()->value.id() == CompileGrammer::integerID ||
	      i->children.begin()->value.id() == CompileGrammer::floatingID ) && 
	    ( ( i->children.begin()+1 )->value.id() == CompileGrammer::integerID ||
	      ( i->children.begin()+1 )->value.id() == CompileGrammer::floatingID ) )
	  {
	    String str1(i->children.begin()->value.begin(), i->children.begin()->value.end());
	    String str2((i->children.begin()+1)->value.begin(), (i->children.begin()+1)->value.end());
	   
	    Real n1 = stringTo<Real>( str1.c_str() );
	    Real n2 = stringTo<Real>( str2.c_str() );	  

	    theStackSize++;

	    if (*i->value.begin() == '*')
	      {
		aCode.push_back( new PUSH( n1 * n2 ) );
	      }	
	    else if (*i->value.begin() == '/')
	      {
		aCode.push_back( new PUSH( n1 / n2 ) ); 
	      }
	    else if (*i->value.begin() == '^')
	      {
		aCode.push_back( new PUSH( pow( n1, n2 ) ) ); 
	      }
	    else if (*i->value.begin() == 'e' || 'E')
	      {
		aCode.push_back( new PUSH( n1 * pow( 10, n2 ) ) );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );

	    return;
	  }
	else
	  {
	    compileTree( i->children.begin(), aCode );
	    compileTree( ( i->children.begin()+1 ), aCode );
	    
	    if (*i->value.begin() == '*')
	      {
		aCode.push_back( new MUL() );
	      }
	    
	    else if (*i->value.begin() == '/')
	      {
		aCode.push_back( new DIV() );
	      }
	    else if (*i->value.begin() == '^')
	      {
		aCode.push_back( new POW() );
	      }
	    else if (*i->value.begin() == 'e' || 'E')
	      {
		aCode.push_back( new EXP() );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );

	    return;
	  }

	return;
      }

    
    /**
       Expression Grammer evaluation
    */

    else if (i->value.id() == CompileGrammer::expressionID)
      {
	assert(i->children.size() == 2);
	
	if( ( i->children.begin()->value.id() == CompileGrammer::integerID ||
	      i->children.begin()->value.id() == CompileGrammer::floatingID ) &&
	    ( ( i->children.begin()+1 )->value.id() == CompileGrammer::integerID ||
	      ( i->children.begin()+1 )->value.id() == CompileGrammer::floatingID ) )
	  {
	    String str1(i->children.begin()->value.begin(), i->children.begin()->value.end());
	    String str2( ( i->children.begin()+1 )->value.begin(), ( i->children.begin()+1 )->value.end() );
	   
	    Real n1 = stringTo<Real>( str1.c_str() );
	    Real n2 = stringTo<Real>( str2.c_str() );	  

	    theStackSize++;

	    if (*i->value.begin() == '+')
	      {
		aCode.push_back( new PUSH( n1 + n2 ) );
	      }	
	    else if (*i->value.begin() == '-')
	      {
		aCode.push_back( new PUSH( n1 - n2 ) );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );
	  }
	else
	  {
	    compileTree(i->children.begin(), aCode);
	    compileTree( ( i->children.begin()+1 ), aCode );
		
	    if (*i->value.begin() == '+')
	      {
		aCode.push_back( new ADD() );
	      }
	    else if (*i->value.begin() == '-')
	      {
		aCode.push_back( new SUB() );
	      }
	    else
	      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );
	  }

	return;
      }

    else
      THROW_EXCEPTION( NoSlot, String( " unexpected error " ) );
    
    return;
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
  
  void ExpressionProcessBase::EXP::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 ) = *( aStackMachine.getStackPtr()-1 ) * pow(10, *( aStackMachine.getStackPtr() ) );

    aStackMachine.getStackPtr()--;
  }
  
  void ExpressionProcessBase::CALL_FUNC::execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr() ) = ( *theFuncPtr )( *( aStackMachine.getStackPtr() ) );
  }

  void ExpressionProcessBase::VARREF_METHOD::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;

    *aStackMachine.getStackPtr() = ( theVariableReference.*theFuncPtr )();
  }

  void ExpressionProcessBase::SYSTEM_METHOD::execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;

    *aStackMachine.getStackPtr() = ( ( theProcessPtr->*theFuncPtr )()->*theAttributePtr)();
  }

  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, ExpressionFluxProcess );
  
} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */
