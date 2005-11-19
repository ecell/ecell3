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
//   Tatsuya Ishida
//
// E-Cell Project.
//

#ifndef __EXPRESSIONPROCESSBASE_HPP
#define __EXPRESSIONPROCESSBASE_HPP

#define EXPRESSION_PROCESS_USE_JIT 0

#define ENABLE_STACKOPS_FOLDING 1


#include <cassert>
#include <limits>

#include "ExpressionCompiler.hpp"

#include "jit/jit.h"
//#include "jit/jit-plus.h"
#include "jit/jit-type.h"
#include "jit/jit-insn.h"

//#if defined( EXPRESSIONPROCESS_USE_JIT )
//#include "JITExpressionProcessBase"
//#else /* defined( EXPRESSIONPROCESS_USE_JIT ) */
//#include "SVMExpressionProcessBase"
//#endif /* defined( EXPRESSIONPROCESS_USE_JIT ) */


USE_LIBECS;

DECLARE_ASSOCVECTOR
( Real(*)(Real),
  jit_value_t(*)( jit_function_t, jit_value_t ),
  std::less<Real(*)(Real)>,
  JITFunctionMap1 );

DECLARE_ASSOCVECTOR
( Real(*)(Real,Real),
  jit_value_t(*)( jit_function_t, jit_value_t, jit_value_t ),
  std::less<Real(*)(Real,Real)>,
  JITFunctionMap2 );


LIBECS_DM_CLASS( ExpressionProcessBase, Process )
{
 protected:

  DECLARE_TYPE( ExpressionCompiler::Code, Code ); 

  typedef void* Pointer;
  typedef Real (*RealFunc0)();

  class VirtualMachine
  {
    union StackElement_
    {
      Real    theReal;
      Pointer thePointer;
      Integer theInteger;
    };
    
    DECLARE_TYPE( StackElement_, StackElement );
    
  public:
    
    VirtualMachine()
      :
      theContext( jit_context_create() )
    {
      // ; do nothing
    }
    
    ~VirtualMachine() 
    {
      jit_context_destroy( theContext );
    }
    
    const Real execute( CodeCref aCode );

    const Real execute()
    {
      jit_function_apply( JIT_Function, NULL, &result );
      //std::cerr << "Result : " << result << std::endl;
      return result;
    }

    void initialize( CodeCref aCode )
    { 
      jit_context_build_start( theContext );
      
      jit_type_t main_signature( jit_type_create_signature
				 (jit_abi_cdecl, jit_type_float64, 
				  NULL, 0, 1 ) );
      
      JIT_Function = jit_function_create( theContext, main_signature );

      jit_function_set_recompilable( JIT_Function );
      //jit_function_set_on_demand_compiler( JIT_Function, &buildJITCode );

      fillJITFunctionMap();
      buildJITCode( aCode );

      jit_context_build_end( theContext );
    }

    void recompile()
    {      
      jit_function_set_optimization_level
	( JIT_Function,
	  jit_function_get_max_optimization_level() );
      
      jit_function_compile( JIT_Function );

      jit_function_clear_recompilable( JIT_Function );
    }


  protected:

    static Real VariableMethodFunction( void* aRealObjectMethodPtr )
    { 
      return ( *( static_cast<ExpressionCompiler::RealObjectMethodProxy*>( aRealObjectMethodPtr ) ) )();
    }

    static void printFunction( void )
    {
      std::cerr << "testtest" << std::endl;
    }

    void buildJITCode( CodeCref aCode );

    void fillJITFunctionMap();

  protected:
    jit_float64 result;
    jit_function_t JIT_Function;
    jit_context_t theContext;

    JITFunctionMap1 theJITFunctionMap1;
    JITFunctionMap2 theJITFunctionMap2;
  };


 public:

  LIBECS_DM_OBJECT_ABSTRACT( ExpressionProcessBase )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( String, Expression );
    }


  ExpressionProcessBase()
    :
    theRecompileFlag( true )
    {
      // ; do nothing
    }

  virtual ~ExpressionProcessBase()
    {
      // ; do nothing
    }

  SET_METHOD( String, Expression )
    {
      theExpression = value;
      theRecompileFlag = true;
    }

  GET_METHOD( String, Expression )
    {
      return theExpression;
    }

  virtual void defaultSetProperty( StringCref aPropertyName,
				   PolymorphCref aValue )
    {
      thePropertyMap[ aPropertyName ] = aValue.asReal();
    } 


  void compileExpression()
    {
      ExpressionCompiler theCompiler( this, &( getPropertyMap() ) );

      theCompiledCode.clear();
      theCompiledCode = theCompiler.compileExpression( theExpression );

      //	theVirtualMachine.resize( theCompiler.getStackSize() );
    }

  PropertyMapCref getPropertyMap() const
    {
      return thePropertyMap;
    }

  virtual void initialize()
    {
      Process::initialize();

      if( theRecompileFlag )
	{
	  compileExpression();
	  theRecompileFlag = false;
	}

      theVirtualMachine.initialize( theCompiledCode );
      theVirtualMachine.recompile();
   }
  
  

 protected:

  PropertyMapRef getPropertyMap()
    {
      return thePropertyMap;
    }

 protected:

  String theExpression;      
  Code theCompiledCode;
  bool theRecompileFlag;
  PropertyMap thePropertyMap;

  VirtualMachine theVirtualMachine;
};



void ExpressionProcessBase::VirtualMachine::fillJITFunctionMap()
{
  theJITFunctionMap1[std::sin]   = jit_insn_sin;
  theJITFunctionMap1[std::cos]   = jit_insn_cos;
  theJITFunctionMap1[std::tan]   = jit_insn_tan;
  theJITFunctionMap1[std::asin]  = jit_insn_asin;
  theJITFunctionMap1[std::acos]  = jit_insn_acos;
  theJITFunctionMap1[std::atan]  = jit_insn_atan;
  theJITFunctionMap1[std::sinh]  = jit_insn_sinh;
  theJITFunctionMap1[std::cosh]  = jit_insn_cosh;
  theJITFunctionMap1[std::tanh]  = jit_insn_tanh;
  theJITFunctionMap1[std::fabs]  = jit_insn_abs;
  theJITFunctionMap1[std::sqrt]  = jit_insn_sqrt;
  theJITFunctionMap1[std::exp]   = jit_insn_exp;
  theJITFunctionMap1[std::log10] = jit_insn_log10;
  theJITFunctionMap1[std::log]   = jit_insn_log;
  theJITFunctionMap1[std::floor] = jit_insn_floor;
  theJITFunctionMap1[std::ceil]  = jit_insn_ceil;
  theJITFunctionMap1[libecs::real_not]  = jit_insn_not;

  theJITFunctionMap2[std::pow]         = jit_insn_pow;
  theJITFunctionMap2[libecs::real_and] = jit_insn_and;
  theJITFunctionMap2[libecs::real_or]  = jit_insn_or;
  theJITFunctionMap2[libecs::real_xor] = jit_insn_xor;
  theJITFunctionMap2[libecs::real_eq]  = jit_insn_eq;
  theJITFunctionMap2[libecs::real_neq] = jit_insn_ne;
  theJITFunctionMap2[libecs::real_gt]  = jit_insn_gt;
  theJITFunctionMap2[libecs::real_lt]  = jit_insn_lt;
  theJITFunctionMap2[libecs::real_geq] = jit_insn_ge;
  theJITFunctionMap2[libecs::real_leq] = jit_insn_le;
}



#define FETCH_OPCODE()\
    reinterpret_cast<const ExpressionCompiler::InstructionHead* const>( aPC )\
      ->getOpcode() 

#define DECODE_INSTRUCTION( OPCODE )\
    typedef ExpressionCompiler::\
      Opcode2Instruction<ExpressionCompiler::OPCODE>::type CurrentInstruction;\
    const CurrentInstruction* const anInstruction\
      ( reinterpret_cast<const CurrentInstruction* const>( aPC ) )



#define INCREMENT_PC( OPCODE )\
    aPC += sizeof( ExpressionCompiler::\
                   Opcode2Instruction<ExpressionCompiler::OPCODE>::type );\
    LIBECS_PREFETCH( aPC, 0, 1 );



// build new jit code
void ExpressionProcessBase::VirtualMachine::buildJITCode( CodeCref aCode )
{
  jit_value_t theStack[30];

  int aStackIterator( 0 );
  int aMaxStackSize( 0 );

  const unsigned char* aPC( &aCode[0] );

  const jit_type_t aType( jit_type_void_ptr );
  jit_type_t RealObjectMethod_Signature
    ( jit_type_create_signature(jit_abi_cdecl, jit_type_float64, 
				(jit_type_t*)&aType, 1, 1 ) );

  jit_type_t test_Signature
    ( jit_type_create_signature(jit_abi_cdecl, jit_type_void_ptr, 
				NULL, 0, 0 ) );

    
#define CREATE_JIT_REAL_VALUE( VALUE )\
    jit_value_t aTempValue;\
    aTempValue = jit_value_create_float64_constant( JIT_Function,\
						    jit_type_float64,\
						    VALUE );
  
#define CREATE_JIT_VALUE()\
    if( aMaxStackSize < aStackIterator )\
       {\
	 theStack[aStackIterator] =\
	   jit_value_create( JIT_Function, jit_type_float64 );\
	 aMaxStackSize = aStackIterator;\
       }
    

  std::cerr << "Build !! " << std::endl;

  while( 1 )
    {
      switch ( FETCH_OPCODE() )
	{
	case ExpressionCompiler::PUSH_REAL:
	  {
	    jit_insn_call_native
	      ( JIT_Function,
		"printFunction",
		(void*)ExpressionProcessBase::VirtualMachine::printFunction,
		test_Signature, NULL, 0, JIT_CALL_NOTHROW );	    

	    DECODE_INSTRUCTION( PUSH_REAL );
	    
	    ++aStackIterator;
	    CREATE_JIT_VALUE();

	    CREATE_JIT_REAL_VALUE( anInstruction->getOperand() );
	    
	    jit_insn_store( JIT_Function, theStack[aStackIterator], aTempValue );
	    
	    INCREMENT_PC( PUSH_REAL );
	    continue;
	  }
	    
	case ExpressionCompiler::LOAD_REAL:
	  {   
	    DECODE_INSTRUCTION( LOAD_REAL );

	    ++aStackIterator;
	    CREATE_JIT_VALUE();

	    CREATE_JIT_REAL_VALUE( *( anInstruction->getOperand() ) );

	    jit_insn_store( JIT_Function, theStack[aStackIterator], aTempValue );

	    INCREMENT_PC( LOAD_REAL );
	    continue;
	  }

	case ExpressionCompiler::OBJECT_METHOD_REAL:
	  {
	    DECODE_INSTRUCTION( OBJECT_METHOD_REAL );
	      
	    ++aStackIterator;
	    CREATE_JIT_VALUE();

	    jit_value_t aTempValue
	      ( jit_value_create_nint_constant
		( JIT_Function,
		  jit_type_nint,
		  (int)( &(anInstruction->getOperand()) ) ) );
	    
	    jit_insn_store( JIT_Function,
			    theStack[aStackIterator],
			    jit_insn_call_native
			    ( JIT_Function,
			      "VariableMethodFunction",
			      (void*)
			      ExpressionProcessBase::VirtualMachine::
			      VariableMethodFunction,
			      RealObjectMethod_Signature, &aTempValue,
			      1, JIT_CALL_NOTHROW ) );

	    INCREMENT_PC( OBJECT_METHOD_REAL );
	    continue;
	  }

	case ExpressionCompiler::OBJECT_METHOD_INTEGER:
	  {
	    DECODE_INSTRUCTION( OBJECT_METHOD_INTEGER );
	     
	    ++aStackIterator;
	    CREATE_JIT_VALUE();

	    jit_value_t aTempValue
	      ( jit_value_create_nint_constant
		( JIT_Function,
		  jit_type_nint,
		  (int)( &(anInstruction->getOperand()) ) ) );
	    
	    jit_insn_store( JIT_Function,
			    theStack[aStackIterator],
			    jit_insn_call_native
			    ( JIT_Function,
			      "VariableMethodFunction",
			      (void*)
			      ExpressionProcessBase::VirtualMachine::
			      VariableMethodFunction,
			      RealObjectMethod_Signature, &aTempValue,
			      1, JIT_CALL_NOTHROW ) );

	    INCREMENT_PC( OBJECT_METHOD_INTEGER );
	    continue;
	  }

	case ExpressionCompiler::CALL_FUNC2:
	  {
	    DECODE_INSTRUCTION( CALL_FUNC2 );

	    JITFunctionMap2Iterator aJITFunctionMap2Iterator;

	    aJITFunctionMap2Iterator = 
	      theJITFunctionMap2.find( anInstruction->getOperand() );

	    if( aJITFunctionMap2Iterator != theJITFunctionMap2.end() )
	      {
		jit_insn_store( JIT_Function,
				theStack[aStackIterator-1],
				( (aJITFunctionMap2Iterator->second)
				  ( JIT_Function, 
				    theStack[aStackIterator-1],
				    theStack[aStackIterator] ) ) );
	      }
	    else
	      {
		/**THROW_EXCEPTION( NoSlot, 
				 aFunctionString +
				 String( " : No such function." ) );*/
	      }

	    --aStackIterator;

	    INCREMENT_PC( CALL_FUNC2 );
	    continue;
	  }


	case ExpressionCompiler::CALL_FUNC1:
	  {
	    DECODE_INSTRUCTION( CALL_FUNC1 );

	    JITFunctionMap1Iterator aJITFunctionMap1Iterator;

	    aJITFunctionMap1Iterator = 
	      theJITFunctionMap1.find( anInstruction->getOperand() );

	    if( aJITFunctionMap1Iterator != theJITFunctionMap1.end() )
	      {
		jit_insn_store( JIT_Function, 
				theStack[aStackIterator],
				( (*aJITFunctionMap1Iterator->second)
				  ( JIT_Function, 
				    theStack[aStackIterator] ) ) );
	      }
	    else
	      {
		/**THROW_EXCEPTION( NoSlot, 
				 aFunctionString +
				 String( " : No such function." ) );*/
	      }

	    INCREMENT_PC( CALL_FUNC1 );
	    continue;
	  }


	case ExpressionCompiler::NEG:
	  {
	    jit_insn_store( JIT_Function,
			    theStack[aStackIterator],
			    jit_insn_neg( JIT_Function, theStack[aStackIterator] ) );
	    
	    INCREMENT_PC( NEG );
	    continue;
	  }


#undef CREATE_JIT_VALUE
#undef CREATE_JIT_REAL_VALUE


	case ExpressionCompiler::ADD:
	  {
	    jit_insn_store( JIT_Function, 
			    theStack[aStackIterator-1],
			    jit_insn_add( JIT_Function, 
					  theStack[aStackIterator-1],
					  theStack[aStackIterator] ) );
	    
	    INCREMENT_PC( ADD );
	    --aStackIterator;

	    continue;
	  }

	case ExpressionCompiler::SUB:
	  {
	    jit_insn_store( JIT_Function,
			    theStack[aStackIterator-1],
			    jit_insn_sub( JIT_Function,
					  theStack[aStackIterator-1],
					  theStack[aStackIterator] ) );

	    INCREMENT_PC( SUB );
	    --aStackIterator;

	    continue;
	  }

	case ExpressionCompiler::MUL:
	  {
	    jit_insn_store( JIT_Function,
			    theStack[aStackIterator-1],
			    jit_insn_mul( JIT_Function, 
					  theStack[aStackIterator-1],
					  theStack[aStackIterator] ) );

	    INCREMENT_PC( MUL );
	    --aStackIterator;

	    continue;
	  }

	case ExpressionCompiler::DIV:
	  {
	    jit_insn_store( JIT_Function,
			    theStack[aStackIterator-1],
			    jit_insn_div( JIT_Function, 
					  theStack[aStackIterator-1],
					  theStack[aStackIterator] ) );
	    
	    INCREMENT_PC( DIV );
	    --aStackIterator;

	    continue;
	  }


#if 0
	case ExpressionCompiler::PUSH_INTEGER:
	  {
	    DECODE_INSTRUCTION( PUSH_INTEGER );

	    ++aStackIterator;
	    aStackIterator->theInteger = anInstruction->getOperand();

	    INCREMENT_PC( PUSH_INTEGER );
	    continue;
	  }

	case ExpressionCompiler::PUSH_POINTER:
	  {
	    DECODE_INSTRUCTION( PUSH_POINTER );

	    ++aStackIterator;
	    aStackIterator->thePointer = anInstruction->getOperand();
	      
	    INCREMENT_PC( PUSH_POINTER );
	    continue;
	  }

#endif // 0

	case ExpressionCompiler::RET:
	  {
	    jit_insn_return( JIT_Function, theStack[aStackIterator] );

	    jit_function_compile( JIT_Function );

	    return;
	  }

	default:
	  {
	    THROW_EXCEPTION( UnexpectedError, "Invalid instruction." );
	  }

	}
    }
}

  


const Real ExpressionProcessBase::VirtualMachine::execute( CodeCref aCode )
{
  StackElement aStack[100];
  //  aStack[0].theReal = 0.0;
  StackElementPtr aStackPtr( aStack - 1 );

  const unsigned char* aPC( &aCode[0] );

  while( 1 )
    {

      Real bypass;

      switch ( FETCH_OPCODE() )
	{

#define SIMPLE_ARITHMETIC( OPCODE, OP )\
	    ( aStackPtr - 1)->theReal OP##= aStackPtr->theReal;\
	    INCREMENT_PC( OPCODE );\
	    --aStackPtr

	  /*
            const Real aTopValue( aStackPtr->theReal );\
	    INCREMENT_PC( OPCODE );\
	    ( aStackPtr - 1 )->theReal OP##= aTopValue;\
	    --aStackPtr;\
	  */

	case ExpressionCompiler::ADD:
	  {
	    SIMPLE_ARITHMETIC( ADD, + );

	    continue;
	  }

	case ExpressionCompiler::SUB:
	  {
	    SIMPLE_ARITHMETIC( SUB, - );

	    continue;
	  }

	case ExpressionCompiler::MUL:
	  {
	    SIMPLE_ARITHMETIC( MUL, * );

	    continue;
	  }

	case ExpressionCompiler::DIV:
	  {
	    SIMPLE_ARITHMETIC( DIV, / );

	    continue;
	  }

#undef SIMPLE_ARITHMETIC

	case ExpressionCompiler::CALL_FUNC2:
	  {
	    DECODE_INSTRUCTION( CALL_FUNC2 );

	    ( aStackPtr - 1 )->theReal
	      = ( anInstruction->getOperand() )( ( aStackPtr - 1 )->theReal, 
						 aStackPtr->theReal );
	    --aStackPtr;

	    INCREMENT_PC( CALL_FUNC2 );
	    continue;
	  }


	case ExpressionCompiler::CALL_FUNC1:
	  {
	    DECODE_INSTRUCTION( CALL_FUNC1 );

	    aStackPtr->theReal
	      = ( anInstruction->getOperand() )( aStackPtr->theReal );

	    INCREMENT_PC( CALL_FUNC1 );
	    continue;
	  }

	case ExpressionCompiler::NEG:
	  {
	    aStackPtr->theReal = - aStackPtr->theReal;

	    INCREMENT_PC( NEG );
	    continue;
	  }

#if 0
	case ExpressionCompiler::PUSH_INTEGER:
	  {
	    DECODE_INSTRUCTION( PUSH_INTEGER );

	    ++aStackPtr;
	    aStackPtr->theInteger = anInstruction->getOperand();

	    INCREMENT_PC( PUSH_INTEGER );
	    continue;
	  }

	case ExpressionCompiler::PUSH_POINTER:
	  {
	    DECODE_INSTRUCTION( PUSH_POINTER );

	    ++aStackPtr;
	    aStackPtr->thePointer = anInstruction->getOperand();
	      
	    INCREMENT_PC( PUSH_POINTER );
	    continue;
	  }

#endif // 0

	case ExpressionCompiler::PUSH_REAL:
	  {
	    DECODE_INSTRUCTION( PUSH_REAL );

	    bypass = anInstruction->getOperand();

	    INCREMENT_PC( PUSH_REAL );
	    goto bypass_real;
	  }
	    
	case ExpressionCompiler::LOAD_REAL:
	  {   
	    DECODE_INSTRUCTION( LOAD_REAL );

	    bypass = *( anInstruction->getOperand() );

	    INCREMENT_PC( LOAD_REAL );
	    goto bypass_real;
	  }

	case ExpressionCompiler::OBJECT_METHOD_REAL:
	  {
	    DECODE_INSTRUCTION( OBJECT_METHOD_REAL );
	      
	    bypass = ( anInstruction->getOperand() )();

	    INCREMENT_PC( OBJECT_METHOD_REAL );
	    goto bypass_real;
	  }

	case ExpressionCompiler::OBJECT_METHOD_INTEGER:
	  {
	    DECODE_INSTRUCTION( OBJECT_METHOD_INTEGER );
	      
	    bypass = static_cast<Real>( ( anInstruction->getOperand() )() );

	    INCREMENT_PC( OBJECT_METHOD_INTEGER );
	    goto bypass_real;
	  }

	case ExpressionCompiler::RET:
	  {
	    return aStackPtr->theReal;
	  }

	default:
	  {
	    THROW_EXCEPTION( UnexpectedError, "Invalid instruction." );
	  }

	}

#if defined( ENABLE_STACKOPS_FOLDING )

    bypass_real:

      // Fetch next opcode, and if it is the target of of the stackops folding,
      // do it here.   If not (default case), start the next loop iteration.
      switch( FETCH_OPCODE() ) 
	{
	case ExpressionCompiler::ADD:
	  {
	    aStackPtr->theReal += bypass;

	    INCREMENT_PC( ADD );
	    break;
	  }

	case ExpressionCompiler::SUB:
	  {
	    aStackPtr->theReal -= bypass;

	    INCREMENT_PC( SUB );
	    break;
	  }

	case ExpressionCompiler::MUL:
	  {
	    aStackPtr->theReal *= bypass;

	    INCREMENT_PC( MUL );
	    break;
	  }

	case ExpressionCompiler::DIV:
	  {
	    aStackPtr->theReal /= bypass;

	    INCREMENT_PC( DIV );
	    break;
	  }

	case ExpressionCompiler::CALL_FUNC2:
	  {
	    DECODE_INSTRUCTION( CALL_FUNC2 );

	    aStackPtr->theReal
	      = ( anInstruction->getOperand() )( aStackPtr->theReal, bypass );

	    INCREMENT_PC( CALL_FUNC2 );
	    break;
	  }

	default:
	  {
	    // no need to do invalid instruction check here because
	    // it will be done in the next cycle.

            ++aStackPtr;
            aStackPtr->theReal = bypass;

	    break;
	  }
	}

      continue;

#else /* defined( ENABLE_STACKOPS_FOLDING ) */

    bypass_real:

      ++aStackPtr;
      aStackPtr->theReal = bypass;

      continue;

#endif /* defined( ENABLE_STACKOPS_FOLDING ) */

    }

}

#undef DECODE_INSTRUCTION
#undef FETCH_INSTRUCTION
#undef INCREMENT_PC


LIBECS_DM_INIT_STATIC( ExpressionProcessBase, Process );

#endif /* __EXPRESSIONPROCESSBASE_HPP */

