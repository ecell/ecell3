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
//   Kouichi Takahashi
//   Tatsuya Ishida
//
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __EXPRESSIONPROCESSBASE_HPP
#define __EXPRESSIONPROCESSBASE_HPP

#define EXPRESSION_PROCESS_USE_JIT 0

#define ENABLE_STACKOPS_FOLDING 1


#include <cassert>
#include <limits>

#include "ExpressionCompiler.hpp"

//#if defined( EXPRESSIONPROCESS_USE_JIT )
//#include "JITExpressionProcessBase"
//#else /* defined( EXPRESSIONPROCESS_USE_JIT ) */
//#include "SVMExpressionProcessBase"
//#endif /* defined( EXPRESSIONPROCESS_USE_JIT ) */


USE_LIBECS;

namespace libecs
{

  LIBECS_DM_CLASS( ExpressionProcessBase, Process )
  {

  protected:

    DECLARE_TYPE( ExpressionCompiler::Code, Code ); 

    typedef void* Pointer;

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
      {
	// ; do nothing
      }
    
      ~VirtualMachine() {}
    
      const Real execute( CodeCref aCode );

    };


  public:

    LIBECS_DM_OBJECT_ABSTRACT( ExpressionProcessBase )
      {
	INHERIT_PROPERTIES( Process );

	PROPERTYSLOT_SET_GET( String, Expression );
      }


    ExpressionProcessBase()
      :
      theNeedRecompile( true )
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
	theNeedRecompile = true;
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

	if( theNeedRecompile )
	  {
	    compileExpression();
	    theNeedRecompile = false;
	  }
      }

  protected:

    PropertyMapRef getPropertyMap()
      {
	return thePropertyMap;
      }


  protected:

    String    theExpression;
      
    Code theCompiledCode;
    VirtualMachine theVirtualMachine;

    bool theNeedRecompile;

    PropertyMap thePropertyMap;
  };


  
  const Real ExpressionProcessBase::VirtualMachine::execute( CodeCref aCode )
  {

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
                   Opcode2Instruction<ExpressionCompiler::OPCODE>::type ); //

    //    std::cout << #OPCODE << std::endl;

    StackElement aStack[100];
    aStack[0].theReal = 0.0;
    StackElementPtr aStackPtr( aStack );

    const unsigned char* aPC( &aCode[0] );

    while( 1 )
      {

	Real bypass;

	switch ( FETCH_OPCODE() )
	  {

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

	  case ExpressionCompiler::VARREF_REAL_METHOD:
	    {
	      DECODE_INSTRUCTION( VARREF_REAL_METHOD );
	      
	      bypass = ( anInstruction->getOperand() )();

	      INCREMENT_PC( VARREF_REAL_METHOD );
	      goto bypass_real;
	    }

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

	  case ExpressionCompiler::NEG:
	    {
	      aStackPtr->theReal = - aStackPtr->theReal;

	      INCREMENT_PC( NEG );
	      continue;
	    }

	  case ExpressionCompiler::ADD:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal += aStackTopValue;

	      INCREMENT_PC( ADD );
	      continue;
	    }

	  case ExpressionCompiler::SUB:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal -= aStackTopValue;

	      INCREMENT_PC( SUB );
	      continue;
	    }

	  case ExpressionCompiler::MUL:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal *= aStackTopValue;

	      INCREMENT_PC( MUL );
	      continue;
	    }

	  case ExpressionCompiler::DIV:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;	      

	      aStackPtr->theReal /= aStackTopValue;

	      INCREMENT_PC( DIV );
	      continue;
	    }

	  case ExpressionCompiler::POW:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal = pow( aStackPtr->theReal, aStackTopValue );

	      INCREMENT_PC( POW );
	      continue;
	    }

	  case ExpressionCompiler::EQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      if( aStackTopValue == aStackPtr->theReal )
		{
		  aStackPtr->theReal = 1.0;
		}
	      else
		{
		  aStackPtr->theReal = 0.0;
		}


	      INCREMENT_PC( EQ );
	      continue;
	    }

	  case ExpressionCompiler::NEQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal =
		Real( aStackPtr->theReal != aStackTopValue );

	      INCREMENT_PC( NEQ );
	      continue;
	    }

	  case ExpressionCompiler::GT:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = Real( aStackPtr->theReal > aStackTopValue );

	      INCREMENT_PC( GT );
	      continue;
	    }

	  case ExpressionCompiler::GEQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = 
		Real( aStackPtr->theReal >= aStackTopValue );

	      INCREMENT_PC( GEQ );
	      continue;
	    }

	  case ExpressionCompiler::LT:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = Real( aStackPtr->theReal < aStackTopValue );

	      INCREMENT_PC( LT );
	      continue;
	    }

	  case ExpressionCompiler::LEQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal =
		Real( aStackPtr->theReal <= aStackTopValue );

	      INCREMENT_PC( LEQ );
	      continue;
	    }

	  case ExpressionCompiler::AND:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal =
		Real( aStackTopValue && aStackPtr->theReal );

	      INCREMENT_PC( AND );
	      continue;
	    }

	  case ExpressionCompiler::OR:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = 
		Real( aStackTopValue || aStackPtr->theReal );

	      INCREMENT_PC( OR );
	      continue;
	    }

	  case ExpressionCompiler::XOR:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = 
		Real( aStackPtr->theReal && !( aStackTopValue ) );

	      INCREMENT_PC( XOR );
	      continue;
	    }

	  case ExpressionCompiler::NOT:
	    {
	      aStackPtr->theReal = !( aStackPtr->theReal );

	      INCREMENT_PC( NOT );
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

	  case ExpressionCompiler::CALL_FUNC2:
	    {
              DECODE_INSTRUCTION( CALL_FUNC2 );
	      
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal
		= ( anInstruction->getOperand() )( aStackPtr->theReal, 
						   aStackTopValue );

	      INCREMENT_PC( CALL_FUNC2 );
	      continue;
	    }

	  case ExpressionCompiler::PROCESS_TO_SYSTEM_METHOD:
	    {
	      DECODE_INSTRUCTION( PROCESS_TO_SYSTEM_METHOD );

	      ProcessPtr const aProcessPtr( static_cast<ProcessPtr>
					    ( aStackPtr->thePointer ) );

	      aStackPtr->thePointer =
		( aProcessPtr->*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( PROCESS_TO_SYSTEM_METHOD );
	      continue;
	    }

	  case ExpressionCompiler::VARREF_TO_SYSTEM_METHOD:
	    {
	      DECODE_INSTRUCTION( VARREF_TO_SYSTEM_METHOD );

	      VariableReferencePtr const 
		aVariableReferencePtr( static_cast<VariableReferencePtr>
				       ( aStackPtr->thePointer ) );

	      aStackPtr->thePointer =
		( aVariableReferencePtr->*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( VARREF_TO_SYSTEM_METHOD );
	      continue;
	    }

	  case ExpressionCompiler::SYSTEM_TO_REAL_METHOD:
	    {
	      DECODE_INSTRUCTION( SYSTEM_TO_REAL_METHOD );

	      SystemPtr const aSystemPtr( static_cast<SystemPtr>
					  ( aStackPtr->thePointer ) );

	      aStackPtr->theReal =
		( aSystemPtr->*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( SYSTEM_TO_REAL_METHOD );
	      continue;
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

      bypass_real:

#if defined( ENABLE_STACKOPS_FOLDING )

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

	  case ExpressionCompiler::POW:
	    {
	      aStackPtr->theReal = pow( aStackPtr->theReal, bypass );

	      INCREMENT_PC( POW );
	      break;
	    }

	  case ExpressionCompiler::NEG:
	    {
	      ++aStackPtr;
	      aStackPtr->theReal = - bypass;

	      INCREMENT_PC( NEG );
	      break;
	    }

	  case ExpressionCompiler::RET:
	    {
	      return bypass;
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
#else /* defined( ENABLE_STACKOPS_FOLDING ) */

	++aStackPtr;
	aStackPtr->theReal = bypass;

#endif /* defined( ENABLE_STACKOPS_FOLDING ) */

      }

#undef DECODE_INSTRUCTION
#undef FETCH_INSTRUCTION
#undef INCREMENT_PC

  }


  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, Process );

} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */

