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

	theNeedRecompile = false;
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

#define FETCH_INSTRUCTION()\
    anInstructionHead = \
      ( reinterpret_cast<const ExpressionCompiler::InstructionHead*>( aPC ) );

#define DECODE_INSTRUCTION( OPCODE )\
    typedef ExpressionCompiler::\
      Opcode2Instruction<ExpressionCompiler::OPCODE>::type CurrentInstruction;\
    const CurrentInstruction* const\
       anInstruction( reinterpret_cast< const CurrentInstruction* >\
      ( anInstructionHead ) );


#define INCREMENT_PC( OPCODE )\
    aPC += sizeof( ExpressionCompiler::\
                   Opcode2Instruction<ExpressionCompiler::OPCODE>::type );

    //    std::cout << #OPCODE << std::endl;

    StackElement    aStack[100];
    StackElementPtr aStackPtr( aStack );
    aStackPtr->theReal = 0.0;

    const char* aPC( &aCode[0] );

    const ExpressionCompiler::InstructionHead* anInstructionHead;
    FETCH_INSTRUCTION();

    while( 1 )
      {
	const ExpressionCompiler::Opcode 
	  anOpcode( anInstructionHead->getOpcode() );

	Real bypass;

	switch ( anOpcode )
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

	  case ExpressionCompiler::VARREF_METHOD:
	    {
	      DECODE_INSTRUCTION( VARREF_METHOD );
	      
	      const ExpressionCompiler::VariableReferenceMethod&
		aVariableReferenceMethod( anInstruction->getOperand() );

	      bypass = 
		( ( *( aVariableReferenceMethod.theOperand1 ) ).*( aVariableReferenceMethod.theOperand2 ) )();

	      INCREMENT_PC( VARREF_METHOD );
	      goto bypass_real;
	    }

	  case ExpressionCompiler::PUSH_INTEGER:
	    {
              DECODE_INSTRUCTION( PUSH_INTEGER );

	      ++aStackPtr;
	      aStackPtr->theInteger = anInstruction->getOperand();

	      INCREMENT_PC( PUSH_INTEGER );
	      goto next;
	    }

	  case ExpressionCompiler::PUSH_POINTER:
	    {
              DECODE_INSTRUCTION( PUSH_POINTER );

	      ++aStackPtr;
	      aStackPtr->thePointer = anInstruction->getOperand();
	      
	      INCREMENT_PC( PUSH_POINTER );
	      goto next;
	    }

	  case ExpressionCompiler::NEG:
	    {
	      aStackPtr->theReal = - aStackPtr->theReal;

	      INCREMENT_PC( NEG );
	      goto next;
	    }

	  case ExpressionCompiler::ADD:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal += aStackTopValue;

	      INCREMENT_PC( ADD );
	      goto next;
	    }

	  case ExpressionCompiler::SUB:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal -= aStackTopValue;

	      INCREMENT_PC( SUB );
	      goto next;
	    }


	  case ExpressionCompiler::MUL:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal *= aStackTopValue;

	      INCREMENT_PC( MUL );
	      goto next;
	    }

	  case ExpressionCompiler::DIV:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal /= aStackTopValue;

	      INCREMENT_PC( DIV );
	      goto next;
	    }

	  case ExpressionCompiler::POW:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal = pow( aStackPtr->theReal, aStackTopValue );

	      INCREMENT_PC( POW );
	      goto next;
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
	      goto next;
	    }

	  case ExpressionCompiler::NEQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal =
		Real( aStackPtr->theReal != aStackTopValue );

	      INCREMENT_PC( NEQ );
	      goto next;
	    }

	  case ExpressionCompiler::GT:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = Real( aStackPtr->theReal > aStackTopValue );

	      INCREMENT_PC( GT );
	      goto next;
	    }

	  case ExpressionCompiler::GEQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = 
		Real( aStackPtr->theReal >= aStackTopValue );

	      INCREMENT_PC( GEQ );
	      goto next;
	    }

	  case ExpressionCompiler::LT:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = Real( aStackPtr->theReal < aStackTopValue );

	      INCREMENT_PC( LT );
	      goto next;
	    }

	  case ExpressionCompiler::LEQ:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal =
		Real( aStackPtr->theReal <= aStackTopValue );

	      INCREMENT_PC( LEQ );
	      goto next;
	    }

	  case ExpressionCompiler::AND:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal =
		Real( aStackTopValue && aStackPtr->theReal );

	      INCREMENT_PC( AND );
	      goto next;
	    }

	  case ExpressionCompiler::OR:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = 
		Real( aStackTopValue || aStackPtr->theReal );

	      INCREMENT_PC( OR );
	      goto next;
	    }

	  case ExpressionCompiler::XOR:
	    {
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;
	      
	      aStackPtr->theReal = 
		Real( aStackPtr->theReal && !( aStackTopValue ) );

	      INCREMENT_PC( XOR );
	      goto next;
	    }

	  case ExpressionCompiler::NOT:
	    {
	      aStackPtr->theReal = !( aStackPtr->theReal );

	      INCREMENT_PC( NOT );
	      goto next;
	    }

	  case ExpressionCompiler::CALL_FUNC1:
	    {
              DECODE_INSTRUCTION( CALL_FUNC1 );

	      aStackPtr->theReal
		= ( anInstruction->getOperand() )( aStackPtr->theReal );
	      
	      INCREMENT_PC( CALL_FUNC1 );
	      goto next;
	    }

	  case ExpressionCompiler::CALL_FUNC2:
	    {
              DECODE_INSTRUCTION( CALL_FUNC2 );
	      
	      Real aStackTopValue( aStackPtr->theReal );
	      --aStackPtr;

	      aStackPtr->theReal
		= ( anInstruction->getOperand() )( aStackPtr->theReal, aStackTopValue );


	      INCREMENT_PC( CALL_FUNC2 );
	      goto next;
	    }
	  

	  case ExpressionCompiler::PROCESS_TO_SYSTEM_METHOD:
	    {
	      DECODE_INSTRUCTION( PROCESS_TO_SYSTEM_METHOD );

	      reinterpret_cast<System*>( aStackPtr->thePointer ) =
		( *( reinterpret_cast<Process*>( aStackPtr->thePointer ) ).*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( PROCESS_TO_SYSTEM_METHOD );
	      goto next;
	    }

	  case ExpressionCompiler::VARREF_TO_SYSTEM_METHOD:
	    {
	      DECODE_INSTRUCTION( VARREF_TO_SYSTEM_METHOD );

	      reinterpret_cast<System*>( aStackPtr->thePointer ) =
		( *( reinterpret_cast<VariableReference*>
		     ( aStackPtr->thePointer ) ).*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( VARREF_TO_SYSTEM_METHOD );
	      goto next;
	    }

	  case ExpressionCompiler::SYSTEM_TO_REAL_METHOD:
	    {
	      DECODE_INSTRUCTION( SYSTEM_TO_REAL_METHOD );

	      aStackPtr->theReal =
		( *( reinterpret_cast<System*>( aStackPtr->thePointer ) ).*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( SYSTEM_TO_REAL_METHOD );
	      goto next;
	    }

	  case ExpressionCompiler::RET:
	    {
              //DECODE_INSTRUCTION( RET );
	      //INCREMENT_PC( RET );

	      return aStackPtr->theReal;
	    }

	  default:
	    THROW_EXCEPTION( UnexpectedError, "Invalid instruction." );
	  }

      bypass_real:

#if defined( ENABLE_STACKOPS_FOLDING )

	FETCH_INSTRUCTION();

	switch( anInstructionHead->getOpcode() )
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
	      if( anInstructionHead->getOpcode() <= ExpressionCompiler::END )
		{
		  ++aStackPtr;
		  aStackPtr->theReal = bypass;
		  continue;
		}
	      else
		{
		  THROW_EXCEPTION( UnexpectedError, "Invalid instruction." );
		}
	    }

	  }
#else /* defined( ENABLE_STACKOPS_FOLDING ) */

	++aStackPtr;
	aStackPtr->theReal = bypass;

#endif /* defined( ENABLE_STACKOPS_FOLDING ) */

      next:
	
	FETCH_INSTRUCTION();

      }

#undef DECODE_INSTRUCTION
#undef FETCH_INSTRUCTION
#undef INCREMENT_PC

  }


  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, Process );

} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */

