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

    class StackMachine;

    DECLARE_TYPE( ExpressionCompiler::Code, Code ); 

    typedef void* Pointer;


    class StackMachine
    {
      //      typedef boost::variant< Real, 
      //			      Pointer,
      //			      Integer > StackElement_;

      union StackElement_
      {
	Real theReal;
	Pointer thePointer;
	Integer theInteger;
      };

      //      StringSharedPtr > Operand;
      DECLARE_TYPE( StackElement_, StackElement );
      DECLARE_VECTOR( StackElement, Stack );

    public:
    
      StackMachine()
      {
	// ; do nothing
      }
    
      ~StackMachine() {}
    
      void resize( Stack::size_type aSize )
      {
	theStack.resize( aSize );
      }
    
      void reset()
      {
	theStackPtr = &theStack[0];
        theStack[0].theReal = 0.0;
      }

      const Real execute( CodeCref aCode );
    
    protected:
    
      Stack           theStack;
    
      StackElementPtr theStackPtr;
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
				     PolymorphCref aValue)
      {
	thePropertyMap[ aPropertyName ] = aValue.asReal();
      } 


    void compileExpression()
      {
	ExpressionCompiler theCompiler;
	
	theCompiler.setProcessPtr( static_cast<Process*>( this ) );
	theCompiler.setPropertyMap( &thePropertyMap );

	theCompiledCode.clear();
	theCompiledCode = theCompiler.compileExpression( theExpression );

	theStackMachine.resize( theCompiler.getStackSize() );

	theNeedRecompile = false;
      }

    /**virtual const Polymorph getExtraPropertyList()
      {
	return thePropertyMap;
	}*/

    virtual void initialize()
      {
	Process::initialize();

	if( theNeedRecompile )
	  {
	    compileExpression();
	  }
      }

  protected:

    String    theExpression;
      
    Code theCompiledCode;
    StackMachine theStackMachine;

    bool theNeedRecompile;

    PropertyMap thePropertyMap;
  };


  
  const Real ExpressionProcessBase::StackMachine::
  execute( CodeCref aCode )
  {

    const char* aPC( &aCode[0] );

    reset();


#define FETCH_INSTRUCTION( PC )\
    const ExpressionCompiler::InstructionHead* anInstructionHead\
      ( reinterpret_cast<const ExpressionCompiler::InstructionHead*>( PC ) );

#define DECODE_INSTRUCTION( OPCODE )\
    typedef ExpressionCompiler::\
      Opcode2Instruction<ExpressionCompiler::OPCODE>::type CurrentInstruction;\
    const size_t SizeOfCurrentInstruction( sizeof( CurrentInstruction ) );\
    const CurrentInstruction*\
       anInstruction( reinterpret_cast< const CurrentInstruction* >\
      ( anInstructionHead ) );

#define INCREMENT_PC( PC )\
    PC += SizeOfCurrentInstruction;


    while( 1 )
      {
	// decode opcode
	//	const ExpressionCompiler::Opcode* 
	//anOpcode( (ExpressionCompiler::Opcode*)aPC );
	FETCH_INSTRUCTION( aPC );

	const ExpressionCompiler::Opcode 
	  anOpcode( anInstructionHead->getOpcode() );

	switch ( anOpcode )
	  {
	  case ExpressionCompiler::PUSH_REAL:
	    {
	      //std::cout << "PUSH_REAL" << std::endl;

              DECODE_INSTRUCTION( PUSH_REAL );

	      ++theStackPtr;
	      theStackPtr->theReal = anInstruction->getOperand();
	      
	      INCREMENT_PC( aPC );
	      break;
	    }
	    
	  case ExpressionCompiler::PUSH_INTEGER:
	    {
	      //std::cout << "PUSH_INTEGER" << std::endl;

              DECODE_INSTRUCTION( PUSH_INTEGER );

	      ++theStackPtr;
	      theStackPtr->theInteger = anInstruction->getOperand();
	      
	      
	      INCREMENT_PC( aPC );
	      break;
	    }
	    
	  case ExpressionCompiler::PUSH_POINTER:
	    {
	      //std::cout << "PUSH_POINTER" << std::endl;

              DECODE_INSTRUCTION( PUSH_POINTER );

	      ++theStackPtr;
	      theStackPtr->thePointer = anInstruction->getOperand();
	      
	      INCREMENT_PC( aPC );
	      break;
	    }
	    
	  case ExpressionCompiler::NEG:
	    {
	      //std::cout << "NEG" << std::endl;

              DECODE_INSTRUCTION( NEG );

	      theStackPtr->theReal = - theStackPtr->theReal;

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::ADD:
	    {
	      //std::cout << "ADD" << std::endl;

              DECODE_INSTRUCTION( ADD );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;

	      theStackPtr->theReal += *aStackTop;

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::SUB:
	    {
	      //std::cout << "SUB" << std::endl;

              DECODE_INSTRUCTION( SUB );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal -= *aStackTop;

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::MUL:
	    {
	      //std::cout << "MUL" << std::endl;

              DECODE_INSTRUCTION( MUL );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;

	      theStackPtr->theReal *= *aStackTop;


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::DIV:
	    {
	      //std::cout << "DIV" << std::endl;

              DECODE_INSTRUCTION( DIV );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal /= *aStackTop;


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::POW:
	    {
	      //std::cout << "POW" << std::endl;

              DECODE_INSTRUCTION( POW );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;

	      theStackPtr->theReal = pow( theStackPtr->theReal, *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::EQ:
	    {
	      //std::cout << "EQ" << std::endl;

              DECODE_INSTRUCTION( EQ );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal =
		Real( *aStackTop == theStackPtr->theReal );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::NEQ:
	    {
	      //std::cout << "NEQ" << std::endl;

              DECODE_INSTRUCTION( NEQ );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal =
		Real( theStackPtr->theReal != *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::GT:
	    {
	      //std::cout << "GT" << std::endl;

              DECODE_INSTRUCTION( GT );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal = Real( theStackPtr->theReal > *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::GEQ:
	    {
	      //std::cout << "GEQ" << std::endl;

              DECODE_INSTRUCTION( GEQ );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal =
		Real( theStackPtr->theReal >= *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::LT:
	    {
	      //std::cout << "LT" << std::endl;

              DECODE_INSTRUCTION( LT );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal = Real( theStackPtr->theReal < *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::LEQ:
	    {
	      //std::cout << "LEQ" << std::endl;

              DECODE_INSTRUCTION( LEQ );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal =
		Real( theStackPtr->theReal <= *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::AND:
	    {
	      //std::cout << "AND" << std::endl;

              DECODE_INSTRUCTION( AND );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal =
		Real( *aStackTop && theStackPtr->theReal );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::OR:
	    {
	      //std::cout << "OR" << std::endl;

              DECODE_INSTRUCTION( OR );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal = 
		Real( *aStackTop || theStackPtr->theReal );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::XOR:
	    {
	      //std::cout << "XOR" << std::endl;

              DECODE_INSTRUCTION( XOR );

	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;
	      
	      theStackPtr->theReal = 
		Real( theStackPtr->theReal && !( *aStackTop ) );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::NOT:
	    {
	      //std::cout << "NOT" << std::endl;

              DECODE_INSTRUCTION( NOT );

	      theStackPtr->theReal = !( theStackPtr->theReal );

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::CALL_FUNC1:
	    {
	      //std::cout << "CALL_FUNC1" << std::endl;

              DECODE_INSTRUCTION( CALL_FUNC1 );

	      theStackPtr->theReal
		= ( anInstruction->getOperand() )( theStackPtr->theReal );
	      
	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::CALL_FUNC2:
	    {
	      //std::cout << "CALL_FUNC2" << std::endl;

              DECODE_INSTRUCTION( CALL_FUNC2 );
	      
	      Real* aStackTop( &theStackPtr->theReal );
	      --theStackPtr;

	      theStackPtr->theReal
		= ( anInstruction->getOperand() )( theStackPtr->theReal, *aStackTop );


	      INCREMENT_PC( aPC );
	      break;
	    }
	  
	  case ExpressionCompiler::LOAD_REAL:
	    {
	      //std::cout << "LOAD_REAL" << std::endl;

	      DECODE_INSTRUCTION( LOAD_REAL );
	      
	      ++theStackPtr;
	      theStackPtr->theReal = *( anInstruction->getOperand() );
	      
	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::VARREF_METHOD:
	    {
	      //std::cout << "VARREF_METHOD" << std::endl;

	      DECODE_INSTRUCTION( VARREF_METHOD );
	      
	      ExpressionCompiler::VariableReferenceMethod
		aVariableReferenceMethod( anInstruction->getOperand() );

	      ++theStackPtr;
	      theStackPtr->theReal =
		( ( *( aVariableReferenceMethod.theOperand1 ) ).*( aVariableReferenceMethod.theOperand2 ) )();

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::PROCESS_TO_SYSTEM_METHOD:
	    {
	      //std::cout << "PROCESS_TO_SYSTEM_METHOD" << std::endl;

	      DECODE_INSTRUCTION( PROCESS_TO_SYSTEM_METHOD );

	      reinterpret_cast<System*>( theStackPtr->thePointer ) =
		( *( reinterpret_cast<Process*>( theStackPtr->thePointer ) ).*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::VARREF_TO_SYSTEM_METHOD:
	    {
	      //std::cout << "VARREF_TO_SYSTEM_METHOD" << std::endl;

	      DECODE_INSTRUCTION( VARREF_TO_SYSTEM_METHOD );

	      reinterpret_cast<System*>( theStackPtr->thePointer ) =
		( *( reinterpret_cast<VariableReference*>( theStackPtr->thePointer ) ).*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::SYSTEM_TO_REAL_METHOD:
	    {
	      //std::cout << "SYSTEM_TO_REAL_METHOD" << std::endl;

	      DECODE_INSTRUCTION( SYSTEM_TO_REAL_METHOD );

	      theStackPtr->theReal =
		( *( reinterpret_cast<System*>( theStackPtr->thePointer ) ).*( anInstruction->getOperand() ) )();

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::HALT:
	    {
	      //std::cout << "HALT" << std::endl;

              DECODE_INSTRUCTION( HALT );

	      INCREMENT_PC( aPC );
	      return theStackPtr->theReal;
	    }

	  default:
	    THROW_EXCEPTION( UnexpectedError, "Invalid instruction." );
	  }
      }

#undef DECODE_INSTRUCTION
#undef FETCH_INSTRUCTION
#undef INCREMENT_PC

  }


  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, Process );

} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */
