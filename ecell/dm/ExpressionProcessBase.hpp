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
//   Kouichi Takahashi
//   Tatsuya Ishida
//
// E-CELL Project, Lab. for Bioinformatics, Keio University.
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
    
      // Stack -> std::vector<variant<>...>
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
      }

    GET_METHOD( String, Expression )
      {
	return theExpression;
      }

    virtual void defaultSetProperty( StringCref aPropertyName,
				     PolymorphCref aValue)
      {
	thePropertyMap[ aPropertyName ] = aValue;
      } 

    /**virtual const Polymorph getExtraPropertyList()
      {
	return thePropertyMap;
	}*/

    virtual void initialize()
      {
	ExpressionCompiler theCompiler;

	Process::initialize();

	theCompiler.setProcessPtr( static_cast<Process*>( this ) );
	theCompiler.setPropertyMap( thePropertyMap );

	theCompiledCode.clear();
	theCompiledCode = theCompiler.compileExpression( theExpression );

	theStackMachine.resize( theCompiler.getStackSize() );
      }

  protected:

    String    theExpression;
      
    Code theCompiledCode;
    StackMachine theStackMachine;

    PropertyMap thePropertyMap;
  };


  
  const Real ExpressionProcessBase::StackMachine::
  execute( CodeCref aCode )
  {
    //using libecs::ExpressionCompiler;

    const char* aPC( &aCode[0] );

    reset();


    //#define DECLARE_CURRENTINSTRUCTION( OPCODE )

#define FETCH_INSTRUCTION( PC )\
    const ExpressionCompiler::InstructionHead* anInstructionHead\
      ( reinterpret_cast<const ExpressionCompiler::InstructionHead*>( PC ) );

//    const INSTRUCTION* anInstruction( (INSTRUCTION*) PC );

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

	std::cout << "--------------------------------" << std::endl;
	std::cout << "OPCode " << anOpcode << std::endl;

	switch ( anOpcode )
	  {
	  case ExpressionCompiler::PUSH:
	    {
              DECODE_INSTRUCTION( PUSH );

	      ++theStackPtr;
	      theStackPtr->theReal = anInstruction->getOperand();
	      
	      std::cerr << "PUSH " << (Real)theStackPtr->theReal << std::endl;
	      
	      INCREMENT_PC( aPC );

	      std::cerr << "PUSH CurrentInstruction Size : " << sizeof(CurrentInstruction) << std::endl;

	      break;
	    }
	    
	  case ExpressionCompiler::NEG:
	    {
              DECODE_INSTRUCTION( NEG );

	      theStackPtr->theReal = - theStackPtr->theReal;

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::ADD:
	    {
	      std::cout << "ADD" << std::endl;

              DECODE_INSTRUCTION( ADD );
	      
	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = *aStackTop1 + *aStackTop2;

	      INCREMENT_PC( aPC );

	      std::cout << "ADD Current InstructionSize : " << sizeof( CurrentInstruction ) << std::endl;
	      break;
	    }

	  case ExpressionCompiler::SUB:
	    {
	      std::cout << "SUB" << std::endl;

              DECODE_INSTRUCTION( SUB );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = *aStackTop1 - *aStackTop2;

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::MUL:
	    {
	      std::cout << "MUL" << std::endl;

              DECODE_INSTRUCTION( MUL );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );

	      std::cout << "StackTop1 : " << *aStackTop1<< std::endl;
	      std::cout << "StackTop2 : " << *aStackTop2<< std::endl;
	      
	      *aStackTop2 = *aStackTop1 * *aStackTop2;

	      std::cout << "MUL Result : " << *aStackTop2 << std::endl;


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::DIV:
	    {
	      std::cout << "DIV" << std::endl;

              DECODE_INSTRUCTION( DIV );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = *aStackTop1 / *aStackTop2;


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::POW:
	    {
              DECODE_INSTRUCTION( POW );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = pow( *aStackTop1, *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::EQ:
	    {
              DECODE_INSTRUCTION( EQ );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 == *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::NEQ:
	    {
              DECODE_INSTRUCTION( NEQ );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 != *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::GT:
	    {
              DECODE_INSTRUCTION( GT );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 > *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::GEQ:
	    {
              DECODE_INSTRUCTION( GEQ );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 >= *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::LT:
	    {
              DECODE_INSTRUCTION( LT );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 < *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::LEQ:
	    {
              DECODE_INSTRUCTION( LEQ );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 <= *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::AND:
	    {
              DECODE_INSTRUCTION( AND );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 && *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::OR:
	    {
              DECODE_INSTRUCTION( OR );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 || *aStackTop2 );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::XOR:
	    {
              DECODE_INSTRUCTION( XOR );

	      Real* aStackTop1( &theStackPtr->theReal );
	      --theStackPtr;
	      Real* aStackTop2( &theStackPtr->theReal );
	      
	      *aStackTop2 = Real( *aStackTop1 && !( *aStackTop2 ) );


	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::NOT:
	    {
              DECODE_INSTRUCTION( NOT );

	      theStackPtr->theReal = !( theStackPtr->theReal );

	      INCREMENT_PC( aPC );
	      break;
	    }

	  case ExpressionCompiler::CALL_FUNC1:
	    {
	      std::cout << "CALL_FUNC1" << std::endl;

              DECODE_INSTRUCTION( CALL_FUNC1 );

	      theStackPtr->theReal
		= ( anInstruction->getOperand() )( theStackPtr->theReal );
	      
	      INCREMENT_PC( aPC );
	      break;
	    }



	    /**case ExpressionCompiler::CALL_FUNC2:
	  case ExpressionCompiler::GET_PROPERTY:
	  case ExpressionCompiler::VARREF_METHOD:
	  case ExpressionCompiler::PROCESS_METHOD:
	  case ExpressionCompiler::PROCESS_METHOD:
	  case ExpressionCompiler::SYSTEM_METHOD:*/



	  case ExpressionCompiler::HALT:
	    {
              DECODE_INSTRUCTION( HALT );

	      std::cout << "HALTSize : " << sizeof( CurrentInstruction ) << std::endl;
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


  
  
  
/**
  void
  ExpressionProcessBase::CALL_FUNC1::
  execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr() )
      = ( *theFuncPtr )( *( aStackMachine.getStackPtr() ) );
  }

  void
  ExpressionProcessBase::CALL_FUNC2::
  execute( StackMachine& aStackMachine )
  {
    *( aStackMachine.getStackPtr()-1 )
      = ( *theFuncPtr )( *( aStackMachine.getStackPtr()-1 ),
			 *( aStackMachine.getStackPtr() ) );

    aStackMachine.getStackPtr()--;
  }

  void
  ExpressionProcessBase::GET_PROPERTY::
  execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;

    *aStackMachine.getStackPtr() = theValue;
  }
  
  void 
  ExpressionProcessBase::VARREF_FUNC::
  execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;
    *aStackMachine.getStackPtr() = ( theVariableReference.*theFuncPtr )();
  }

  void
  ExpressionProcessBase::PROCESS_SYSTEM_FUNC::
  execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;
    
    *aStackMachine.getStackPtr()
      = ( ( theProcessPtr->*theFuncPtr )()->*theAttributePtr)();
  }

  void
  ExpressionProcessBase::VARIABLE_SYSTEM_FUNC::
  execute( StackMachine& aStackMachine )
  {
    aStackMachine.getStackPtr()++;
    
    *aStackMachine.getStackPtr()
      = ( ( theVariableReference.*theFuncPtr )()->*theAttributePtr)();
      }*/

  LIBECS_DM_INIT_STATIC( ExpressionProcessBase, Process );

} // namespace libecs


#endif /* __EXPRESSIONPROCESSBASE_HPP */
