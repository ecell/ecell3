//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kazunari Kaizu <kaizu@sfc.keio.ac.jp>,
// E-Cell Project.
//
#ifndef __BOOLEANPROCESS_HPP
#define __BOOLEANPROCESS_HPP

#include "ExpressionProcessBase.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( BooleanProcess, ExpressionProcessBase )
{
 public:
  
  LIBECS_DM_OBJECT( BooleanProcess, Process )
  {
    INHERIT_PROPERTIES( ExpressionProcessBase );

    PROPERTYSLOT_SET_GET( Real, J );
  }

  BooleanProcess()
    :
    J( 0.1 )
  {
    // FIXME: additional properties:
    // Unidirectional -> call declareUnidirectional() in initialize()
    //                   if this is set
  }

  virtual ~BooleanProcess()
  {
    ; // do nothing
  }

  SIMPLE_SET_GET_METHOD( Real, J );

  virtual void initialize()
  {
    ExpressionProcessBase::initialize();

    const Real aSize( theVariableReferenceVector.size() );
    const Real 
      aZeroVariableReferenceOffset( getZeroVariableReferenceOffset() );
    const Real 
      aPositiveVariableReferenceOffset( getPositiveVariableReferenceOffset() );

    if ( aZeroVariableReferenceOffset 
	 + aSize - aPositiveVariableReferenceOffset != 1 )
      {
	THROW_EXCEPTION( InitializationFailed,
			 getClassNameString() +
			 ": Only One target is allowed "
			 "in each BooleanProcess." );
      }
    else if ( aZeroVariableReferenceOffset > 0 )
      {
	target = theVariableReferenceVector[ 0 ];
      }
    else if ( aPositiveVariableReferenceOffset < aSize )
      {
	target 
	  = theVariableReferenceVector[ aPositiveVariableReferenceOffset ];
      }
  }

  void evaluate()
  {
    //     addValue( theVirtualMachine.execute( theCompiledCode ) );    
    setFlux( theVirtualMachine.execute( theCompiledCode ) );    
  }
  
  virtual void fire()
  { 
    const Real Vmax( theVirtualMachine.execute( theCompiledCode ) );

    Real aValue;
    if ( target.getCoefficient() < 0 )
      {
	aValue = target.getVariable()->getValue();
      }
    else
      {
	aValue = 1.0 - target.getVariable()->getValue();
      }

    const Real anActivity( Vmax * aValue / ( J + aValue ) );
    setFlux( anActivity );
  }

  virtual const bool isContinuous() const
  {
    return true;
  }

 protected:

  VariableReference target;
  Real J;
};

#endif /* __BOOLEANPROCESS_HPP */
