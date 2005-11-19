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
// written by Tomoya Kitayama <tomo@e-cell.org>, 
// E-Cell Project.
//

#include <gsl/gsl_randist.h>

#include <libecs/libecs.hpp>
#include <libecs/ContinuousProcess.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>

USE_LIBECS;

LIBECS_DM_CLASS( TauLeapProcess, ContinuousProcess )
{

  typedef const Real (TauLeapProcess::* getPropensityMethodPtr)( ) const;
  typedef const Real (TauLeapProcess::* getPDMethodPtr)( VariablePtr ) const;

 public:

  LIBECS_DM_OBJECT( TauLeapProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );
      PROPERTYSLOT_SET_GET( Real, k );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Propensity );
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Order );
    }
  
  TauLeapProcess() 
    :
    theOrder( 0 ),
    k( 0.0 ),
    theGetPropensityMethodPtr( &TauLeapProcess::getZero ),
    theGetPDMethodPtr( &TauLeapProcess::getZero )
    {
      ; // do nothing
    }

  virtual ~TauLeapProcess()
    {
      ; // do nothing
    }

  SIMPLE_SET_GET_METHOD( Real, k );
  
  GET_METHOD( Integer, Order )
    {
      return theOrder;
    }
  
  GET_METHOD( Real, Propensity )
    {
      return ( this->*theGetPropensityMethodPtr )();
    }
  
  const Real getPD( VariablePtr value )const
    {
      return ( this->*theGetPDMethodPtr )( value );
    }

  virtual void initialize()
    {
      ContinuousProcess::initialize();

      calculateOrder();
      
      if( ! ( getOrder() == 1 || getOrder() == 2 ) )
	{
	  THROW_EXCEPTION( ValueError, 
			   String( getClassName() ) + 
			   "[" + getFullID().getString() + 
			   "]: Only first or second order scheme is allowed." );
	}
    }  

  virtual void fire()
    {
      setFlux( gsl_ran_poisson( getStepper()->getRng(), getPropensity() ) );
    }
  
 protected:

  void calculateOrder();
  
  static void checkNonNegative( const Real aValue )
    {
      if( aValue < 0.0 )
	{
	  THROW_EXCEPTION( SimulationError, "Variable value <= -1.0" );
	}
  }

  const Real getZero( VariablePtr value ) const
    {
      return 0.0;
    }

  const Real getZero( ) const
    {
      return 0.0;
    }
    
  const Real getPropensity_FirstOrder() const
    {
      const Real 
	aMultiplicity( theVariableReferenceVector[0].getValue() );
      
      if( aMultiplicity > 0.0 )
	{
	  return k * aMultiplicity;
	}
      else
	{
	  return 0.0;
	}
    }

  const Real getPD_FirstOrder( VariablePtr value ) const
    {
      if( theVariableReferenceVector[0].getVariable() == value )
	{
	  return k;
	}
      else
	{
	  return 0.0;
	}
    }

  const Real getPropensity_SecondOrder_TwoSubstrates() const
    {
      const Real 
	aMultiplicity( theVariableReferenceVector[0].getValue() *
		       theVariableReferenceVector[1].getValue() );
      
      if( aMultiplicity > 0.0 )
	{
	  return ( k * aMultiplicity ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
	}
      else
	{
	  return 0;
	}
    }

  const Real getPD_SecondOrder_TwoSubstrates( VariablePtr value ) const
    {
      if( theVariableReferenceVector[0].getVariable() == value )
	{
	  return ( k * theVariableReferenceVector[1].getValue() ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
	}
      else if( theVariableReferenceVector[1].getVariable() == value )
	{
	  return ( k * theVariableReferenceVector[0].getValue() ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
	}
      else
	{
	  return 0;
	}
    }
  
  const Real getPropensity_SecondOrder_OneSubstrate() const
    {
      const Real aValue( theVariableReferenceVector[0].getValue() );
      
      if( aValue > 1.0 ) // there must be two or more molecules
	{
	  return ( k * aValue * ( aValue - 1.0 ) ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A );
	    
	}
      else
	{
	  checkNonNegative( aValue );
	  return 0;
	}
      
    }

  const Real getPD_SecondOrder_OneSubstrate( VariablePtr value ) const
    {
      if( theVariableReferenceVector[0].getVariable() == value )
	{
	  const Real aValue( theVariableReferenceVector[0].getValue() );
	  if( aValue > 1.0 ) // there must be two or more molecules
	    {
	      return  ( ( 2 * k * aValue - k ) / ( getSuperSystem()->getSizeVariable()->getValue() * N_A ) );
	    }
	  else
	    {
	      checkNonNegative( aValue );
	      return 0.0;
	    }      
	}
      else
	{
	  return 0.0;
	}
    }
  
 protected:
  
  Real k;
  Integer theOrder;
  
  getPropensityMethodPtr theGetPropensityMethodPtr;
  getPDMethodPtr theGetPDMethodPtr;
  
};


