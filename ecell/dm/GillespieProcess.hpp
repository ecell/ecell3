//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
#ifndef __GILLESPIEPROCESS_HPP
#define __GILLESPIEPROCESS_HPP

#include <limits>

#include <gsl/gsl_rng.h>

#include <libecs/libecs.hpp>
#include <libecs/Process.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>
#include <libecs/MethodProxy.hpp>


USE_LIBECS;

/***************************************************************************
     GillespieProcess 
***************************************************************************/

LIBECS_DM_CLASS( GillespieProcess, Process )
{
  
  typedef MethodProxy<GillespieProcess, Real> RealMethodProxy;
  typedef const Real (GillespieProcess::* PDMethodPtr)( VariablePtr ) const; 
  
 public:
  
  LIBECS_DM_OBJECT( GillespieProcess, Process )
    {
      INHERIT_PROPERTIES( Process );
      
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, c );
      PROPERTYSLOT_SET_GET( Real, k );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Propensity );
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Order );
    }

  
  GillespieProcess() 
    :
    theOrder( 0 ),
    c( 0.0 ),
    theGetPropensityMethodPtr(
	RealMethodProxy::create<
	    &GillespieProcess::getZero>() ),
    theGetMinValueMethodPtr(
	RealMethodProxy::create<
	    &GillespieProcess::getZero>() ),
    theGetPDMethodPtr( &GillespieProcess::getPD_Zero )
  {
      ; // do nothing
  }

  virtual ~GillespieProcess()
  {
      ; // do nothing
  }
  

  // c means stochastic reaction constant
  SIMPLE_SET_GET_METHOD( Real, k );
  SIMPLE_SET_GET_METHOD( Real, c );
  
  GET_METHOD( Real, Propensity )
  {
    const Real aPropensity( theGetPropensityMethodPtr( this ) );

    if ( aPropensity < 0.0 )
      {
	THROW_EXCEPTION( SimulationError, "Variable value <= -1.0" );
	return 0.0;
      }
    else
      {
	return aPropensity;
      }
  }

  GET_METHOD( Real, Propensity_R )
  {
    const Real aPropensity( getPropensity() );

    if ( aPropensity > 0.0 )
      {
	return 1.0 / aPropensity;
      }
    else
      {
	return libecs::INF;
      }
  }

  const Real getPD( VariablePtr aVariable ) const
  {
    return ( this->*theGetPDMethodPtr )( aVariable );
  }

  virtual const bool isContinuous() const
  {
    return true;
  }

  // The order of the reaction, i.e. 1 for a unimolecular reaction.

  GET_METHOD( Integer, Order )
  {
    return theOrder;
  }

  /*
  virtual GET_METHOD( Real, TimeScale )
  {
    return theGetMinValueMethodPtr( this ) * getStepInterval();
  }
  */

  //  virtual void updateStepInterval()
  virtual GET_METHOD( Real, StepInterval )
  {
    return getPropensity_R() * 
      ( - log( gsl_rng_uniform_pos( getStepper()->getRng() ) ) );
  }

  void calculateOrder();

  virtual void initialize()
  {
    Process::initialize();
    declareUnidirectional();
  
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
    Real velocity( getk() * N_A );
    velocity *= getSuperSystem()->getSize();

    for( VariableReferenceVectorConstIterator
           s( theVariableReferenceVector.begin() );
         s != theZeroVariableReferenceIterator; ++s )
      {
        VariableReference aVariableReference( *s );
        Integer aCoefficient( aVariableReference.getCoefficient() );
        do {
          ++aCoefficient;
          velocity *= aVariableReference.getMolarConc();
        } while( aCoefficient != 0 );
         
      }
     
    setActivity( velocity );
  }


protected:

  const Real getZero() const
  {
    return 0.0;
  }

  const Real getPD_Zero( VariablePtr aVariable ) const
  {
    return 0.0;
  }

  /**
  const Real getInf() const
  {
    return libecs::INF;
  }
  */

  /**
     FirstOrder_OneSubstrate
   */

  const Real getPropensity_FirstOrder() const
  {
    const Real aValue(  theVariableReferenceVector[ 0 ].getValue() );

    if ( aValue > 0.0 )
      {
	return c * aValue;
      }
    else
      {
	return 0.0;
      }
  }

  const Real getMinValue_FirstOrder() const
  {
    return theVariableReferenceVector[ 0 ].getValue();
  }

  const Real getPD_FirstOrder( VariablePtr aVariable ) const
    {
      if ( theVariableReferenceVector[ 0 ].getVariable() == aVariable )
	{
	  return c;
	}
      else
	{
	  return 0.0;
	}
    }

  /**
     SecondOrder_TwoSubstrates
   */

  const Real getPropensity_SecondOrder_TwoSubstrates() const
  {
    const Real aValue1( theVariableReferenceVector[ 0 ].getValue() );
    const Real aValue2( theVariableReferenceVector[ 1 ].getValue() );

    if ( aValue1 > 0.0 && aValue2 > 0.0 )
      {
	return c * aValue1 * aValue2;
      }
    else
      {
	return 0.0;
      }
  }

  const Real getMinValue_SecondOrder_TwoSubstrates() const
  {
    const Real aFirstValue( theVariableReferenceVector[ 0 ].getValue() );
    const Real aSecondValue( theVariableReferenceVector[ 1 ].getValue() );

    return fmin( aFirstValue, aSecondValue );
  }

  const Real getPD_SecondOrder_TwoSubstrates( VariablePtr aVariable ) const
    {
      if ( theVariableReferenceVector[ 0 ].getVariable() == aVariable )
	{
	  return c * theVariableReferenceVector[ 1 ].getValue();
	}
      else if ( theVariableReferenceVector[ 1 ].getVariable() == aVariable )
	{
	  return c * theVariableReferenceVector[ 0 ].getValue();
	}
      else
	{
	  return 0.0;
	}
    }
  
  /**
     SecondOrder_OneSubstrate
   */

  const Real getPropensity_SecondOrder_OneSubstrate() const
  {
    const Real aValue( theVariableReferenceVector[ 0 ].getValue() );

    if ( aValue > 1.0 ) // there must be two or more molecules
      {
	return c * 0.5 * aValue * ( aValue - 1.0 );
      }
    else
      {
	return 0.0;
      }
  }

  const Real getMinValue_SecondOrder_OneSubstrate() const
  {
    return theVariableReferenceVector[ 0 ].getValue() * 0.5;
  }

  const Real getPD_SecondOrder_OneSubstrate( VariablePtr aVariable ) const
    {
      if( theVariableReferenceVector[ 0 ].getVariable() == aVariable )
	{
	  const Real aValue( theVariableReferenceVector[ 0 ].getValue() );

	  if ( aValue > 0.0 ) // there must be at least one molecule
	    {
	      return  c * 0.5 * ( 2.0 * aValue - 1.0 );
	    }
	  else
	    {
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
  Real c;

  Integer theOrder;

  RealMethodProxy theGetPropensityMethodPtr;  
  RealMethodProxy theGetMinValueMethodPtr;
  PDMethodPtr     theGetPDMethodPtr; // this should be MethodProxy

};


inline void GillespieProcess::calculateOrder()
{
  theOrder = 0;
    
  for( VariableReferenceVectorConstIterator 
	 i( theVariableReferenceVector.begin() );
       i != theVariableReferenceVector.end() ; ++i )
    {
      VariableReferenceCref aVariableReference( *i );
      const Integer aCoefficient( aVariableReference.getCoefficient() );
	
      // here assume aCoefficient != 0
      if( aCoefficient == 0 )
	{
	  THROW_EXCEPTION( InitializationFailed,
			   "[" + getFullID().getString() + 
			   "]: Zero stoichiometry is not allowed." );
	}

      if( aCoefficient < 0 )
	{
	  // sum the coefficient to get the order of this reaction.
	  theOrder -= aCoefficient; 
	}
    }

  // set theGetPropensityMethodPtr and theGetMinValueMethodPtr

  if( getOrder() == 0 )   // no substrate
    {
      theGetPropensityMethodPtr =
            RealMethodProxy::create<&GillespieProcess::getZero>();
      theGetMinValueMethodPtr   =
            RealMethodProxy::create<&GillespieProcess::getZero>();
      theGetPDMethodPtr         = &GillespieProcess::getPD_Zero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      theGetPropensityMethodPtr =
	    RealMethodProxy::create<&GillespieProcess::getPropensity_FirstOrder>();
      theGetMinValueMethodPtr   =
	    RealMethodProxy::create<&GillespieProcess::getMinValue_FirstOrder>();
      theGetPDMethodPtr         = &GillespieProcess::getPD_FirstOrder;
    }
  else if( getOrder() == 2 )
    {
      if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
	{  
	  theGetPropensityMethodPtr = RealMethodProxy::
	    create<&GillespieProcess::
	    getPropensity_SecondOrder_TwoSubstrates>();
	  theGetMinValueMethodPtr   = RealMethodProxy::
	    create<&GillespieProcess::getMinValue_SecondOrder_TwoSubstrates>();
	  theGetPDMethodPtr         
	    = &GillespieProcess::getPD_SecondOrder_TwoSubstrates;
	}
      else // one substrate, second order (coeff == -2)
	{
	  theGetPropensityMethodPtr = RealMethodProxy::
	    create<&GillespieProcess::
	    getPropensity_SecondOrder_OneSubstrate>();
	  theGetMinValueMethodPtr   = RealMethodProxy::
	    create<&GillespieProcess::getMinValue_SecondOrder_OneSubstrate>();
	  theGetPDMethodPtr 
	    = &GillespieProcess::getPD_SecondOrder_OneSubstrate;
	}
    }
  else
    {
      //FIXME: generic functions should come here.
      theGetPropensityMethodPtr =
            RealMethodProxy::create<&GillespieProcess::getZero>();
      theGetPropensityMethodPtr =
            RealMethodProxy::create<&GillespieProcess::getZero>();
      theGetPDMethodPtr         = &GillespieProcess::getPD_Zero;
    }



  //
  if ( theOrder == 1 ) 
    {
      c = k;
    }
  else if ( theOrder == 2 && getZeroVariableReferenceOffset() == 1 )
    {
      c = k * 2.0 / ( N_A * getSuperSystem()->getSize() );
    }
  else if ( theOrder == 2 && getZeroVariableReferenceOffset() == 2 )
    {
      c = k / ( N_A * getSuperSystem()->getSize() );
    }
  else
    {
      NEVER_GET_HERE;
    } 


}

#endif /* __GILLESPIEPROCESS_HPP */
