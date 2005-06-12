#ifndef __GILLESPIEPROCESS_HPP
#define __GILLESPIEPROCESS_HPP

#include <limits>

#include <gsl/gsl_rng.h>

#include <libecs/libecs.hpp>
#include <libecs/DiscreteEventProcess.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>
#include <libecs/MethodProxy.hpp>


USE_LIBECS;

/***************************************************************************
     GillespieProcess 
***************************************************************************/

LIBECS_DM_CLASS( GillespieProcess, DiscreteEventProcess )
{
  
  typedef MethodProxy<GillespieProcess,Real> RealMethodProxy;
  typedef const Real (GillespieProcess::* PDMethodPtr)( VariablePtr ) const; 
  
 public:
  
  LIBECS_DM_OBJECT( GillespieProcess, Process )
    {
      INHERIT_PROPERTIES( DiscreteEventProcess );
      
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, c );
      PROPERTYSLOT_SET_GET( Real, k );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Propensity );
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Order );
    }

  
  GillespieProcess() 
    :
    theOrder( 0 ),
    c( 0.0 ),
    theGetPropensityMethodPtr( RealMethodProxy::
			       create<&GillespieProcess::getZero>() ),
    theGetMinValueMethodPtr( RealMethodProxy::
			     create<&GillespieProcess::getZero>() ),
    theGetPDMethodPtr( &GillespieProcess::getZero )
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

  virtual GET_METHOD( Real, TimeScale )
  {
    return theGetMinValueMethodPtr( this ) * getStepInterval();
  }


  virtual void updateStepInterval()
  {
    theStepInterval = getPropensity_R() * 
      ( - log( gsl_rng_uniform_pos( getStepper()->getRng() ) ) );
  }

  void calculateOrder();

  virtual void initialize()
  {
    DiscreteEventProcess::initialize();
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

  const Real getZero( VariablePtr aVariable ) const
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
    return c * theVariableReferenceVector[ 0 ].getValue();
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
    return c * theVariableReferenceVector[ 0 ].getValue() * theVariableReferenceVector[ 1 ].getValue();
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


#endif /* __NRPROCESS_HPP */
