#ifndef __GILLESPIEPROCESS_HPP
#define __GILLESPIEPROCESS_HPP

#include <limits>

#include <gsl/gsl_rng.h>

#include <libecs/libecs.hpp>
#include <libecs/DiscreteEventProcess.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/MethodProxy.hpp>


USE_LIBECS;

DECLARE_CLASS( GillespieProcess );


/***************************************************************************
     GillespieProcess 
***************************************************************************/

LIBECS_DM_CLASS( GillespieProcess, DiscreteEventProcess )
{
  
  typedef MethodProxy<GillespieProcess,Real> RealMethodProxy;
  
 public:
  
  LIBECS_DM_OBJECT( GillespieProcess, Process )
    {
      INHERIT_PROPERTIES( DiscreteEventProcess );
      
      PROPERTYSLOT_SET_GET( Real, k );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Propensity );
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Integer,  Order );
    }

  
  GillespieProcess() 
    :
    theOrder( 0 ),
    k( 0.0 ),
    theGetPropensity_RMethodPtr( RealMethodProxy::
				 create<&GillespieProcess::getInf>() ),
    theGetMinValueMethodPtr( RealMethodProxy::
			     create<&GillespieProcess::getZero>() )
    {
      ; // do nothing
    }

  virtual ~GillespieProcess()
    {
      ; // do nothing
    }
  

  SIMPLE_SET_GET_METHOD( Real, k );

  GET_METHOD( Real, Propensity )
  {
    return 1.0 / getPropensity_R();
  }


  GET_METHOD( Real, Propensity_R )
  {
    return theGetPropensity_RMethodPtr( this );
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


  virtual void initialize();

  virtual void fire()
  {
    for( VariableReferenceVectorConstIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end() ; ++i )
      {
	VariableReferenceCref aVariableReference( *i );
	aVariableReference.addValue( aVariableReference.getCoefficient() );
      }
  }


protected:

  static void checkNonNegative( const Real aValue )
  {
    if( aValue < 0.0 )
      {
	THROW_EXCEPTION( SimulationError, "Variable value <= -1.0" );
      }
  }

  const Real getZero() const
  {
    return 0.0;
  }

  const Real getInf() const
  {
    return libecs::INF;
  }

  const Real getPropensity_R_FirstOrder() const
  {
    const Real aMultiplicity( theVariableReferenceVector[0].getValue() );

    if( aMultiplicity > 0.0 )
      {
	return 1.0 / ( k * aMultiplicity );
      }
    else
      {
	checkNonNegative( aMultiplicity );

	return libecs::INF;
      }
  }

  const Real getPropensity_R_SecondOrder_TwoSubstrates() const
  {
    const Real aMultiplicity( theVariableReferenceVector[0].getValue() *
			      theVariableReferenceVector[1].getValue() );

    if( aMultiplicity > 0.0 )
      {
	return ( getSuperSystem()->getSizeVariable()->getValue() * N_A ) /
	  ( k * aMultiplicity );
      }
    else
      {
	checkNonNegative( aMultiplicity );

	return libecs::INF;
      }
  }

  const Real getPropensity_R_SecondOrder_OneSubstrate() const
  {
    const Real aValue( theVariableReferenceVector[0].getValue() );

    if( aValue >= 2.0 ) // there must be two or more molecules
      {
	return ( getSuperSystem()->getSizeVariable()->getValue() * N_A ) /
	  ( k * aValue * ( aValue - 1.0 ) );
      }
    else
      {
	checkNonNegative( aValue );

	return libecs::INF;
      }

  }

  const Real getMinValue_FirstOrder() const
  {
    return theVariableReferenceVector[0].getValue();
  }

  const Real getMinValue_SecondOrder_TwoSubstrates() const
  {
    const Real aFirstValue( theVariableReferenceVector[0].getValue() );
    const Real aSecondValue( theVariableReferenceVector[1].getValue() );

    return fmin( aFirstValue, aSecondValue );
  }

  const Real getMinValue_SecondOrder_OneSubstrate() const
  {
    return theVariableReferenceVector[0].getValue() * 0.5;
  }


protected:

  Real k;    

  Integer theOrder;

  RealMethodProxy theGetPropensity_RMethodPtr;
  
  RealMethodProxy theGetMinValueMethodPtr;

};




#endif /* __NRPROCESS_HPP */
