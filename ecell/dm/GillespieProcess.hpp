#ifndef __GILLESPIEPROCESS_HPP
#define __GILLESPIEPROCESS_HPP

#include <limits>
#include <vector>

#include <gsl/gsl_rng.h>

#include <libecs/libecs.hpp>
#include <libecs/DiscreteEventProcess.hpp>
#include <libecs/Stepper.hpp>
#include <libecs/FullID.hpp>

USE_LIBECS;

DECLARE_CLASS( GillespieProcess );
DECLARE_VECTOR( GillespieProcessPtr, GillespieProcessVector );


/***************************************************************************
     GillespieProcess 
***************************************************************************/

LIBECS_DM_CLASS( GillespieProcess, DiscreteEventProcess )
{
  
  typedef const Real (GillespieProcess::* RealMethodPtr)() const;
  
 public:
  
  LIBECS_DM_OBJECT( GillespieProcess, Process )
    {
      INHERIT_PROPERTIES( DiscreteEventProcess );
      
      PROPERTYSLOT_SET_GET( Real, k );

      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Mu );
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Int,  Order );
    }


  
  GillespieProcess() 
    :
    theOrder( 0 ),
    k( 0.0 ),
    theGetMultiplicityMethodPtr( &GillespieProcess::getZero ),
    theGetMinValueMethodPtr( &GillespieProcess::getZero )
    {
      ; // do nothing
    }

  virtual ~GillespieProcess()
    {
      ; // do nothing
    }
  

  SIMPLE_SET_GET_METHOD( Real, k );


  GET_METHOD( Real, Mu )
  {
    return k * ( this->*theGetMultiplicityMethodPtr )();
  }


  // The order of the reaction, i.e. 1 for a unimolecular reaction.

  GET_METHOD( Int, Order )
  {
    return theOrder;
  }

  virtual GET_METHOD( Real, TimeScale )
  {
    return ( this->*theGetMinValueMethodPtr )() * getStepInterval();
  }
  


  // a uniform random number (0...1) must be given as u
  virtual void updateStepInterval()
  {
    const Real aMu( getMu() );

    if( aMu > 0.0 )
      {
	const Real u( gsl_rng_uniform_pos( getStepper()->getRng() ) );
	theStepInterval = - log( u ) / aMu;

	if( getOrder() == 2 )
	  {
	    theStepInterval *= getSuperSystem()->getSizeN_A();
	  }
      }
    else // aMu == 0.0 (or aMu < 0.0 but this won't happen)
      {
	theStepInterval = std::numeric_limits<Real>::max();
      }
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

  virtual void process()
  {
    for( VariableReferenceVectorConstIterator 
	   i( theVariableReferenceVector.begin() );
	 i != theVariableReferenceVector.end() ; ++i )
      {
	VariableReferenceCref aVariableReference( *i );
	aVariableReference.addValue( aVariableReference.getCoefficient() );
      }
  }

  //  virtual const bool 
  //    checkEffect( GillespieProcessPtr anGillespieProcessPtr ) const;


protected:


  inline static const Real roundValue( RealCref aValue )
  {
    const Real aRoundedValue( trunc( aValue ) );

    if( aRoundedValue < 0.0 )
      {
	THROW_EXCEPTION( SimulationError, "Variable value <= -1.0" );
      }

    return aRoundedValue;
  }


  const Real getZero() const
  {
    return 0.0;
  }

  const Real getMultiplicity_FirstOrder() const
  {
    return roundValue( theVariableReferenceVector[0].getValue() );
  }

  const Real getMultiplicity_SecondOrder_TwoSubstrates() const
  {
    Real aMultiplicity( roundValue( theVariableReferenceVector[0].
				    getValue() ) );
    aMultiplicity *= roundValue( theVariableReferenceVector[1].getValue() );

    return aMultiplicity;
  }

  const Real getMultiplicity_SecondOrder_OneSubstrate() const
  {
    Real aMultiplicity( roundValue( theVariableReferenceVector[0].
				    getValue() ) );

    aMultiplicity *= ( aMultiplicity - 1.0 ) * 0.5;

    return aMultiplicity;
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

  GillespieProcessVector theEffectList;

  Int theOrder;

  RealMethodPtr theGetMultiplicityMethodPtr;
  RealMethodPtr theGetMinValueMethodPtr;

};


void GillespieProcess::calculateOrder()
{
  theOrder = 0;
    
  for( VariableReferenceVectorConstIterator 
	 i( theVariableReferenceVector.begin() );
       i != theVariableReferenceVector.end() ; ++i )
    {
      VariableReferenceCref aVariableReference( *i );
      const Int aCoefficient( aVariableReference.getCoefficient() );
	
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

  // set theGetMultiplicityMethodPtr and theGetMinValueMethodPtr

  if( getOrder() == 0 )   // no substrate
    {
      theGetMultiplicityMethodPtr = &GillespieProcess::getZero;
      theGetMinValueMethodPtr     = &GillespieProcess::getZero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      theGetMultiplicityMethodPtr = 
	&GillespieProcess::getMultiplicity_FirstOrder;
      theGetMinValueMethodPtr = &GillespieProcess::getMinValue_FirstOrder;
    }
  else if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
    {  
      theGetMultiplicityMethodPtr = 
	&GillespieProcess::getMultiplicity_SecondOrder_TwoSubstrates;
      theGetMinValueMethodPtr = 
	&GillespieProcess::getMinValue_SecondOrder_TwoSubstrates;
    }
  else // one substrate, second order (coeff == -2)
    {
      theGetMultiplicityMethodPtr = 
	&GillespieProcess::getMultiplicity_SecondOrder_OneSubstrate;
      theGetMinValueMethodPtr = 
	&GillespieProcess::getMinValue_SecondOrder_OneSubstrate;
    }

}




#endif /* __NRPROCESS_HPP */
