
#include <libecs/FullID.hpp>

#include "GillespieProcess.hpp"


LIBECS_DM_INIT( GillespieProcess, Process );

void GillespieProcess::initialize()
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

void GillespieProcess::calculateOrder()
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


