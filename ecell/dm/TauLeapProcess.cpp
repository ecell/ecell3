#include "TauLeapProcess.hpp"
 
LIBECS_DM_INIT( TauLeapProcess, Process );

void TauLeapProcess::calculateOrder()
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
      theGetPropensityMethodPtr       = &TauLeapProcess::getZero;
      theGetPDMethodPtr       = &TauLeapProcess::getZero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      theGetPropensityMethodPtr = 
	&TauLeapProcess::getPropensity_FirstOrder;
      theGetPDMethodPtr = 
	&TauLeapProcess::getPD_FirstOrder;
      
    }
  else if( getOrder() == 2 )
    {
      if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
	{  
	  theGetPropensityMethodPtr = 
	    &TauLeapProcess::getPropensity_SecondOrder_TwoSubstrates;
	  theGetPDMethodPtr = 
	    &TauLeapProcess::getPD_SecondOrder_TwoSubstrates;
	}
      else // one substrate, second order (coeff == -2)
	{
	  theGetPropensityMethodPtr = 
	    &TauLeapProcess::getPropensity_SecondOrder_OneSubstrate;
	  theGetPDMethodPtr = 
	    &TauLeapProcess::getPD_SecondOrder_OneSubstrate;
	}
    }
  else
    {
      //FIXME: generic functions should come here.
      theGetPropensityMethodPtr       = &TauLeapProcess::getZero;
      theGetPDMethodPtr           = &TauLeapProcess::getZero;
    }
}
