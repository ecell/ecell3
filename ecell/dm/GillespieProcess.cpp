
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

  // set theGetPropensity_RMethodPtr and theGetMinValueMethodPtr

  if( getOrder() == 0 )   // no substrate
    {
      //      theGetPropensity_RMethodPtr       = &GillespieProcess::getInf;
      theGetPropensity_RMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getInf>();
      theGetMinValueMethodPtr     = &GillespieProcess::getZero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      //      theGetPropensity_RMethodPtr = 
      //	&GillespieProcess::getPropensity_R_FirstOrder;
      theGetPropensity_RMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getPropensity_R_FirstOrder>();
      theGetMinValueMethodPtr = &GillespieProcess::getMinValue_FirstOrder;
    }
  else if( getOrder() == 2 )
    {
      if( getZeroVariableReferenceOffset() == 2 ) // 2 substrates, 2nd order
	{  
	  //	  theGetPropensity_RMethodPtr = 
	  //	    &GillespieProcess::getPropensity_R_SecondOrder_TwoSubstrates;
	  theGetPropensity_RMethodPtr = RealMethodProxy::
	    create<&GillespieProcess::
	    getPropensity_R_SecondOrder_TwoSubstrates>();
	  theGetMinValueMethodPtr = 
	    &GillespieProcess::getMinValue_SecondOrder_TwoSubstrates;
	}
      else // one substrate, second order (coeff == -2)
	{
	  //	  theGetPropensity_RMethodPtr = 
	  //	    &GillespieProcess::getPropensity_R_SecondOrder_OneSubstrate;
	  theGetPropensity_RMethodPtr = RealMethodProxy::
	    create<&GillespieProcess::
	    getPropensity_R_SecondOrder_OneSubstrate>();
	  theGetMinValueMethodPtr = 
	    &GillespieProcess::getMinValue_SecondOrder_OneSubstrate;
	}
    }
  else
    {
      //FIXME: generic functions should come here.
      //      theGetPropensity_RMethodPtr       = &GillespieProcess::getInf;
      theGetPropensity_RMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getInf>();
      theGetMinValueMethodPtr     = &GillespieProcess::getZero;
    }
}


