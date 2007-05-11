
#include "GillespieProcess.hpp"


LIBECS_DM_INIT( GillespieProcess, Process );

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

  // set theGetPropensityMethodPtr and theGetMinValueMethodPtr

  if( getOrder() == 0 )   // no substrate
    {
      theGetPropensityMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getZero>();
      theGetMinValueMethodPtr   = RealMethodProxy::
	create<&GillespieProcess::getZero>();
      theGetPDMethodPtr         = &GillespieProcess::getPD_Zero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      theGetPropensityMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getPropensity_FirstOrder>();
      theGetMinValueMethodPtr   = RealMethodProxy::
	create<&GillespieProcess::getMinValue_FirstOrder>();
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
      theGetPropensityMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getZero>();
      theGetPropensityMethodPtr = RealMethodProxy::
	create<&GillespieProcess::getZero>();
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
