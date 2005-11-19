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
      theGetPropensityMethodPtr = &TauLeapProcess::getZero;
      theGetPDMethodPtr = &TauLeapProcess::getZero;
    }
  else if( getOrder() == 1 )   // one substrate, first order.
    {
      theGetPropensityMethodPtr = &TauLeapProcess::getPropensity_FirstOrder;
      theGetPDMethodPtr = &TauLeapProcess::getPD_FirstOrder;
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
      theGetPropensityMethodPtr = &TauLeapProcess::getZero;
      theGetPDMethodPtr = &TauLeapProcess::getZero;
    }
}
