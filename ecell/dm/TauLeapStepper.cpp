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
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include "TauLeapStepper.hpp"
 
LIBECS_DM_INIT( TauLeapStepper, Stepper );

const Real TauLeapStepper::getTau( )
{
  const Real anA0( getTotalPropensity() );      
  
  TauLeapProcessVector::size_type aTauLeapProcessVectorSize( theTauLeapProcessVector.size() );
  for( TauLeapProcessVector::size_type i( 0 ); i < aTauLeapProcessVectorSize; ++i )
    {
      for( TauLeapProcessVector::size_type j( 0 ); j < aTauLeapProcessVectorSize; ++j )
	{
	  Real aFF( 0 );
	  VariableReferenceVector aVariableReferenceVector( theTauLeapProcessVector[j]->getVariableReferenceVector() );
	  
	  VariableReferenceVector::size_type aVariableReferenceVectorSize( aVariableReferenceVector.size() );
	  for( VariableReferenceVector::size_type k( 0 ); 
	       k < aVariableReferenceVectorSize; ++k )
	    {
	      VariableReferenceRef aVariableReferenceRef( aVariableReferenceVector[k] );
	      aFF += theTauLeapProcessVector[i]->getPD( aVariableReferenceRef.getVariable() ) * aVariableReferenceRef.getCoefficient();
	    }
	  
	  theFFVector[j] = aFF;
	}
      
      Real aMean( 0 );
      Real aVariance( 0 );
      
      RealVector::size_type aFFVectorSize( theFFVector.size() );
      for( RealVector::size_type j( 0 ); j < aFFVectorSize; ++j )
	{
	  aMean += theFFVector[j] * theTauLeapProcessVector[j]->getPropensity();
	  aVariance += pow( theFFVector[j], 2 ) * theTauLeapProcessVector[j]->getPropensity();
	}
      theMeanVector[i] = Epsilon * anA0 / std::abs( aMean );
      theVarianceVector[i] = pow( Epsilon, 2 ) * pow( anA0, 2 ) / aVariance;
    }
  
  return std::min( *std::min_element( theMeanVector.begin(), theMeanVector.end() ), 
		   *std::min_element( theVarianceVector.begin(), theVarianceVector.end() ) );
  
}
