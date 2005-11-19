//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 2002 Keio University
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
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#include "Variable.hpp"
#include "Process.hpp"
#include "PropertyInterface.hpp"
#include "ESSYNSStepper.hpp"

#include <gsl/gsl_sf.h>

LIBECS_DM_INIT( ESSYNSStepper, Stepper );

void ESSYNSStepper::initialize()
{
  AdaptiveDifferentialStepper::initialize();
 
  // initialize()

  if( theProcessVector.size() == 1 )
    {
      theESSYNSProcessPtr = DynamicCaster<ESSYNSProcessPtr,ProcessPtr>()( theProcessVector[ 0 ]);
	
      theSystemSize = theESSYNSProcessPtr->getSystemSize();
    }
  else
    {
      THROW_EXCEPTION( InitializationFailed, 
		       "Error:in ESYYNSStepper::initialize() " );
    }

  theTaylorOrder = getOrder();

  theESSYNSMatrix.resize(theSystemSize+1);
  RealVector tmp;
  tmp.resize(theTaylorOrder+1);

  for(int i( 0 ); i < theSystemSize; i++)
    {
      theESSYNSMatrix[i] = tmp;
    }

  theIndexVector.resize( theSystemSize );
  VariableReferenceVectorCref aVariableReferenceVectorCref
    ( theESSYNSProcessPtr->getVariableReferenceVector() );

  for ( VariableReferenceVector::size_type c( theESSYNSProcessPtr->getPositiveVariableReferenceOffset() ); c < theSystemSize; c++  )
    {
      VariableReferenceCref aVariableReferenceCref( aVariableReferenceVectorCref[ c ] );
      const VariablePtr aVariablePtr( aVariableReferenceCref.getVariable() );

      theIndexVector[ c ] = getVariableIndex( aVariablePtr );
    }

  /* for( int i( 1 ); i < theSystemSize+1; i++)
     {
     std::cout<< (theESSYNSMatrix[i-1])[0] << std::endl;
     }
  */
}

bool ESSYNSStepper::calculate()
{
  const VariableVector::size_type aSize( getReadOnlyVariableOffset() );

  Real aCurrentTime( getCurrentTime() );
  Real aStepInterval( getStepInterval() );

  // write step() function
  
  theESSYNSMatrix = theESSYNSProcessPtr->getESSYNSMatrix();

  //integrate
  for( int i( 1 ); i < theSystemSize+1; i++ )
    {
      Real aY( 0.0 ); //reset aY 
      for( int m( 1 ); m <= theTaylorOrder; m++ )
	{
	  aY += ((theESSYNSMatrix[i-1])[m] *
		 gsl_sf_pow_int( aStepInterval, m ) / gsl_sf_fact( m ));
	}
      (theESSYNSMatrix[i-1])[0] += aY;
      //std::cout<< (theESSYNSMatrix[i-1])[0] <<std::endl;
    }
    
  //set value
  for( int c( 0 ); c < aSize; ++c )
    {
      const VariableVector::size_type anIndex( theIndexVector[ c ] );
      VariablePtr const aVariable( theVariableVector[ anIndex ] );
	
      const Real aVelocity( ( exp( (theESSYNSMatrix[c])[0] ) - ( aVariable->getValue() ) ) / aStepInterval );
		     
      theTaylorSeries[ 0 ][ anIndex ] = aVelocity;
    }

  return true;
}
 
