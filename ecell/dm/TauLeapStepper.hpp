//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Tomoya Kitayama <tomo@e-cell.org>, 
// E-Cell Project.
//

#ifndef __TAULEAP_HPP
#define __TAULEAP_HPP

#ifdef WIN32
// Avoid conflicts with min() / max() macros in windows.h
#define NOMINMAX
#endif /* WIN32 */

#include "GillespieProcess.hpp"

#include "libecs/DifferentialStepper.hpp"
#include "libecs/libecs.hpp"

USE_LIBECS;

DECLARE_CLASS( GillespieProcess );
DECLARE_VECTOR( GillespieProcessPtr, GillespieProcessVector );

LIBECS_DM_CLASS( TauLeapStepper, DifferentialStepper )
{  
  
public:

  LIBECS_DM_OBJECT( TauLeapStepper, Stepper )
    {
      INHERIT_PROPERTIES( DifferentialStepper );

      PROPERTYSLOT_SET_GET( Real, Epsilon );
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Tau );
    }

  TauLeapStepper( void )
    :
    epsilon( 0.03 ),
    tau( libecs::INF )
    {
      ; // do nothing
    }
  
  virtual ~TauLeapStepper( void )
    {
      ; // do nothing
    }  

  virtual void initialize();
  virtual void step();
  
  GET_METHOD( Real, Epsilon )
  {
    return epsilon;
  }

  SET_METHOD( Real, Epsilon )
  {
    epsilon = value;
  }

  GET_METHOD( Real, Tau )
  {
    return tau;
  }

 protected:

  const Real getTotalPropensity()
    {
      Real totalPropensity( 0.0 );
      FOR_ALL( GillespieProcessVector, theGillespieProcessVector )
	{
	  totalPropensity += (*i)->getPropensity();
	}

      return totalPropensity;
    }

  void calculateTau()
    {
      tau = libecs::INF;

      const Real totalPropensity( getTotalPropensity() );
      
      const GillespieProcessVector::size_type 
	aSize( theGillespieProcessVector.size() );  
      for( GillespieProcessVector::size_type i( 0 ); i < aSize; ++i )
	{
	  Real aMean( 0.0 );
	  Real aVariance( 0.0 );
	  
	  for( GillespieProcessVector::size_type j( 0 ); j < aSize; ++j )
	    {
	      const Real aPropensity
		( theGillespieProcessVector[ j ]->getPropensity() );
	      VariableReferenceVectorCref aVariableReferenceVector
		( theGillespieProcessVector[ j ]->getVariableReferenceVector() );
	      
	      // future works : theDependentProcessVector
	      Real expectedChange( 0.0 );
	      for( VariableReferenceVectorConstIterator 
		     k( aVariableReferenceVector.begin() ); 
		   k != aVariableReferenceVector.end(); ++k )
		{
		  expectedChange += theGillespieProcessVector[ i ]->getPD( (*k).getVariable() ) * (*k).getCoefficient();
		}
	      
	      aMean += expectedChange * aPropensity;
	      aVariance += expectedChange * expectedChange * aPropensity;
	    }
	  
	  const Real aTolerance( epsilon * totalPropensity );
	  const Real expectedTau
	    ( std::min( aTolerance / std::abs( aMean ), 
			aTolerance * aTolerance / aVariance ) );
	  if ( expectedTau < tau )
	    {
	      tau = expectedTau;
	    }
	}
    }

 protected:
  
  Real epsilon;
  Real tau;
  GillespieProcessVector theGillespieProcessVector;

};

#endif /* __TAULEAP_HPP */
