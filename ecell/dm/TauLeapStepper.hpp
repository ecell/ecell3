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

#ifndef __TAULEAP_HPP
#define __TAULEAP_HPP

#include "TauLeapProcess.hpp"

#include "libecs/DifferentialStepper.hpp"
#include "libecs/libecs.hpp"

#include <vector>

USE_LIBECS;

DECLARE_CLASS( TauLeapProcess );
DECLARE_VECTOR( TauLeapProcessPtr, TauLeapProcessVector );

LIBECS_DM_CLASS( TauLeapStepper, DifferentialStepper )
{  
  
public:

  LIBECS_DM_OBJECT( TauLeapStepper, Stepper )
    {
      INHERIT_PROPERTIES( DifferentialStepper );
      PROPERTYSLOT_SET_GET( Real, Epsilon );
    }

  TauLeapStepper( void )
    :
    Epsilon( 0.03 )
    {
      ; // do nothing
    }
  
  virtual ~TauLeapStepper( void )
    {
      ; // do nothing
    }  
  
  SIMPLE_SET_GET_METHOD( Real, Epsilon );

  void initialize()
    {
      DifferentialStepper::initialize();
      
      theTauLeapProcessVector.clear();
      theTauLeapProcessVector.resize( theProcessVector.size() );
      
      for( ProcessVector::size_type i( 0 ); i < theProcessVector.size(); ++i )
	{
	  TauLeapProcessPtr aTauLeapProcessPtr( dynamic_cast<TauLeapProcessPtr>( theProcessVector[ i ] ) );
	  theTauLeapProcessVector[ i ] = aTauLeapProcessPtr;
	}

      // resize tmp vector.
      theFFVector.clear();
      theMeanVector.clear();
      theVarianceVector.clear();
      theFFVector.resize( theProcessVector.size() );
      theMeanVector.resize( theProcessVector.size() );
      theVarianceVector.resize( theProcessVector.size() );

    }

  void step()
    {      
      const VariableVector::size_type aSize( getReadOnlyVariableOffset() );
      
      clearVariables();
      
      setStepInterval( getTau() );
      
      fireProcesses();
      
      for( VariableVector::size_type c( 0 ); c < aSize; ++c )
	{
	  VariablePtr const aVariable( theVariableVector[ c ] );
	  theTaylorSeries[ 0 ][ c ] = aVariable->getVelocity();
	}
    }

 protected:
  
  const Real getTotalPropensity( )
    {
      Real anA0( 0.0 );
      for( std::vector<TauLeapProcessPtr>::iterator i( theTauLeapProcessVector.begin() ); 
	   i != theTauLeapProcessVector.end(); ++i )
	{
	  anA0 += (*i)->getPropensity();
	}
      return anA0;
    }

  const Real getTau( );
  
 protected:
  
  Real Epsilon;
  std::vector< TauLeapProcessPtr > theTauLeapProcessVector;
  
  // tmp vectors.
  RealVector theFFVector;
  RealVector theMeanVector;
  RealVector theVarianceVector;

};

#endif /* __TAULEAP_HPP */
