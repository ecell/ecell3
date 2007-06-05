//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
#include "libecs.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( OrderedUniBiFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( OrderedUniBiFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, KcR );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP0 );
      PROPERTYSLOT_SET_GET( Real, KmP1 );
      PROPERTYSLOT_SET_GET( Real, KiP );
    }
  

  OrderedUniBiFluxProcess()
    :
    KcF( 0.0 ),
    KcR( 0.0 ),
    Keq( 1.0 ),
    KmS( 1.0 ),
    KmP0( 1.0 ),
    KmP1( 1.0 ),
    KiP( 1.0 ),
    Keq_Inv( 1.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, KcR );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KmP0 );
  SIMPLE_SET_GET_METHOD( Real, KmP1 );
  SIMPLE_SET_GET_METHOD( Real, KiP );
    
  virtual void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      P0 = getVariableReference( "P0" );
      P1 = getVariableReference( "P1" );
      C0 = getVariableReference( "C0" );

      Keq_Inv = 1.0 / Keq;
    }

  virtual void fire()
    {
      Real S0Concentration = S0.getMolarConc();
      Real P0Concentration = P0.getMolarConc();
      Real P1Concentration = P1.getMolarConc();
      
      Real Den( KcR * KmS + KcR * S0Concentration 
		+ KcF * KmP1 * P0Concentration * Keq_Inv
		+ KcF * KmP0 * P1Concentration * Keq_Inv
		+ KcR * S0Concentration * P0Concentration / KiP
		+ KcR * P0Concentration * P1Concentration * Keq_Inv );
      Real velocity( KcF * KcR * C0.getValue()
		     * (S0Concentration - P0Concentration * 
			P1Concentration * Keq_Inv) / Den );
      setFlux( velocity );
    }

 protected:

  Real KcF;
  Real KcR;
  Real Keq;

  Real KmS;
  Real KmP0;
  Real KmP1;
  Real KiP;

  Real Keq_Inv;

  VariableReference S0;
  VariableReference P0;
  VariableReference P1;
  VariableReference C0;
  
};

LIBECS_DM_INIT( OrderedUniBiFluxProcess, Process );
