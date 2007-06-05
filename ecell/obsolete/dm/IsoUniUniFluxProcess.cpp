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
#include "Util.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( IsoUniUniFluxProcess, ContinuousProcess )
{

  
 public:

  LIBECS_DM_OBJECT( IsoUniUniFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, KmS );
      PROPERTYSLOT_SET_GET( Real, KmP );
      PROPERTYSLOT_SET_GET( Real, KcF );
      PROPERTYSLOT_SET_GET( Real, Keq );
      PROPERTYSLOT_SET_GET( Real, KiiP );
    }

  IsoUniUniFluxProcess()
    :
    KmS( 1.0 ),
    KmP( 1.0 ),
    KcF( 0.0 ),
    Keq( 0.0 ),
    KiiP( 1.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, KmS );
  SIMPLE_SET_GET_METHOD( Real, KmP );
  SIMPLE_SET_GET_METHOD( Real, KcF );
  SIMPLE_SET_GET_METHOD( Real, Keq );
  SIMPLE_SET_GET_METHOD( Real, KiiP );
    
  virtual void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      P0 = getVariableReference( "P0" );
      C0 = getVariableReference( "C0" );  
    }

  virtual void fire()
    {
      Real S( S0.getMolarConc() );
      Real P( P0.getMolarConc() );
      Real velocity( KcF * C0.getMolarConc() * 
	(S-P/Keq) / KmS * (1+P/KmP) + S * (1+P/KiiP) );

      setFlux( velocity );
    }

 protected:
  

  Real KmS;
  Real KmP;
  Real KcF;
  Real Keq;
  
  Real KiiP;

  VariableReference S0;
  VariableReference P0;
  VariableReference C0;
  
};

LIBECS_DM_INIT( IsoUniUniFluxProcess, Process );
