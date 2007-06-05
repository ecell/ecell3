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

LIBECS_DM_CLASS( RandomUniBiFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( RandomUniBiFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );
      
      PROPERTYSLOT_SET_GET( Real, k1 );
      PROPERTYSLOT_SET_GET( Real, k_1 );
      PROPERTYSLOT_SET_GET( Real, k2 );
      PROPERTYSLOT_SET_GET( Real, k_2 );
      PROPERTYSLOT_SET_GET( Real, k3 );
      PROPERTYSLOT_SET_GET( Real, k_3 );
      PROPERTYSLOT_SET_GET( Real, k4 );
      PROPERTYSLOT_SET_GET( Real, k_4 );
      PROPERTYSLOT_SET_GET( Real, k5 );
      PROPERTYSLOT_SET_GET( Real, k_5 );
    }

  RandomUniBiFluxProcess()
    :
    k1( 0.0 ),
    k_1( 0.0 ),
    k2( 0.0 ),
    k_2( 0.0 ),
    k3( 0.0 ),
    k_3( 0.0 ),
    k4( 0.0 ),
    k_4( 0.0 ),
    k5( 0.0 ),
    k_5( 0.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, k1 );
  SIMPLE_SET_GET_METHOD( Real, k_1 );
  SIMPLE_SET_GET_METHOD( Real, k2 );
  SIMPLE_SET_GET_METHOD( Real, k_2 );
  SIMPLE_SET_GET_METHOD( Real, k3 );
  SIMPLE_SET_GET_METHOD( Real, k_3 );
  SIMPLE_SET_GET_METHOD( Real, k4 );
  SIMPLE_SET_GET_METHOD( Real, k_4 );
  SIMPLE_SET_GET_METHOD( Real, k5 );
  SIMPLE_SET_GET_METHOD( Real, k_5 );
    
  virtual void initialize()
    {
      Process::initialize();
      S_0 = getVariableReference( "S0" );
      P_0 = getVariableReference( "P0" );
      P_1 = getVariableReference( "P1" );
      C_0 = getVariableReference( "C0" );
    }

  virtual void fire()
    {
      Real velocity( C_0.getValue() );
      
      Real S0 = S_0.getMolarConc();
      Real P0 = P_0.getMolarConc();
      Real P1 = P_1.getMolarConc();
      
      Real MP( k1*k4*k5*(k2+k3)*S0+k1*k2*k4*k_3*S0*P0+k_1*(k_2*k_4*k5-k4*k_3*k_5)*P0*P1+k1*k_2*k3*k5*S0*P1-k_1*k_2*k_4*k_3*pow(P0,2)*P1-k_1*k_2*k_3*k_5*P0*pow(P1,2) );

      Real Den( k4*k5*(k_1+k2+k3)+k1*(k2*k5+k4*k5+k4*k3)*S0+(k_1*k_2*k5+k_2*k3*k5+k_1*k4*k_5+k2*k4*k_5+k4*k3*k_5)*P1+(k_1*k4*k_3+k2*k4*k_3+k_1*k_4*k5+k2*k_4*k5+k_4*k3*k5)*P0+k1*k_3*(k2+k4)*S0*P0+k1*k_2*(k5+k3)*S0*P1+(k_1*k_2*k_3+k_2*k_4*k5+k_2*k_4*k3+k2*k_3*k_5+k4*k_3*k_5)*P0*P1+k_4*k_3*(k_1+k2)*pow(P0,2)+k_2*k_5*(k_1+k3)*pow(P1,2)+k1*k_2*k_3*S0*P0*P1+k_2*k_4*k_3*pow(P0,2)*P1+k_2*k_3*k_5*P0*pow(P1,2) );
      velocity *= MP;
      velocity /= Den;

      setFlux( velocity );
    }

 protected:

  Real k1;
  Real k_1;
  Real k2;
  Real k_2;
  Real k3;
  Real k_3;
  Real k4;
  Real k_4;
  Real k5;
  Real k_5;
   
  VariableReference S_0;
  VariableReference P_0;
  VariableReference P_1;
  VariableReference C_0;
  
};

LIBECS_DM_INIT( RandomUniBiFluxProcess, Process );
