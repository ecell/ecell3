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
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Interpolant.hpp"

#include "Process.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( RapidEquilibriumProcess, Process )
{

 public:

  LIBECS_DM_OBJECT( RapidEquilibriumProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, Keq );
    }

  // FIXME: initial values
  RapidEquilibriumProcess()
    {
      ; // do nothing
    }

  GET_METHOD( Real, Keq )
    {
      return theOriginalKeq;
    }

  SET_METHOD( Real, Keq )
    {
      theOriginalKeq = value;
    }

  void fire();

  void initialize()
    {

      Process::initialize();
      Integer d_Keq( 0 );
      for( VariableReferenceVectorConstIterator
	     i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
	{
	  d_Keq += (*i).getCoefficient();
	}

      for( VariableReferenceVectorConstIterator
	     i ( thePositiveVariableReferenceIterator );
           i != theVariableReferenceVector.end() ; ++i )
	{
	  d_Keq += (*i).getCoefficient();
	}

      Keq = theOriginalKeq * pow(N_A, d_Keq);

    }

 protected:

    Real Keq;
    Real theOriginalKeq;
};

LIBECS_DM_INIT( RapidEquilibriumProcess, Process );


void RapidEquilibriumProcess::fire()
{
 
  Real d( 0 );
  Integer figure( 0 );
  Real velocity( 0 );
  Real velocity_posi( 1 );
  Real velocity_nega( 1 );
  Integer least( 0 );
  Real Keq_qua( 1 );
  
  for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
  {
    for(Integer j( 0 ); j < -(*i).getCoefficient(); j++ )
    {
      Keq_qua *= getSuperSystem()->getSize();
    }
  }

  Keq_qua = Keq / Keq_qua;
  for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
  {
    for(Integer j( 0 ); j < (*i).getCoefficient(); j++ )
    {
      Keq_qua *= getSuperSystem()->getSize();
    }
  }  

  for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
  {
    for(Integer j( 0 ); j < -(*i).getCoefficient(); j++ )
    {
      velocity_posi *= (*i).getValue();
    }
  }
  velocity_posi *= Keq_qua;
  
  for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
  {
    for(Integer j( 0 ); j < (*i).getCoefficient(); j++ )
    {
      velocity_nega *= (*i).getValue();
    }
  }  

  if( velocity_posi > velocity_nega )
  {
    VariableReferenceVectorConstIterator least( theVariableReferenceVector.begin() );
    for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
    {
      if( (*i).getValue() / -(*i).getCoefficient() < (*least).getValue() / -(*least).getCoefficient() )
      {
        least = i;
      }
    }
    
    figure = (Integer)log10( -(*least).getValue() / -(*least).getCoefficient() );
    d = pow(10.0,figure);
    
    while(figure >= 0){
      
      velocity += d;
  
      if( (*least).getValue() < velocity * -(*least).getCoefficient())
      {
        velocity -= d;
        figure--;
        d /= 10;
        continue;
      }
  
      velocity_posi = 1; 
      velocity_nega = 1;

      for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
      {
        for(Integer j( 0 ); j < -(*i).getCoefficient(); j++ )
        {
          velocity_posi *= (*i).getValue() - ( velocity * -(*i).getCoefficient() );
        }
      }
      velocity_posi *= Keq_qua;

      for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
      {
        for(Integer j( 0 ); j < (*i).getCoefficient(); j++ )
        {
          velocity_nega *= (*i).getValue() + ( velocity * (*i).getCoefficient() );
        }
      }  

      if( velocity_posi == velocity_nega )
      {
        break;
      }else if( velocity_posi < velocity_nega )
      {
        velocity -= d;
        figure--;
        d /= 10;
      }else{
        continue;
      }    
    }
  
  }else if(velocity_posi < velocity_nega ){

    VariableReferenceVectorConstIterator least( thePositiveVariableReferenceIterator );

    for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
    {
       if( (*i).getValue() / (*i).getCoefficient() < (*least).getValue() / (*least).getCoefficient() )
       {
         least = i;
       }
    }
    figure = (Integer)log10( (Real)(*least).getValue() / (*least).getCoefficient() );

    d = pow(10.0,figure);
  
    while(figure >= 0){
      velocity -= d;
  
      if( (*least).getValue() + ( velocity * (*least).getCoefficient() ) < 0 ){
        velocity += d;
        figure--;
        d *= 10;
        continue;
      }
  
      velocity_posi = 1; 
      velocity_nega = 1;
      for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
      {
        for(Integer j( 0 ); j < -(*i).getCoefficient(); j++ )
        {
          velocity_posi *= (*i).getValue() - ( velocity * -(*i).getCoefficient() );
        }
      }

      velocity_posi *= Keq_qua;

      for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
      {
        for(Integer j( 0 ); j < (*i).getCoefficient(); j++ )
        {
          velocity_nega *= (*i).getValue() + ( velocity * (*i).getCoefficient() );
        }
      }  
  
      if( velocity_posi < velocity_nega ){
        continue;
      }else if( velocity_posi > velocity_nega ){
        velocity += d;
        figure--;
        d /= 10;
      }else{
        break;
      }
    }
  }else{
    velocity = 0;
  }

  for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
    {
      (*i).setValue( (*i).getValue() - velocity );
    }

  for( VariableReferenceVectorConstIterator
         i ( thePositiveVariableReferenceIterator );
           i != theVariableReferenceVector.end() ; ++i )
    {
      (*i).setValue( (*i).getValue() + velocity );
    }

}
