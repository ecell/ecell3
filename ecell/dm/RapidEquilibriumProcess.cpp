#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

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

  SIMPLE_SET_GET_METHOD( Real, Keq );
    
  void process();

  void initialize()
    {

      Process::initialize();
      Int d_Keq( 0 );
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

      Keq *= pow(N_A, d_Keq);

    }

 protected:

    Real Keq;
};

LIBECS_DM_INIT( RapidEquilibriumProcess, Process );


void RapidEquilibriumProcess::process()
{
 
  Real d( 0 );
  Int figure( 0 );
  Real velocity( 0 );
  Real velocity_posi( 1 );
  Real velocity_nega( 1 );
  Int least( 0 );
  Real Keq_qua( 1 );
  
  for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
  {
    for(Int j( 0 ); j < -(*i).getCoefficient(); j++ )
    {
      Keq_qua *= getSuperSystem()->getSize();
    }
  }

  Keq_qua = Keq / Keq_qua;
  for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
  {
    for(Int j( 0 ); j < (*i).getCoefficient(); j++ )
    {
      Keq_qua *= getSuperSystem()->getSize();
    }
  }  

  for( VariableReferenceVectorConstIterator
         i ( theVariableReferenceVector.begin() );
           i != theZeroVariableReferenceIterator ; ++i )
  {
    for(Int j( 0 ); j < -(*i).getCoefficient(); j++ )
    {
      velocity_posi *= (*i).getValue();
    }
  }
  velocity_posi *= Keq_qua;
  
  for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
  {
    for(Int j( 0 ); j < (*i).getCoefficient(); j++ )
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
    
    figure = (Int)log10( -(*least).getValue() / -(*least).getCoefficient() );
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
        for(Int j( 0 ); j < -(*i).getCoefficient(); j++ )
        {
          velocity_posi *= (*i).getValue() - ( velocity * -(*i).getCoefficient() );
        }
      }
      velocity_posi *= Keq_qua;

      for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
      {
        for(Int j( 0 ); j < (*i).getCoefficient(); j++ )
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
    figure = (Int)log10( (Real)(*least).getValue() / (*least).getCoefficient() );

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
        for(Int j( 0 ); j < -(*i).getCoefficient(); j++ )
        {
          velocity_posi *= (*i).getValue() - ( velocity * -(*i).getCoefficient() );
        }
      }

      velocity_posi *= Keq_qua;

      for( VariableReferenceVectorConstIterator
       i ( thePositiveVariableReferenceIterator );
         i != theVariableReferenceVector.end() ; ++i )
      {
        for(Int j( 0 ); j < (*i).getCoefficient(); j++ )
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
