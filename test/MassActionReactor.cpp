
#include <iostream>

#include "libecs/System.hpp"
#include "libecs/Substance.hpp"
#include "libecs/Util.hpp"


#include "MassActionReactor.hpp"


using namespace libecs;

extern "C"
{
  ReactorAllocatorFunc CreateObject =
  &MassActionReactor::createInstance;
}  

MassActionReactor::MassActionReactor()
  :
  K( 0.0 )
{
  makeSlots();
}

MassActionReactor::~MassActionReactor()
{
}

void MassActionReactor::makeSlots()
{
  createPropertySlot( "K", 
		      *this,
		      &MassActionReactor::setK,
		      &MassActionReactor::getK );
}

void MassActionReactor::setK( RealCref value )
{
  K = value;
}
 
void MassActionReactor::initialize()
{
  FluxReactor::initialize();

  SystemPtr aSystemPtr( getReactant( "S0" ).getSubstance()->getSuperSystem() );
  S0SuperSystem_Volume = aSystemPtr->getPropertySlot( "Volume", this );

  theReactantVelocitySlotVector.clear();

  for( ReactantMapIterator s( theReactantMap.begin() );
       s != theReactantMap.end() ; ++s )
    {
      SubstancePtr aSubstance( s->second.getSubstance() );
      theReactantConcentrationSlotVector.
	push_back( aSubstance->getPropertySlot( "Concentration", this ) );
    }
}

void MassActionReactor::differentiate()   
{
  Real velocity( K * N_A );

  velocity *= S0SuperSystem_Volume->getReal(); 

  ReactantMapIterator s( theReactantMap.begin() );
  for( int i( 0 ) ; i < theReactantMap.size() ; ++i )
    {
      int j ( (s[i]).second.getStoichiometry() );
     
      // iterate over only substrates ( stoichiometry > 0 )
      while(j > 0)
	{
	  j--;
	  velocity *= (*theReactantConcentrationSlotVector[i]).getReal();
	}
    }
  process(velocity);

 }

void MassActionReactor::compute()   
{

}


  


