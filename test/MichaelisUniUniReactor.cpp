
#include <iostream>

#include "libecs/System.hpp"
#include "libecs/Substance.hpp"
#include "libecs/Util.hpp"
#include "libecs/PropertySlotMaker.hpp"

#include "MichaelisUniUniReactor.hpp"


using namespace libecs;

extern "C"
{
  ReactorAllocatorFunc CreateObject =
  &MichaelisUniUniReactor::createInstance;
}  

MichaelisUniUniReactor::MichaelisUniUniReactor()
{
  makeSlots();
  KmS = 0.0;
  KcF = 0.0;
}

MichaelisUniUniReactor::~MichaelisUniUniReactor()
{
}

void MichaelisUniUniReactor::makeSlots()
{
  registerSlot( getPropertySlotMaker()->
		createPropertySlot( "KmS", 
				    *this,
				    Type2Type<Real>(),
				    &MichaelisUniUniReactor::setKmS,
				    &MichaelisUniUniReactor::getKmS ) );

  registerSlot( getPropertySlotMaker()->
		createPropertySlot( "KcF",
				    *this,
				    Type2Type<Real>(),
				    &MichaelisUniUniReactor::setKcF,
				    &MichaelisUniUniReactor::getKcF ) );

}

void MichaelisUniUniReactor::setKmS( RealCref value )
{
  KmS = value;
}

void MichaelisUniUniReactor::setKcF( RealCref value )
{
  KcF = value;
}



void MichaelisUniUniReactor::initialize()
{
  FluxReactor::initialize();

  S0_Concentration = getPropertySlotOfReactant( "S0", "Concentration" );
  C0_Quantity = getPropertySlotOfReactant( "C0", "Quantity" );

}

void MichaelisUniUniReactor::react()   
{
  Real velocity( KcF );

  //velocity *= getCatalyst(0)->getSubstance().getQuantity();
  //  Real S = getSubstrate(0)->getSubstance().getConcentration();
  

  velocity *= C0_Quantity->getReal();
  const Real S( S0_Concentration->getReal() );
  velocity *= S;
  velocity /= ( KmS + S );

  process( velocity );
}

void MichaelisUniUniReactor::compute()   
{

}
