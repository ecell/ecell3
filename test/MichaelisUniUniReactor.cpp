
#include <iostream>

#include "libecs/System.hpp"
#include "libecs/Substance.hpp"
#include "libecs/Util.hpp"
#include "libecs/PropertySlotMaker.hpp"

#include "MichaelisUniUniReactor.hpp"


using namespace libecs;

extern "C"
{
  Reactor::AllocatorFuncPtr CreateObject =
  &MichaelisUniUniReactor::createInstance;
}  

MichaelisUniUniReactor::MichaelisUniUniReactor()
{
  makeSlots();
  S0_Concentration = NULLPTR;
  C0_Quantity = NULLPTR;
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

  if( S0_Concentration == NULLPTR)
    {
      S0_Concentration = getPropertySlotOfReactant( "S0", "Concentration" );
    }
  if( C0_Quantity == NULLPTR)
    {
      C0_Quantity = getPropertySlotOfReactant( "C0", "Quantity" );
    }

}

void MichaelisUniUniReactor::react()   
{
  Real velocity( KcF );

  velocity *= C0_Quantity->getReal();
  const Real S( S0_Concentration->getReal() );
  velocity *= S;
  velocity /= ( KmS + S );

  process( velocity );
}

