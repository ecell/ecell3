#line 1 "/home/tomo/wrk/munch/dmtemplates/reactor/reactor.tmpl"


#include "libecs/libecs.hpp"
#include "FluxReactor.hpp"

namespace libecs
{

  class MichaelisUniUniReactor
    :  
    public FluxReactor
  {
  
  public:

    MichaelisUniUniReactor();
    ~MichaelisUniUniReactor();

    void setKmS( RealCref value ) { KmS = value; }
    const Real getKmS() const { return KmS; }
    void setKcF( RealCref value ) { KcF = value; }
    const Real getKcF() const { return KcF; }




    virtual void initialize();

    static ReactorPtr createInstance() 
    { 
      return new MichaelisUniUniReactor;
    }
   
    StringLiteral getClassName() const { return "MichaelisUniUniReactor"; }


  protected:
    
    void makeSlots();
            
    Real KmS;
    Real KcF;


PropertySlotPtr S0_Concentration;
PropertySlotPtr C0_Quantity;


  };


}


#include <iostream>

#include "libecs/System.hpp"
#include "libecs/Substance.hpp"
#include "libecs/Util.hpp"


//#include "MichaelisUniUniReactor.hpp"


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
createPropertySlot( "KmS", *this, &MichaelisUniUniReactor::setKmS, &MichaelisUniUniReactor::getKmS );
createPropertySlot( "KcF", *this, &MichaelisUniUniReactor::setKcF, &MichaelisUniUniReactor::getKcF );

}


void MichaelisUniUniReactor::initialize()
{
  FluxReactor::initialize();

S0_Concentration = getPropertySlotOfReactant( "S0", "Concentration" );
C0_Quantity = getPropertySlotOfReactant( "C0", "Quantity" );



}


void MichaelisUniUniReactor::react()
{
#line 83 "/home/tomo/wrk/munch/dmtemplates/reactor/reactor.tmpl"

  Real velocity( KcF );
  velocity *= C0_Quantity->getReal();
  const Real S( S0_Concentration->getReal() );
  velocity *= S;
  velocity /= ( KmS + S );

  process( velocity );

}

#line 91 "/home/tomo/wrk/munch/dmtemplates/reactor/reactor.tmpl"

