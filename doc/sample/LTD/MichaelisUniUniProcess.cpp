#ifndef __MichaelisUniUniProcess_CPP
#define __MichaelisUniUniProcess_CPP

#include <iostream>

#include "libecs.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertySlotMaker.hpp"

#include "FluxProcess.hpp"



namespace libecs
{

  class MichaelisUniUniProcess
    :  
    public FluxProcess
  {
  
  public:

    MichaelisUniUniProcess();
    ~MichaelisUniUniProcess();

    void setKm( RealCref value ) { Km = value; }
    const Real getKm() const { return Km; }
    void setKcat( RealCref value ) { Kcat = value; }
    const Real getKcat() const { return Kcat; }



    
    virtual void process();
    virtual void initialize();
    
    static ProcessPtr createInstance() 
    { 
      return new MichaelisUniUniProcess;
    }
   
    StringLiteral getClassName() const { return "MichaelisUniUniProcess"; }


  protected:
    
    void makeSlots();

    Real Km;
    Real Kcat;




  VariableReference S0;
  VariableReference C0;


  private:



  };

}

using namespace libecs;

extern "C"
{
  Process::AllocatorFuncPtr CreateObject =
  &MichaelisUniUniProcess::createInstance;
}  

MichaelisUniUniProcess::MichaelisUniUniProcess()
{
  makeSlots();
    Km = 0.0;
  Kcat = 0.0;

}

MichaelisUniUniProcess::~MichaelisUniUniProcess()
{
}

void MichaelisUniUniProcess::makeSlots()
{
    DEFINE_PROPERTYSLOT( Real, Km, &MichaelisUniUniProcess::setKm, &MichaelisUniUniProcess::getKm );
  DEFINE_PROPERTYSLOT( Real, Kcat, &MichaelisUniUniProcess::setKcat, &MichaelisUniUniProcess::getKcat );

}


void MichaelisUniUniProcess::initialize()
{
  FluxProcess::initialize();

  
  

  #line 1 "MichaelisUniUniProcess::initialize() in <file:1 (MichaelisUniUniProcess.dm)>"


  declareUnidirectional();

  S0 = getVariableReference( "S0" );
  C0 = getVariableReference( "C0" );




}

void MichaelisUniUniProcess::process()
{
#line 1 "MichaelisUniUniProcess::process() in <file:1 (MichaelisUniUniProcess.dm)>"


  const Real S( S0.getConcentration() );
  const Real E( C0.getValue() );
  Real velocity( (Kcat * E * S /( Km + S)) );
  setFlux( velocity );



}

#endif
