#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "FluxProcess.hpp"
#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public FluxProcess
{

  ECELL3_DM_OBJECT;
  
 public:

  ECELL3_DM_CLASSNAME()
    {
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k_1 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k2 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k_2 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k3 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k_3 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k4 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k_4 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k5 );
      ECELL3_CREATE_PROPERTYSLOT_SET_GET( Real, k_5 );
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
      FluxProcess::initialize();
      S_0 = getVariableReference( "S0" );
      P_0 = getVariableReference( "P0" );
      P_1 = getVariableReference( "P1" );
      C_0 = getVariableReference( "C0" );
    }

  virtual void process()
    {
      Real velocity( C_0.getValue() );
      
      Real S0 = S_0.getConcentration();
      Real P0 = P_0.getConcentration();
      Real P1 = P_1.getConcentration();
      
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

ECELL3_DM_INIT;
