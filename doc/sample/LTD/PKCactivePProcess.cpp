#include "libecs.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"
#include "PropertySlotMaker.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "VariableProxy.hpp"

#include "ecell3_dm.hpp"

#define ECELL3_DM_TYPE Process
#define ECELL3_DM_CLASSNAME PKCactivePProcess

USE_LIBECS;

ECELL3_DM_CLASS
  :  
  public Process
{
  
  ECELL3_DM_OBJECT;
  
 public:
    
  void initialize()
    {
      Process::initialize();
      S0 = getVariableReference( "S0" );
      S1 = getVariableReference( "S1" );
      S2 = getVariableReference( "S2" );
      S3 = getVariableReference( "S3" );
      P0 = getVariableReference( "P0" );
    }

  void process()
    {
      const Real s0( S0.getValue() );
      const Real s1( S1.getValue() );
      const Real s2( S2.getValue() );
      const Real s3( S3.getValue() );
      
      const Real p( s0 + s1 + s2 + s3 );
      
      P0.setValue( p );
    }

 protected:

  VariableReference S0;
  VariableReference S1;
  VariableReference S2;
  VariableReference S3;
  VariableReference P0;

};

ECELL3_DM_INIT;
