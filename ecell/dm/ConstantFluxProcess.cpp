#include "libecs.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

#include "System.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ConstantFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( ConstantFluxProcess, Process )
    {
      INHERIT_PROPERTIES( Process );

      PROPERTYSLOT_SET_GET( Real, Flux );
    }

  ConstantFluxProcess()
    :
    Flux( 0.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, Flux );
  
  virtual void process()
  {
    setFlux(Flux);
  }
  
  virtual void initialize()
  {
    Process::initialize();
    declareUnidirectional();
  }  

 protected:
  
  Real Flux;
    
};

LIBECS_DM_INIT( ConstantFluxProcess, Process );
