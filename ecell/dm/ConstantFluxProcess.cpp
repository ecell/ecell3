#include "libecs.hpp"

#include "ContinuousProcess.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( ConstantFluxProcess, ContinuousProcess )
{

 public:

  LIBECS_DM_OBJECT( ConstantFluxProcess, Process )
    {
      INHERIT_PROPERTIES( ContinuousProcess );

      PROPERTYSLOT_SET_GET( Real, k);
    }

  ConstantFluxProcess()
    :
    k( 0.0 )
    {
      ; // do nothing
    }
  
  SIMPLE_SET_GET_METHOD( Real, k );
  
  virtual void initialize()
  {
    Process::initialize();
  
    // force unset isAccessor flag of all variablereferences.
    std::for_each( theVariableReferenceVector.begin(),
		   theVariableReferenceVector.end(),
		   std::bind2nd
		   ( std::mem_fun_ref
		     ( &VariableReference::setIsAccessor ), false ) );
  }  

  virtual void fire()
  {
    // constant flux
    setFlux( k );
  }
  
 protected:
  
  Real k;
    
};

LIBECS_DM_INIT( ConstantFluxProcess, Process );
