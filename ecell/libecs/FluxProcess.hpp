#ifndef __FLUXPROCESS_HPP
#define __FLUXPROCESS_HPP

#include "libecs/libecs.hpp"
#include "libecs/Process.hpp"
#include "libecs/VariableReference.hpp"
#include "libecs/Variable.hpp"
#include "libecs/Stepper.hpp"

namespace libecs
{

  class FluxProcess         
    :  
    public Process
  {
  
  public:

    FluxProcess()
    {
      ; // do nothing
    }

    ~FluxProcess()
    {
      ; // do nothing
    }

    StringLiteral getClassName() const { return "FluxProcess"; }
    
    void initialize()
    {
      Process::initialize();
    }

    void setFlux( RealCref velocity )
    {
      Real aVelocity( velocity );

      setActivity( aVelocity );

      // Increase or decrease variables.

      for( VariableReferenceMapIterator s( theVariableReferenceMap.begin() );
	   s != theVariableReferenceMap.end() ; ++s )
	{
	  VariableReference aVariableReference( s->second );
	  const Int aCoefficient( aVariableReference.getCoefficient() );
	  if( aCoefficient != 0 )
	    {
	      aVariableReference.getVariable()->
		addVelocity( aVelocity * aCoefficient );
	    }
	}
      
    }
    
  };

}

#endif /* __FluxProcess_HPP */







