#ifndef __FLUXPROCESS_HPP
#define __FLUXPROCESS_HPP

#include "libecs/libecs.hpp"
#include "libecs/Process.hpp"
#include "libecs/Connection.hpp"
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

      // Increase or decrease connections.

      for( ConnectionMapIterator s( theConnectionMap.begin() );
	   s != theConnectionMap.end() ; ++s )
	{
	  Connection aConnection( s->second );
	  const Int aCoefficient( aConnection.getCoefficient() );
	  if( aCoefficient != 0 )
	    {
	      aConnection.getVariable()->
		addVelocity( aVelocity * aCoefficient );
	    }
	}
      
    }
    
  };

}

#endif /* __FluxProcess_HPP */







