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

    inline void setVariableFlux( VariableReferenceCref aVariableReference, 
				 const Real aVelocity )
    {
      const Int aCoefficient( aVariableReference.getCoefficient() );
      aVariableReference.getVariable()->
	addVelocity( aVelocity * aCoefficient );
    }

    void setFlux( RealCref velocity )
    {
      Real aVelocity( velocity );

      setActivity( aVelocity );

      // Increase or decrease variables.
      for( VariableReferenceVectorConstIterator 
	     i( theVariableReferenceVector.begin() ); 
	   i != theFirstZeroVariableReferenceIterator ; ++i )
	{
	  setVariableFlux( *i, aVelocity );
	}

      // skip zero coefficients
      for( VariableReferenceVectorConstIterator 
	     i( theFirstPositiveVariableReferenceIterator );
	   i != theVariableReferenceVector.end() ; ++i )
	{
	  setVariableFlux( *i, aVelocity );
	}
      
    }
    
  };

}

#endif /* __FluxProcess_HPP */







