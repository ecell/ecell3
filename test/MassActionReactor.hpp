#ifndef __MassActionReactor_HPP
#define __MassActionReactor_HPP

#include "libecs/libecs.hpp"

#include "FluxReactor.hpp"

namespace libecs
{

  class MassActionReactor
    :  
    public FluxReactor
  {
  
  public:

    MassActionReactor();
    ~MassActionReactor();

    void setK( RealCref k );
    const Real getK() const { return K; }

    virtual void initialize();
    virtual void differentiate();
    virtual void compute();

    static ReactorPtr createInstance() 
    { 
      return new MassActionReactor; 
    }
   
    StringLiteral getClassName() const { return "MassActionReactor"; }


  protected:
    
    void makeSlots();
            
    Real K;


    PropertySlotPtr S0SuperSystem_Volume;
    PropertySlotVector theReactantConcentrationSlotVector;
  };


}

#endif /* __MassActionReactor_H */

















































