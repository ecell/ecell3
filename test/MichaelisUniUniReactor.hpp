#ifndef __MichaelisUniUniReactor_HPP
#define __MichaelisUniUniReactor_HPP

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

    void setKmS( RealCref kms );
    const Real getKmS() const { return KmS; }
    void setKcF( RealCref kcf );
    const Real getKcF() const { return KcF; }

    virtual void initialize();
    virtual void differentiate();
    virtual void compute();

    static ReactorPtr createInstance() 
    { 
      return new MichaelisUniUniReactor; 
    }
   
    StringLiteral getClassName() const { return "MichaelisUniUniReactor"; }


  protected:
    
    void makeSlots();
            
    Real KmS;
    Real KcF;

    PropertySlotPtr C0_Quantity;
    PropertySlotPtr S0_Concentration;

  };


}

#endif /* __MichaelisUniUniReactor_H */

















































