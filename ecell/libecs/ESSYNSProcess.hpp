#ifndef __ESSYNSPROCESS_HPP
#define __ESSYNSPROCESS_HPP

#include <vector>

#include "libecs.hpp"
#include "System.hpp"
#include "Variable.hpp"
#include "Stepper.hpp"
#include "Process.hpp"
#include "Util.hpp"
#include "PropertyInterface.hpp"

using namespace std;

namespace libecs
{

  LIBECS_DM_CLASS( ESSYNSProcess, Process )
  {
  
  public:

    LIBECS_DM_OBJECT_ABSTRACT( ESSYNSProcess )
    {
      INHERIT_PROPERTIES( Process );  
    }
  
    ESSYNSProcess()
      {
	;
      }

    virtual ~ESSYNSProcess()
      {
	;
      }

    virtual void initialize()
      {
	Process::initialize();
      }
    
    virtual void process()
      {
	;
      }
    
    virtual const vector<RealVector>& getESSYNSMatrix() = 0;

    virtual Int getSystemSize() = 0;
    
  protected:

  };

}

#endif /* __ESSYNSPROCESS_HPP */
