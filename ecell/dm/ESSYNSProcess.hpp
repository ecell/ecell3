#ifndef __ESSYNSPROCESS_HPP
#define __ESSYNSPROCESS_HPP

#include <vector>

#include "libecs.hpp"
#include "Process.hpp"

USE_LIBECS;


LIBECS_DM_CLASS( ESSYNSProcess, Process )
{
  
 public:

  DECLARE_VECTOR( libecs::Real, RealVector );


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
    
  virtual void fire()
    {
      ;
    }
    
  virtual const std::vector<RealVector>& getESSYNSMatrix() = 0;

  virtual GET_METHOD( Integer, SystemSize ) = 0;
    
 protected:

};

LIBECS_DM_INIT_STATIC( ESSYNSProcess, Process );


#endif /* __ESSYNSPROCESS_HPP */
