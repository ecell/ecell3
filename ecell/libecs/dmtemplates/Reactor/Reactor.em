#ifndef __@(CLASSNAME)_CPP
#define __@(CLASSNAME)_CPP

#include <iostream>

#include "libecs.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Substance.hpp"
#include "Reactor.hpp"
#include "Util.hpp"
#include "PropertySlotMaker.hpp"

#include "@(BASECLASS).hpp"

@{fileincludes()}

namespace libecs
{

  class @(CLASSNAME)
    :  
    public @(BASECLASS)
  {
  
  public:

    @(CLASSNAME)();
    ~@(CLASSNAME)();

@{propertymethods()}

@(PUBLIC_AUX)
    
    virtual void react();
    virtual void initialize();
    
    static ReactorPtr createInstance() 
    { 
      return new @(CLASSNAME);
    }
   
    StringLiteral getClassName() const { return "@(CLASSNAME)"; }


  protected:
    
    void makeSlots();

@{propertyvariabledecls()}
@{reactantpropertyslotvariabledecls()}

@(PROTECTED_AUX)

  private:

@(PRIVATE_AUX)

  };

}

using namespace libecs;

extern "C"
{
  Reactor::AllocatorFuncPtr CreateObject =
  &@(CLASSNAME)::createInstance;
}  

@(CLASSNAME)::@(CLASSNAME)()
{
  makeSlots();
  @{propertyvariableinit()}
}

@(CLASSNAME)::~@(CLASSNAME)()
{
}

void @(CLASSNAME)::makeSlots()
{
  @{createpropertyslots()}
}


void @(CLASSNAME)::initialize()
{
  @(BASECLASS)::initialize();

  @{getpropertyslotofreactant()}
  @{allreactantslotsinit()}

  @{methodDefs('initialize')}

}

void @(CLASSNAME)::react()
{
@{methodDefs('react')}
}

#endif
