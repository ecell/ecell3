#ifndef __@(CLASSNAME)_CPP
#define __@(CLASSNAME)_CPP

#include <iostream>

#include "libecs.hpp"
#include "System.hpp"
#include "Stepper.hpp"
#include "Variable.hpp"
#include "Process.hpp"
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
    
    virtual void process();
    virtual void initialize();
    
    static ProcessPtr createInstance() 
    { 
      return new @(CLASSNAME);
    }
   
    StringLiteral getClassName() const { return "@(CLASSNAME)"; }


  protected:
    
    void makeSlots();

@{propertyvariabledecls()}
@{variablepropertyslotvariabledecls()}

@(PROTECTED_AUX)

  private:

@(PRIVATE_AUX)

  };

}

using namespace libecs;

extern "C"
{
  Process::AllocatorFuncPtr CreateObject =
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

  @{getpropertyslotofvariable()}
  @{allvariableslotsinit()}

  @{methodDefs('initialize')}

}

void @(CLASSNAME)::process()
{
@{methodDefs('process')}
}

#endif
