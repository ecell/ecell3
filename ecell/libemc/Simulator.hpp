#ifndef ___SIMULATOR_H___
#define ___SIMULATOR_H___

#include "libecs.hpp"
#include "Message.hpp"

#include "SimulatorImplementation.hpp"


class Simulator
{

public:

  Simulator();
  virtual ~Simulator() {}

  void makePrimitive( StringCref classname, 
		      FQPICref fqpi,
		      StringCref name )
  {
    theSimulatorImplementation->makePrimitive( classname, fqpi, name );
  }

  void sendMessage( FQPICref fqpi, 
		    MessageCref message )
  {
    theSimulatorImplementation->sendMessage( fqpi, message );
  }


  Message getMessage( FQPICref fqpi, 
		      StringCref propertyName )
  {
    return theSimulatorImplementation->getMessage( fqpi, propertyName ); 
  }

  void step()
  {
    theSimulatorImplementation->step();
  }

  void initialize()
  {
    theSimulatorImplementation->initialize();
  }


private:

  SimulatorImplementation* theSimulatorImplementation;

};

#endif   /* ___SIMULATOR_H___ */





