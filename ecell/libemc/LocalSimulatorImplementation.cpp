#include "util/Message.hpp"
#include "libecs/FQPI.hpp"

#include "LocalSimulatorImplementation.hpp"

RootSystem* theRootSystem = new RootSystem();

LocalSimulatorImplementation::LocalSimulatorImplementation()
{
}

void LocalSimulatorImplementation::makePrimitive( StringCref classname,
						  FQPICref fqpn, 
						  StringCref name )
{
  cerr << classname << endl;
  cerr << fqpn.getString() << endl;
  cerr << name << endl;

}

void LocalSimulatorImplementation::sendMessage( FQPICref fqpn, 
						MessageCref message)
{
  cerr << message.getKeyword() << endl;
}


Message LocalSimulatorImplementation::getMessage( FQPICref fqpn,
						  StringCref propertyName )
{
  return Message( propertyName, "empty" );
}


void LocalSimulatorImplementation::step()
{
  //  theRootSystem->getStepperLeader().step();  
}



