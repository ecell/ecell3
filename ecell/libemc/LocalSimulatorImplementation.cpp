#include "LocalSimulatorImplementation.hpp"

RootSystem* theRootSystem = new RootSystem();

LocalSimulatorImplementation::LocalSimulatorImplementation()
{
}

void LocalSimulatorImplementation::makePrimitive( StringCref classname,
					     FQPNCref fqpn, 
					     StringCref name )
{
}

void LocalSimulatorImplementation::sendMessage( FQPNCref fqpn, 
						MessageCref message)
{
}

/*
Message LocalSimulatorImplementation::getMessage( StringCref fqpn,
StringCref propertyName )
{
}
*/

void LocalSimulatorImplementation::step()
{
  //  theRootSystem->getStepperLeader().step();  
}



