#include "Simulator.hpp"

Simulator::Simulator()
{
  theSimulatorImplementation = new LocalSimulatorImplementation();
}

void Simulator::makePrimitive( StringCref classname, FQPNCref fqpn, StringCref name )
{
  theSimulatorImplementation->makePrimitive( classname, fqpn, name );
}

void Simulator::sendMessage( FQPNCref fqpn, Message message )
{
  theSimulatorImplementation->sendMessage( fqpn, message );
}

/*
Message Simulator::getMessage( FQPNCref fqpn, StringCref propertyName )
{
  theSimulatorImplementation->sendMessage( fqpn, message );
}
*/

void Simulator::step()
{
  theSimulatorImplementation->step();
}
