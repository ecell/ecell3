#include "Simulator.hpp"

Simulator::Simulator()
{
  theSimulatorImplementation = new SimulatorImplementation();
}

//void Simulator::makePrimitive( StringCref classname, FQPNCref fqpn, StringCref name )
void Simulator::makePrimitive()
{
  //  theSimulatorImplementation->makePrimitive( classname, fqpn, name );
  theSimulatorImplementation->makePrimitive();
}

//void Simulator::sendMessage( FQPNCref fqpn, Message message )
void Simulator::sendMessage()
{
  //  theSimulatorImplementation->sendMessage( fqpn, message );
  theSimulatorImplementation->sendMessage();
}


//Message Simulator::getMessage( FQPNCref fqpn, StringCref propertyName )
void Simulator::getMessage()
{
  //  theSimulatorImplementation->sendMessage( fqpn, message );
  theSimulatorImplementation->sendMessage();
}

void Simulator::step()
{
  theSimulatorImplementation->step();
}
