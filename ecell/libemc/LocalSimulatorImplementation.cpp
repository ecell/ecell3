#include "Message.hpp"
#include "FQPI.hpp"

#include "LocalSimulatorImplementation.hpp"

LocalSimulatorImplementation::LocalSimulatorImplementation()
{
  ; // do nothing
}

void LocalSimulatorImplementation::makePrimitive( StringCref classname,
						  FQPICref fqpi, 
						  StringCref name )
{
  cerr << classname << endl;
  cerr << fqpi.getString() << endl;
  cerr << name << endl;

}

void LocalSimulatorImplementation::sendMessage( FQPICref fqpi, 
						MessageCref message)
{
  cerr << fqpi.getString() << endl;
  cerr << message.getKeyword() << endl;
}


Message LocalSimulatorImplementation::getMessage( FQPICref fqpi,
						  StringCref propertyName )
{
  return Message( propertyName, "empty" );
}


void LocalSimulatorImplementation::step()
{
  //  theRootSystem->getStepperLeader().step();  
}



