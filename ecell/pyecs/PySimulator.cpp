#include "libecs/FQPN.hpp"
#include "util/Message.hpp"

#include "PySimulator.hpp"


PySimulator::PySimulator()
{
  ; // do nothing
}

void PySimulator::init_type()
{
  behaviors().name("Simulator");
  behaviors().doc("E-CELL Python class");

  add_varargs_method( "makePrimitive", &PySimulator::makePrimitive );
  add_varargs_method( "sendMessage", &PySimulator::sendMessage );
  add_varargs_method( "getMessage", &PySimulator::getMessage );
  add_varargs_method( "step", &PySimulator::step );
}

Object PySimulator::step( const Tuple& args )
{
  cout<<"this is PySimulator::step module."<<endl;
  // Simulator::step();
  return Object();
}

Object PySimulator::makePrimitive( const Tuple& args )
{
  args.verify_length( 3 );
  const string classname( static_cast<Py::String>( args[0] ) );
  const string fqpn( static_cast<Py::String>( args[1] ) );
  const string name( static_cast<Py::String>( args[2] ) );

  Simulator::makePrimitive( classname, FQPN( fqpn ), name );

  return Py::Object();
}
  
Object PySimulator::sendMessage( const Tuple& args )
{
  args.verify_length( 3 );
  const string fqpn( static_cast<Py::String>( args[0] ) );
  const string propertyname( static_cast<Py::String>( args[1] ) );
  const string message( static_cast<Py::String>( args[2] ) );
  
  Simulator::sendMessage( FQPN( fqpn ), Message( propertyname, message ) );

  return Object();
}

Object PySimulator::getMessage( const Tuple& args )
{
  cout<<"this is PySimulator::getMessage module."<<endl;
  return Object();
}



