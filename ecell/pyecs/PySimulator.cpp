#include "libecs/libecs.hpp"
#include "libecs/FQPI.hpp"
#include "util/Message.hpp"

#include "PySimulator.hpp"

#define ECS_TRY try {

#define ECS_CATCH\
    }\
  catch( ::ExceptionCref e )\
    {\
      throw Py::Exception( e.message() );\
    }\
  catch( ... ) \
    {\
      throw Py::SystemError( "E-CELL internal error." );\
    }

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
  ECS_TRY;

  cout<<"this is PySimulator::step module."<<endl;
  // Simulator::step();
  return Object();

  ECS_CATCH;
}

Object PySimulator::makePrimitive( const Tuple& args )
{
  ECS_TRY;

  args.verify_length( 3 );
  const string classname( static_cast<Py::String>( args[0] ) );
  const FQPI fqpi( static_cast<Py::String>( args[1] ) );
  const string name( static_cast<Py::String>( args[2] ) );

  Simulator::makePrimitive( classname, fqpi, name );

  return Py::Object();

  ECS_CATCH;
}
  
Object PySimulator::sendMessage( const Tuple& args )
{
  ECS_TRY;

  args.verify_length( 3 );
  const string fqpi( static_cast<Py::String>( args[0] ) );
  const string propertyname( static_cast<Py::String>( args[1] ) );
  const string message( static_cast<Py::String>( args[2] ) );
  
  Simulator::sendMessage( FQPI( fqpi ), Message( propertyname, message ) );

  return Object();

  ECS_CATCH;
}

Object PySimulator::getMessage( const Tuple& args )
{
  ECS_TRY;

  args.verify_length( 2 );
  
  const FQPI aFqpi( static_cast<Py::String>( args[0] ) );
  const string aPropertyName( static_cast<Py::String>( args[1] ) );

  Message aMessage( Simulator::getMessage( aFqpi, aPropertyName ) );
  Tuple aTuple( 2 );
  aTuple[0] = static_cast<Py::String>( aMessage.getKeyword() );
  aTuple[1] = static_cast<Py::String>( aMessage.getBody() );

  return aTuple;

  ECS_CATCH;
}



