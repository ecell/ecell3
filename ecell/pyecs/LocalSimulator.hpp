#ifndef ___LOCAL_SIMULATOR_H___
#define ___LOCAL_SIMULATOR_H___

#include "Simulator.h"

class LocalSimulator : public Simulator
{
public:
  LocalSimulator();
  ~LocalSimulator() {};
  void run( const double t );
  void step();
  void makeSubstance( const string id, const string fqen, const string const name );
  void makeReactor( const string id, const string fqen, const string const name );
  void makeSystem( const string id, const string fqen, const string const name );
};   //end of class LocalSimulator

#endif   /* ___LOCAL_SIMULATOR_H___ */












