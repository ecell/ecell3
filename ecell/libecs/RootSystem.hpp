//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
// 		This file is part of Serizawa (E-CELL Core System)
//
//	       written by Kouichi Takahashi  <shafi@sfc.keio.ac.jp>
//
//                              E-CELL Project,
//                          Lab. for Bioinformatics,  
//                             Keio University.
//
//             (see http://www.e-cell.org for details about E-CELL)
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// Serizawa is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// Serizawa is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with Serizawa -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER





#ifndef ___ROOTSYSTEM_H___
#define ___ROOTSYSTEM_H___
#include <stl.h>
#include <fstream.h>
#include "ecscore/System.h"
#include "ecscore/Primitive.h"
//#include "CDS.h"

class Cell;
class System;
class Environment;
class ReactorMaker;
class SubstanceMaker;
class SystemMaker;
class StepperMaker;
class AccumulatorMaker;
class StepperLeader;


class RootSystem : public MetaSystem
{
public:

  class MalformedSystemName : public NotFound
    {
    public:
      MalformedSystemName(const string& method,const string& message) 
	: NotFound(method,message) {}
      const string what() const {return "Malformed system name.";}
    };

private:

//  Cell* _Cell;
//  Environment* _Environment;

  void install();

  StepperLeader& _stepperLeader;

  ReactorMaker& _reactorMaker;
  SubstanceMaker& _substanceMaker;
  SystemMaker& _systemMaker;
  StepperMaker& _stepperMaker;
  AccumulatorMaker& _accumulatorMaker;

public:

  RootSystem();
  ~RootSystem();

  int check();

  System* findSystem(const SystemPath& systempath) throw(NotFound,
							 MalformedSystemName);
  Primitive getPrimitive(const FQPN& fqpn) throw(UnmatchedSystemClass,
						 InvalidPrimitiveType,
						 NotFound);

  virtual void initialize();

  StepperLeader& stepperLeader() const   {return _stepperLeader;}

  ReactorMaker& reactorMaker() const     {return _reactorMaker;}
  SubstanceMaker& substanceMaker() const {return _substanceMaker;}
  SystemMaker& systemMaker() const       {return _systemMaker;}
  StepperMaker& stepperMaker() const     {return _stepperMaker;}
  AccumulatorMaker& accumulatorMaker() const {return _accumulatorMaker;}

  /// only the Application can instantiate RootSystem.
  System* instance() { return NULL; }
  virtual const char* const className() const {return "RootSystem";}
};

extern RootSystem *theRootSystem;

#endif /* ___ROOTSYSTEM_H___ */


