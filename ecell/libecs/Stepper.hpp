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





#ifndef ___STEPPER_H___
#define ___STEPPER_H___
#include <stl.h>
#include "include/Defs.h"

class System;
class Integrator;
class Substance;

//class Eular1;

typedef Integrator* IntegratorPtr;
typedef IntegratorPtr (*IntegratorAllocator) (Substance&);

class MasterStepper;

class Stepper
{
  friend class MasterStepper;
  friend class SystemMaker;

protected:

  System* _owner;

  virtual void distributeIntegrator(IntegratorAllocator*);

public:

  Stepper(); 
  virtual ~Stepper() {}

  void setOwner(System* owner) {_owner = owner;}
  System* owner() const {return _owner;}

  virtual void initialize();

  virtual Float deltaT() =0;
  virtual bool isMasterStepper() const {return false;}

  virtual void clear() = 0;
  virtual void react() = 0;
  virtual void transit() = 0;
  virtual void postern() = 0;


//  virtual Stepper* instance(System*){cerr << className() << 
//				       ": instance() not defined" << endl;
//				     return NULL;}

  virtual const char* const className() const  {return "Stepper";}


};

typedef Stepper (*StepperAllocatorFunc)();



class SlaveStepper;

class MasterStepper : public Stepper
{
  friend class SystemMaker;

protected:

  typedef list<SlaveStepper*> SlaveStepperList;
  typedef SlaveStepperList::iterator SlaveStepperListIterator;


  int _pace;
  Float _deltaT;

  IntegratorAllocator _allocator;

  SlaveStepperList _slavesList;

  virtual void distributeIntegrator(IntegratorAllocator);
  void registerSlaves(System*);


public:

  MasterStepper();
//  MasterStepper(System* owner,int pace=1); 

  void setPace(int pace){_pace = pace;}
  void setdeltaT(Float dt){_deltaT = dt;}

  virtual ~MasterStepper() {}

  virtual int numSteps() const {return 1;}
  int pace() const {return _pace;}

  virtual Float deltaT(); 
  virtual void initialize();

  virtual bool isMasterStepper() const {return true;}

  virtual const char* const className() const  {return "MasterStepper";}
};


class StepperLeader : public Stepper
{
  friend class SystemMaker;

  static int _DEFAULT_UPDATE_DEPTH;
  int _updateDepth;
  int _baseClock;

protected:

  typedef multimap<int,MasterStepper*> MasterStepperMap;
  MasterStepperMap _stepperList;

  void setBaseClock(int );

public:

  StepperLeader();
  virtual ~StepperLeader() {}

  void registerMasterStepper(MasterStepper* newstepper);

  static void setDefaultUpdateDepth(int d) {_DEFAULT_UPDATE_DEPTH = d;}
  static int defaultUpdateDepth() {return _DEFAULT_UPDATE_DEPTH;}
  void setUpdateDepth(int d) {_updateDepth = d;}
  int updateDepth() {return _updateDepth;}

  virtual void initialize();

  Float deltaT();
  int baseClock() {return _baseClock;}
  void step();
  virtual void clear();
  virtual void react();
  virtual void transit();
  virtual void postern();

  void update();
  virtual const char* const className() const  {return "StepperLeader";}
};


class SlaveStepper : public Stepper
{
  friend class SystemMaker;
  friend class MasterStepper;

  MasterStepper* _master;
  void masterIs(MasterStepper* m) {_master = m;}

protected:


  virtual void initialize();

public:

  SlaveStepper();
//  SlaveStepper(System* owner);
  virtual ~SlaveStepper() {}

  virtual void clear();
  virtual void react();
  virtual void turn();
  virtual void transit();
  virtual void postern();

  Float deltaT() {return _master->deltaT();}

  static Stepper* instance() {return new SlaveStepper;}
  virtual const char* const className() const  {return "SlaveStepper";}
};


class Eular1Stepper : public MasterStepper
{
  friend class SystemMaker;

protected:

  virtual void initialize();
  static Integrator* newEular1(Substance& substance);

public:

  Eular1Stepper();
//  Eular1Stepper(System* owner);
  virtual ~Eular1Stepper() {}

  static Stepper* instance() {return new Eular1Stepper;}

  virtual int numSteps() {return 1;}
  virtual void clear();
  virtual void react();
//  virtual void check();
  virtual void transit();
  virtual void postern();

  virtual const char* const className() const  {return "Eular1Stepper";}
};


class RungeKutta4Stepper : public MasterStepper
{
  friend class SystemMaker;

protected:

  virtual void initialize();
  static Integrator* newRungeKutta4(Substance& substance);

public:

  RungeKutta4Stepper();
  virtual ~RungeKutta4Stepper() {}

  static Stepper* instance() {return new RungeKutta4Stepper;}

  virtual int numSteps() {return 4;}
  virtual void clear();
  virtual void react();
//  virtual void check();
  virtual void transit();
  virtual void postern();

//  instance(RungeKuttaStepper);
  virtual const char* const className() const  {return "RungeKuttaStepper";}
};

#endif /* ___STEPPER_H___ */
