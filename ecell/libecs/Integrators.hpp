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





#ifndef ___INTEGRATORS_H___
#define ___INTEGRATORS_H___
#include "ecscore/Substance.h"

class Integrator
{

protected:

  Substance& _substance;
  int _counter;   // current step
  //  Float _velocity;  
  Float _numMolCache;
  
  void setQuantity(Float f) {_substance._quantity = f;}
  void setVelocity(Float f) {_substance._velocity = f;}

public:
  Integrator(Substance&);
  virtual ~Integrator() {}


  // how many times to be called during an single
  // integration cycle  
  virtual int numSteps() =0;

  virtual void clear();
  //  virtual Float velocity(Float v)=0;
  virtual void turn() {}
  virtual void transit()=0;
};

class Eular1Integrator  : public Integrator
{

public:
  Eular1Integrator(Substance&); 
  virtual ~Eular1Integrator() {}

  virtual int numSteps() {return 1;}
  //  virtual Float velocity(Float );
  virtual void turn();
  virtual void transit() {/* do nothing */}
};

class RungeKutta4Integrator : public Integrator
{
  typedef void (RungeKutta4Integrator::*TurnFunc)();

  static TurnFunc _turns[4];
  TurnFunc* _turnp;

  void turn0();
  void turn1();
  void turn2();
  void turn3();

  static const Float _one6th;
  Float _k[4];

public:
  RungeKutta4Integrator(Substance&);
  virtual ~RungeKutta4Integrator() {}


  virtual int numSteps() {return 4;}
  virtual void clear();
  //  virtual Float velocity(Float );
  virtual void turn();
  virtual void transit();

};


#endif /* ___INTEGRATORS_H___ */


