//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2000 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-CELL is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-CELL is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-CELL -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org> at
// E-CELL Project, Lab. for Bioinformatics, Keio University.
//


#ifndef ___INTEGRATORS_H___
#define ___INTEGRATORS_H___
#include "Substance.h"

class Integrator
{

public:

  Integrator(SubstanceRef);
  virtual ~Integrator() {}

  /**
     how many times to be called during an single integration cycle  
  */
  virtual int getNumberOfSteps() = 0;

  virtual void clear();
  //  virtual Float velocity(Float v) = 0;
  virtual void turn() {}
  virtual void transit() = 0;

protected:

  void setQuantity( Float quantity ) { theSubstance.theQuantity = quantity; }
  void setVelocity( Float velocity ) { theSubstance.theVelocity = velocity; }

protected:

  SubstanceRef theSubstance;
  int          theStepCounter;
  Float        theNumberOfMoleculesCache;

};

class Eular1Integrator  : public Integrator
{

public:

  Eular1Integrator( SubstanceRef ); 
  virtual ~Eular1Integrator() {}

  virtual int getNumberOfSteps() { return 1; }
  //  virtual Float velocity(Float );
  virtual void turn();
  virtual void transit() {}
};

class RungeKutta4Integrator : public Integrator
{

  typedef void (RungeKutta4Integrator::*TurnFunc_)();
  DECLARE_TYPE(TurnFunc_,TurnFunc);

public:

  RungeKutta4Integrator( SubstanceRef );
  virtual ~RungeKutta4Integrator() {}


  virtual int getNumberOfSteps() { return 4; }
  virtual void clear();
  //  virtual Float velocity( Float );
  virtual void turn();
  virtual void transit();

private:

  void turn0();
  void turn1();
  void turn2();
  void turn3();

private:

  static TurnFunc theTurnFuncs[4];
  TurnFuncPtr theTurnFuncPtr;

  static const Float theOne6th;
  Float theK[4];

};


#endif /* ___INTEGRATORS_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

