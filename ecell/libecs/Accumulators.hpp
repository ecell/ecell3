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


#ifndef ___ACCUMULATOR_H___
#define ___ACCUMULATOR_H___

#include "Substance.h"


class Accumulator
{

public:

  Accumulator() : theSubstance( NULL ) {}
  virtual ~Accumulator() {}

  void setOwner( SubstancePtr substance ) { theSubstance = substance; }

  virtual Float save() { return getQuantity(); }
  virtual void update() {}

  virtual void doit() = 0;

  virtual const char* const className() const { return "Accumulator"; }

protected:

  Float& getQuantity() 
  { return const_cast<Float&>( theSubstance->theQuantity ); }
  Float& getVelocity() 
  { return const_cast<Float&>( theSubstance->theVelocity ); }

protected:

  SubstancePtr theSubstance;

};

class SimpleAccumulator : public Accumulator
{

public:

  SimpleAccumulator() {}
  static AccumulatorPtr instance() { return new SimpleAccumulator; }

  virtual void doit();

  virtual const char* const className() const {return "SimpleAccumulator";}

};

class RoundDownAccumulator : public Accumulator
{

public:

  RoundDownAccumulator() {}
  static AccumulatorPtr instance() { return new RoundDownAccumulator; }

  virtual void update();
  virtual void doit();

  virtual const char* const className() const 
  { return "RoundDownAccumulator"; }

};

class RoundOffAccumulator : public Accumulator
{

public:

  RoundOffAccumulator() {}
  static AccumulatorPtr instance() { return new RoundOffAccumulator; }

  virtual void update();
  virtual void doit();

  virtual const char* const className() const
  { return "RoundOffAccumulator"; }

};

class ReserveAccumulator : public Accumulator
{

public:

  ReserveAccumulator() : theFraction( 0 ) {}
  static AccumulatorPtr instance() { return new ReserveAccumulator; }

  virtual Float save();
  virtual void update();
  virtual void doit();

  virtual const char* const className() const {return "ReserveAccumulator";}

private:

  Float theFraction;

};

class MonteCarloAccumulator : public Accumulator
{

public:

  MonteCarloAccumulator() {}
  static AccumulatorPtr instance() { return new MonteCarloAccumulator; }

  virtual void update();
  virtual void doit();

  virtual const char* const className() const 
  { return "MonteCarloAccumulator"; }

};


#endif /* ___ACCUMULATOR_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
