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





#ifndef ___ACCUMULATOR_H___
#define ___ACCUMULATOR_H___

#include "ecscore/Substance.h"



class Accumulator
{
protected:

  Substance* _substance;
  Float& quantity() {return const_cast<Float&>(_substance->_quantity);}
  Float& velocity() {return const_cast<Float&>(_substance->_velocity);}

public:
  Accumulator() : _substance(NULL) {}
  virtual ~Accumulator() {}

//  static Accumulator* instance() {return new Accumulator;}

  void setOwner(Substance* s) {_substance = s;}

  virtual Float save() {return quantity();}
  virtual void update() {}

  virtual void doit() = 0;
  virtual const char* const className() const {return "Accumulator";}
};

class SimpleAccumulator : public Accumulator
{
public:
  SimpleAccumulator() {}
  static Accumulator* instance() {return new SimpleAccumulator;}

  virtual void doit();
  virtual const char* const className() const {return "SimpleAccumulator";}
};

class RoundDownAccumulator : public Accumulator
{
public:
  RoundDownAccumulator() {}
  static Accumulator* instance() {return new RoundDownAccumulator;}

  virtual void update();
  virtual void doit();
  virtual const char* const className() const {return "RoundDownAccumulator";}
};

class RoundOffAccumulator : public Accumulator
{
public:
  RoundOffAccumulator() {}
  static Accumulator* instance() {return new RoundOffAccumulator;}

  virtual void update();
  virtual void doit();
  virtual const char* const className() const {return "RoundOffAccumulator";}
};

class ReserveAccumulator : public Accumulator
{
  Float _fraction;

public:
  ReserveAccumulator() : _fraction(0) {}
  static Accumulator* instance() {return new ReserveAccumulator;}

  virtual Float save();
  virtual void update();

  virtual void doit();
  virtual const char* const className() const {return "ReserveAccumulator";}
};

class MonteCarloAccumulator : public Accumulator
{

public:
  MonteCarloAccumulator() {}
  static Accumulator* instance() {return new MonteCarloAccumulator;}

  virtual void update();
  virtual void doit();
  virtual const char* const className() const {return "MonteCarloAccumulator";}
};


#endif /* ___ACCUMULATOR_H___ */
