
char const Accumulators_C_rcsid[] = "$Id$";
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




#include <cmath>

#include "Accumulators.h"
#include "AccumulatorMaker.h"

void AccumulatorMaker::makeClassList()
{
  NewAccumulatorModule(SimpleAccumulator);
  NewAccumulatorModule(RoundDownAccumulator);
  NewAccumulatorModule(RoundOffAccumulator);
  NewAccumulatorModule(ReserveAccumulator);
  NewAccumulatorModule(MonteCarloAccumulator);
}



void SimpleAccumulator::doit()
{
  quantity() += velocity();
}

void RoundDownAccumulator::doit()
{
  velocity() = floor(velocity());
  quantity() += velocity();
}

void RoundDownAccumulator::update()
{
  quantity() = floor(quantity());
}

void RoundOffAccumulator::doit()
{
  velocity() = rint(velocity());
  quantity() +=  velocity();
}

void RoundOffAccumulator::update()
{
  quantity() = rint(quantity());
}

void ReserveAccumulator::doit()
{
  Float tmp;
  velocity() += _fraction;
  _fraction = MODF(velocity(),&tmp);
  quantity() += tmp;
  velocity() = tmp;
}

Float ReserveAccumulator::save()
{
  return quantity() + _fraction;
}

void ReserveAccumulator::update()
{
  Float tmp;
  _fraction = MODF(quantity(),&tmp);
  quantity() = tmp;
}

void MonteCarloAccumulator::doit()
{
  Float whole;
  Float fraction = MODF(velocity(),&whole);

  if( theRandomNumberGenerator->toss(fraction) )
    {
      whole++;
    }
  velocity() = whole;
  quantity() += whole;
}

void MonteCarloAccumulator::update()
{
  Float whole;
  Float fraction = MODF(quantity(),&whole);
  if( theRandomNumberGenerator->toss(fraction) )
    {
      whole++;
    }
  quantity() = whole;
}
