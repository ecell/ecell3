
char const Integrators_C_rcsid[] = "$Id$";
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




#include "Integrators.h"


////////////////////////////// Integrator

Integrator::Integrator(Substance& substance) 
  :_substance(substance)/*,_numSteps(1)*/,_counter(0)
{
  _substance.setIntegrator(this);
}

void Integrator::clear()
{
  _counter=0;
  _numMolCache = _substance.quantity();
}

////////////////////////////// Eular1Integrator


Eular1Integrator::Eular1Integrator(Substance& substance) : Integrator(substance)
{
  //  _numSteps = 1;

}

void Eular1Integrator::turn()
{
  _counter++;
}


////////////////////////////// RungeKutta4Integrator

const Float RungeKutta4Integrator::_one6th = (Float)1.0 / (Float)6.0;
RungeKutta4Integrator::TurnFunc RungeKutta4Integrator::_turns[4] =
{&RungeKutta4Integrator::turn0,
 &RungeKutta4Integrator::turn1,
 &RungeKutta4Integrator::turn2,
 &RungeKutta4Integrator::turn3
};


RungeKutta4Integrator::RungeKutta4Integrator(Substance& substance) : Integrator(substance)
{
  //  _numSteps = 4;
}

void RungeKutta4Integrator::clear()
{
  Integrator::clear();
  _turnp = _turns;
}

void RungeKutta4Integrator::turn()
{
  _k[_counter] = _substance.velocity();
  (this->*(*_turnp))();
  _turnp++;
  _counter++;
  setVelocity(0);
}

void RungeKutta4Integrator::turn0()
{
  setQuantity((_k[0]*.5) + _numMolCache);
}

void RungeKutta4Integrator::turn1()
{
  setQuantity((_k[1]*.5) + _numMolCache);
}

void RungeKutta4Integrator::turn2()
{
  setQuantity((_k[2]) + _numMolCache);
}

void RungeKutta4Integrator::turn3()
{
  setQuantity(_numMolCache);
}


void RungeKutta4Integrator::transit()
{
  //// x(n+1) = x(n) + 1/6 * (k1 + k4 + 2 * (k2 + k3)) + O(h^5)

  Float* k = _k;

  // FIXME: prefetching here makes this faster on alpha?

  //                      k1  + (2 * k2)  + (2 * k3)  + k4
#ifdef __GNUC__
  // This probably be faster on gcc, but miscompiles on the other compilers
  // which doesn't use LR parsing algorithm.
  Float result = *k++ + *k + *k++ + *k + *k++ + *k;
#else
  // This will rather be safer on any flavor of compillers.
  Float result = *k++;
  result += *k;
  result += *k++;
  result += *k;
  result += *k++;
  result += *k;
#endif /* __GNUC__ */

  result *= _one6th;

  setVelocity(result);
}

