//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
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


#include "Util.hpp"
#include "Accumulators.hpp"


namespace libecs
{

  void SimpleAccumulator::accumulate()
  {
    setQuantity( theSubstance->getQuantity() + 
		 theSubstance->getVelocity() );
  }

  void RoundDownAccumulator::accumulate()
  {
    theSubstance->setVelocity( floor( theSubstance->getVelocity() ) );
    setQuantity( theSubstance->getQuantity() + 
		 theSubstance->getVelocity() );
  }

  void RoundDownAccumulator::update()
  {
    setQuantity( floor( theSubstance->getQuantity() ) );
  }

  void RoundOffAccumulator::accumulate()
  {
    theSubstance->setVelocity( rint( theSubstance->getVelocity() ) );
    setQuantity( theSubstance->getQuantity() + 
		 theSubstance->getVelocity() );
  }

  void RoundOffAccumulator::update()
  {
    setQuantity( rint( theSubstance->getQuantity() ) );
  }

  void ReserveAccumulator::accumulate()
  {
    Real aVelocity( theSubstance->getVelocity() );
    aVelocity += theFraction;
    Real tmp;
    theFraction = modf( aVelocity, &tmp );

    theSubstance->setVelocity( tmp );
    setQuantity( theSubstance->getQuantity() + tmp );
  }

  Real ReserveAccumulator::save()
  {
    return theSubstance->getQuantity() + theFraction;
  }

  void ReserveAccumulator::update()
  {
    Real tmp;
    theFraction = modf( theSubstance->getQuantity(), &tmp );
    setQuantity( tmp );
  }

  void MonteCarloAccumulator::accumulate()
  {
    Real aWhole;
    Real aFraction = modf( theSubstance->getVelocity(), &aWhole );

    if( theRandomNumberGenerator->toss( aFraction ) )
      {
	++aWhole;
      }

    theSubstance->setVelocity( aWhole );
    setQuantity( theSubstance->getQuantity() + aWhole );
  }

  void MonteCarloAccumulator::update()
  {
    Real aWhole;
    Real aFraction = modf( theSubstance->getQuantity(), &aWhole );
    if( theRandomNumberGenerator->toss( aFraction ) )
      {
	++aWhole;
      }

    setQuantity( aWhole );
  }


} // namespace libecs
