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

#include "Substance.hpp"

#include "Integrators.hpp"


namespace libecs
{

  ////////////////////////////// Integrator

  Integrator::Integrator( SubstanceRef substance ) 
    :
    theSubstance( substance ),
    theStepCounter( 0 )
  {
    theSubstance.setIntegrator( this );
  }

  void Integrator::clear()
  {
    theStepCounter=0;
    theOriginalQuantity = theSubstance.saveQuantity();
  }

  ////////////////////////////// Euler1Integrator

  Euler1Integrator::Euler1Integrator( SubstanceRef aSubstance ) 
    : 
    Integrator( aSubstance )
  {
    ; // do nothing
  }

  void Euler1Integrator::turn()
  {
    ++theStepCounter;
  }


  ////////////////////////////// RungeKutta4Integrator

  const Real RungeKutta4Integrator::theOne6th( Real( 1.0 ) / Real( 6.0 ) );

  RungeKutta4Integrator::TurnMethod RungeKutta4Integrator::theTurnMethods[4] =
  {
    &RungeKutta4Integrator::turn0,
    &RungeKutta4Integrator::turn1,
    &RungeKutta4Integrator::turn2,
    &RungeKutta4Integrator::turn3
  };


  RungeKutta4Integrator::RungeKutta4Integrator( SubstanceRef aSubstance ) 
    : 
    Integrator( aSubstance )
  {
    ; // do nothing
  }

  void RungeKutta4Integrator::clear()
  {
    Integrator::clear();
    theTurnMethodPtr = &theTurnMethods[0];
  }

  void RungeKutta4Integrator::turn()
  {
    theK[ theStepCounter ] = theSubstance.getVelocity();
    ( this->*( *theTurnMethodPtr ) )();
    ++theTurnMethodPtr;
    ++theStepCounter;
    theSubstance.setVelocity( 0 );
  }

  void RungeKutta4Integrator::turn0()
  {
    theSubstance.loadQuantity( ( theK[0] * .5 ) + theOriginalQuantity );
  }

  void RungeKutta4Integrator::turn1()
  {
    theSubstance.loadQuantity( ( theK[1] * .5 ) + theOriginalQuantity );
  }

  void RungeKutta4Integrator::turn2()
  {
    theSubstance.loadQuantity( theK[2] + theOriginalQuantity );
  }

  void RungeKutta4Integrator::turn3()
  {
    theSubstance.loadQuantity( theOriginalQuantity );
  }

  void RungeKutta4Integrator::integrate()
  {
    //// x(n+1) = x(n) + 1/6 * (k1 + k4 + 2 * (k2 + k3)) + O(h^5)

    Real aResult( theOne6th );
    aResult *= theK[0] + theK[1] + theK[1] + theK[2] + theK[2] + theK[3];

    // an alternative impl.  which is faster?
    //    Real* k( &theK[0] );
    //    Real aResult = *k;
    //    k++;
    //    aResult += *k;
    //    aResult += *k;
    //    k++;
    //    aResult += *k;
    //    aResult += *k;
    //    k++;
    //    aResult += *k;
    //    aResult *= theOne6th;

    theSubstance.setVelocity( aResult );
  }


} // namespace libecs


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
