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


#ifndef ___INTEGRATORS_H___
#define ___INTEGRATORS_H___
#include "libecs.hpp"


namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  DECLARE_CLASS( Euler1Integrator );
  DECLARE_CLASS( RungeKutta4Integrator );


  class Integrator
  {

  public:

    Integrator( SubstanceRef aSubstance );
    virtual ~Integrator() {}

    /**
       how many react->turn steps does needed in a single integration cycle  
    */

    virtual int getNumberOfSteps() = 0;

    virtual void clear();
    virtual void turn() {}
    virtual void integrate() = 0;


  protected:

    SubstanceRef theSubstance;
    int          theStepCounter;
    Real         theOriginalQuantity;

  };

  class Euler1Integrator  : public Integrator
  {

  public:

    Euler1Integrator( SubstanceRef aSubstance ); 
    virtual ~Euler1Integrator() {}

    virtual int getNumberOfSteps() { return 1; }
    virtual void turn();
    virtual void integrate() {}
  };

  class RungeKutta4Integrator : public Integrator
  {

    typedef void ( RungeKutta4Integrator::*TurnMethod_ )();
    DECLARE_TYPE( TurnMethod_, TurnMethod );

  public:

    RungeKutta4Integrator( SubstanceRef aSubstance );
    virtual ~RungeKutta4Integrator() {}


    virtual int getNumberOfSteps() { return 4; }
    virtual void clear();
    virtual void turn();
    virtual void integrate();

  private:

    void turn0();
    void turn1();
    void turn2();
    void turn3();

  private:

    TurnMethodPtr     theTurnMethodPtr;
    Real              theK[4];


    // turn method table.
    static TurnMethod theTurnMethods[4];

    // constant number: 1/6;  for internal use
    static const Real theOne6th;

  };

  /** @} */ //end of libecs_module 

} // namespace libecs

#endif /* ___INTEGRATORS_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/

