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

#include "Substance.hpp"

namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  DECLARE_CLASS( SimpleAccumulator );
  DECLARE_CLASS( RoundDownAccumulator );
  DECLARE_CLASS( RoundOffAccumulator );
  DECLARE_CLASS( ReserveAccumulator );
  DECLARE_CLASS( MonteCarloAccumulator );


  class Accumulator
  {

  public:

    Accumulator() 
      : 
      theSubstance( NULLPTR ) 
    {
      ; // do nothing
    }

    virtual ~Accumulator() 
    {
      ; // do nothing
    }

    void setOwner( SubstancePtr substance ) 
    { 
      theSubstance = substance; 
    }

    virtual Real save() 
    { 
      return getQuantity(); 
    }

    virtual void update() 
    {
      ; // do nothing
    }

    virtual void accumulate() = 0;

    virtual StringLiteral getClassName() const { return "Accumulator"; }

  protected:

    Real& getQuantity() 
    { 
      return const_cast<Real&>( theSubstance->theQuantity ); 
    }

    Real& getVelocity() 
    { 
      return const_cast<Real&>( theSubstance->theVelocity ); 
    }

  protected:

    SubstancePtr theSubstance;

  };

  class SimpleAccumulator : public Accumulator
  {

  public:

    SimpleAccumulator() 
    {
      ; // do nothing
    }

    static AccumulatorPtr createInstance() { return new SimpleAccumulator; }

    virtual void accumulate();

    virtual StringLiteral getClassName() const {return "SimpleAccumulator";}

  };

  class RoundDownAccumulator : public Accumulator
  {

  public:

    RoundDownAccumulator() 
    {
      ; // do nothing
    }

    static AccumulatorPtr createInstance() { return new RoundDownAccumulator; }

    virtual void update();
    virtual void accumulate();

    virtual StringLiteral getClassName() const 
    { return "RoundDownAccumulator"; }

  };

  class RoundOffAccumulator : public Accumulator
  {

  public:

    RoundOffAccumulator() 
    {
      ; // do nothing
    }

    static AccumulatorPtr createInstance() { return new RoundOffAccumulator; }

    virtual void update();
    virtual void accumulate();

    virtual StringLiteral getClassName() const
    { return "RoundOffAccumulator"; }

  };

  class ReserveAccumulator : public Accumulator
  {

  public:

    ReserveAccumulator() 
      : 
      theFraction( 0 ) 
    {
      ; // do nothing
    }

    static AccumulatorPtr createInstance() { return new ReserveAccumulator; }

    virtual Real save();
    virtual void update();
    virtual void accumulate();

    virtual StringLiteral getClassName() const {return "ReserveAccumulator";}

  private:

    Real theFraction;

  };

  class MonteCarloAccumulator : public Accumulator
  {

  public:

    MonteCarloAccumulator() 
    {
      ; // do nothing
    }

    static AccumulatorPtr createInstance() 
    { return new MonteCarloAccumulator; }

    virtual void update();
    virtual void accumulate();

    virtual StringLiteral getClassName() const 
    { return "MonteCarloAccumulator"; }

  };

  /** @} */ //end of libecs_module 

} // namespace libecs


#endif /* ___ACCUMULATOR_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
