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

#ifndef ___SUBSTANCE_H___
#define ___SUBSTANCE_H___

#include <iostream>
#include "libecs.hpp"
#include "EntityType.hpp"
#include "Integrators.hpp"
#include "Entity.hpp"
#include "System.hpp"


namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


  /**
     Substance class is used to represent state variables, such as
     amounts of molecular species in a compartment.

  */

  class Substance 
    : 
    public Entity
  {

  public:

    /** 
	A function type that returns a pointer to Substance.

	Every subclass must have this type of a function which returns
	an instance for the SubstanceMaker.
    */

    typedef SubstancePtr (* AllocatorFuncPtr )();


    Substance();
    virtual ~Substance();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::SUBSTANCE );
    }


    /**
       Initializes this substance. 
    */
    virtual void initialize();


    /**
       Clear phase.
    */

    virtual void clear() 
    {
      ; // do nothing
    }

    virtual void turn()
    {
      ; // do nothing
    }

    virtual void integrate()
    {
      ; // do nothing
    }

    /**
       @return the quantity of this substance.
    */

    virtual const Real getQuantity() const = 0;


    /**
       @param aQuantity the quantity of this substance.
    */

    virtual void setQuantity( RealCref aQuantity ) = 0;


    /**
       Returns the concentration of this Substance.

       @return Concentration in M (mol/L).
    */

    const Real getConcentration() const
    {
      return getQuantity() / ( getSuperSystem()->getVolume() * N_A );
    }


    virtual void setVelocity( RealCref aVelocity ) = 0;

    /**
       @return current velocity value in (number of molecules)/(step)
    */

    virtual const Real getVelocity() const = 0;

    /**
       @param v velocity in number of molecules to be added.
    */

    virtual void addVelocity( RealCref aVelocity ) = 0;

    /**
       Returns activity value of a Substance object.
       The activity is current velocity.
       @see getActivityPerSecond
       @return activity value of Substance in Real.
    */

    virtual const Real getActivity();

    virtual StringLiteral getClassName() const { return "Substance"; }

  protected:

    void makeSlots();

  };



  class PlainSubstance 
    : 
    public Substance
  {

  public:

    PlainSubstance();
    virtual ~PlainSubstance();

    static SubstancePtr createInstance() { return new PlainSubstance; }

    /**
       @return the number of molecules.
    */
    virtual const Real getQuantity() const
    { 
      return theQuantity; 
    }

    /**
       Initializes this substance. 
       Called at startup.
    */
    virtual void initialize();


    /**
       Clear phase.
    */

    virtual void clear()
    { 
      theVelocity = 0.0; 
    }

    virtual void turn()
    {
      //      theVelocity = 0.0;
    }

    virtual void integrate()
    {
      // PlainSubstance doesn't currently support integration
      //      theQuantity += theVelocity;
    }

    /**
       Set a quantity with no check. (i.e. isFixed() is ignored.)

       Use setQuantity() for usual purposes.

       This updates the accumulator immediately.

       @see setQuantity
    */

    virtual void loadQuantity( RealCref aQuantity );

    /**
       This simply set the quantity of this Substance if getFixed() is false.
       This updates the accumulator immediately.

       @see getFixed()
    */

    void setQuantity( RealCref aQuantity )
    { 
      if( ! getFixed() ) 
	{
	  loadQuantity( aQuantity ); 
	}
    }

    virtual void setVelocity( RealCref aVelocity )
    {
      theVelocity = aVelocity;
    }

    /**
       @return current velocity value in (number of molecules)/(step)
    */

    virtual const Real getVelocity() const
    { 
      return theVelocity; 
    }

    /**
       @param v velocity in number of molecules to be added.
    */

    virtual void addVelocity( RealCref aVelocity ) 
    {
      theVelocity += aVelocity; 
    }


    /**
       Get a quantity via save() method of the Accumulator.
    */

    const Real saveQuantity();

    void setFixed( const bool aValue )
    {
      theFixed = aValue;
    }

    /**
       @return true if the Substance is fixed or false if not.
    */

    const bool isFixed() const
    {
      return theFixed;
    }



    // wrappers to expose is/setFixed as PropertySlots
    void setFixed( RealCref aValue );
    const Real getFixed() const;


    virtual StringLiteral getClassName() const { return "PlainSubstance"; }

  protected:

    void makeSlots();

  protected:

    Real theQuantity;
    Real theVelocity;

    bool theFixed;

  };


  /**
     Substance class is used to represent state variables, such as
     amounts of molecular species in a compartment.

  */

  class SRMSubstance 
    : 
    public PlainSubstance
  {
    //FIXME: for Accumulators:: to be deleted
    friend class Accumulator;
    void accumulatorSetQuantity( const Real aQuantity )
    {
      theQuantity = aQuantity;
    }


  public: // message slots

    void setAccumulatorClass( StringCref anAccumulatorClassname );
    const String getAccumulatorClass() const;

  public:

    SRMSubstance();
    virtual ~SRMSubstance();

    static SubstancePtr createInstance() { return new SRMSubstance; }

    void setIntegrator( IntegratorPtr anIntegrator ) 
    { 
      delete theIntegrator;
      theIntegrator = anIntegrator; 
    }


    /**
       Initializes this substance. 
       Called at startup.
    */
    virtual void initialize();

    /**
       Clear phase.
       Then call clear() of the integrator.
    */
    virtual void clear()
    { 
      PlainSubstance::clear();
      theIntegrator->clear();
    }

    /**
       This is called one or several times in react phase.
    */
    void turn()
    {
      theIntegrator->turn();
    }

    /**
       integrate phase.
       Perform integration by a result calculated by integrator.
    */
    virtual void integrate();

    /**
       Set a quantity with no check. (i.e. isFixed() is ignored.)

       Use setQuantity() for usual purposes.

       This updates the accumulator immediately.

       @see setQuantity
    */

    virtual void loadQuantity( RealCref aQuantity );


    /**
       Get a quantity via save() method of the Accumulator.
    */

    const Real saveQuantity();


    virtual StringLiteral getClassName() const { return "SRMSubstance"; }

  protected:

    void setAccumulator( AccumulatorPtr anAccumulator );


    virtual void makeSlots();

  public:

    /**
       A class name string of system default accumulator
    */
    const static String SYSTEM_DEFAULT_ACCUMULATOR_NAME;

  protected:


    AccumulatorPtr theAccumulator;
    IntegratorPtr theIntegrator;

    Real theFraction;

  };

  /*@}*/

} // namespace libecs

#endif /* ___SUBSTANCE_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



