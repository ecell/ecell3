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

#include "libecs.hpp"
#include "EntityType.hpp"
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

    DECLARE_VECTOR( RealCptr, RealPtrVector);

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
      theVelocity = 0.0; 
    }

    /** 
	integrate phase
    */

    virtual void integrate( RealCref aTime )
    {
      if( isFixed() == false )
	{
	  updateQuantity( aTime );
	}
    }


    const Real calculateTotalVelocity() const
    {
      Real aVelocitySum( 0.0 );
      for( RealPtrVectorConstIterator i( theVelocityVector.begin() ); 
	   i != theVelocityVector.end(); ++i )
	{
	  aVelocitySum += **i;
	}

      return aVelocitySum;
    }

    void updateQuantity( RealCref aTime )
    {
      const Real aDeltaT( aTime - theLastTime );

      theTotalVelocity = calculateTotalVelocity();
      const Real aTotalVelocityPerDeltaT( theTotalVelocity * aDeltaT );

      loadQuantity( getQuantity() + aTotalVelocityPerDeltaT );

      theLastTime = aTime;
    }


    /**
       Check if the current total velocity doesn't exceed value range of 
       this object.


       @return true -> if estimated quantity at the next step is
       within the value range, false -> otherwise

       @note Substance class itself doesn't have the value range, thus
       this check always succeed.  Each subclass of Substance should override
       this method if it has the range.
    */

    virtual const bool checkRange( RealCref aStepInterval ) const
    {
      // this class has no range limit, thus this check always success
      return true;
    }

    /**
       This simply set the quantity of this Substance if getFixed() is false.

       @see getFixed()
    */

    void setQuantity( RealCref aQuantity )
    { 
      if( ! getFixed() ) 
	{
	  loadQuantity( aQuantity ); 
	}
    }

    virtual void loadQuantity( RealCref aQuantity )
    {
      theQuantity = aQuantity;
    }

    const Real getQuantity() const
    { 
      return theQuantity; 
    }

    virtual const Real saveQuantity()
    {
      return getQuantity();
    }



    void setVelocity( RealCref aVelocity )
    {
      theVelocity = aVelocity;
    }

    /**
       @return current velocity value in (number of molecules)/(step)
    */

    const Real getVelocity() const
    { 
      return theVelocity; 
    }

    const Real getTotalVelocity() const
    {
      return theTotalVelocity;
    }

    /**
       @param v velocity in number of molecules to be added.
    */

    void addVelocity( RealCref aVelocity ) 
    {
      theVelocity += aVelocity; 
    }


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
    void setFixed( IntCref aValue );
    const Int getFixed() const;


    /**
       Returns the concentration of this Substance.

       @return Concentration in M (mol/L).
    */

    const Real getConcentration() const
    {
      return getQuantity() / ( getSuperSystem()->getVolume() * N_A );
    }

    void registerStepper( StepperPtr aStepper );

    static SubstancePtr createInstance() { return new Substance; }

    virtual StringLiteral getClassName() const { return "Substance"; }

  protected:

    void makeSlots();

  protected:

    Real theQuantity;
    Real theVelocity;

    Real theTotalVelocity;

    Real theLastTime;

    StepperVector theStepperVector;
    // this is a list of indices of Steppers' SubstanceCache of this Substance.
    RealPtrVector theVelocityVector;

    bool theFixed;

  };



  class PositiveSubstance
    :
    public Substance
  {

  public:


    PositiveSubstance()
    {
      // do nothing
    }

    virtual ~PositiveSubstance()
    {
      // do nothing
    }


    /** 
	integrate phase
    */

    virtual void integrate( RealCref aTime );

    virtual const bool checkRange( RealCref aStepInterval ) const
    {
      const Real aPutativeVelocity( calculateTotalVelocity() * aStepInterval );
      const Real aPutativeQuantity( getQuantity() + aPutativeVelocity );

      if( aPutativeQuantity >= 0 )
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }

    static SubstancePtr createInstance() { return new PositiveSubstance; }
      
    virtual StringLiteral getClassName() const { return "PositiveSubstance"; }

  };





#if 0 // unmaintained

  /**
     Substance class is used to represent state variables, such as
     amounts of molecular species in a compartment.

  */

  class SRMSubstance 
    : 
    public Substance
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

    /**
       Initializes this substance. 
       Called at startup.
    */
    virtual void initialize();


    /**
       integrate phase.
       Perform integration by a result calculated by integrator.
    */

    virtual void integrate( RealCref aTime );

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

    virtual const Real saveQuantity();


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

    Real theFraction;

  };

#endif // 0

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



