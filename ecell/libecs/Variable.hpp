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

#ifndef __VARIABLE_HPP
#define __VARIABLE_HPP

#include <utility>

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
     Variable class is used to represent state variables, such as
     amounts of molecular species in a compartment.

  */

  class Variable 
    : 
    public Entity
  {

    DECLARE_VECTOR( RealCptr, RealPtrVector);

    DECLARE_VECTOR( InterpolatorPtr, InterpolatorVector );


  public:

    /** 
	A function type that returns a pointer to Variable.

	Every subclass must have this type of a function which returns
	an instance for the VariableMaker.
    */

    typedef VariablePtr (* AllocatorFuncPtr )();


    Variable();
    virtual ~Variable();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::VARIABLE );
    }


    /**
       Initializes this variable. 
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
	  updateValue( aTime );
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

    void updateValue( RealCref aTime )
    {
      const Real aDeltaT( aTime - theLastTime );

      theTotalVelocity = calculateTotalVelocity();
      const Real aTotalVelocityPerDeltaT( theTotalVelocity * aDeltaT );

      loadValue( getValue() + aTotalVelocityPerDeltaT );

      theLastTime = aTime;
    }


    /**
       Check if the current total velocity doesn't exceed value range of 
       this object.


       @return true -> if estimated value at the next step is
       within the value range, false -> otherwise

       @note Variable class itself doesn't have the value range, thus
       this check always succeed.  Each subclass of Variable should override
       this method if it has the range.
    */

    virtual const bool checkRange( RealCref aStepInterval ) const
    {
      // this class has no range limit, thus this check always success
      return true;
    }

    /**
       This simply set the value of this Variable if getFixed() is false.

       @see getFixed()
    */

    void setValue( RealCref aValue )
    { 
      if( ! getFixed() ) 
	{
	  loadValue( aValue ); 
	}
    }

    virtual void loadValue( RealCref aValue )
    {
      theValue = aValue;
    }

    const Real getValue() const
    { 
      return theValue; 
    }

    virtual const Real saveValue()
    {
      return getValue();
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
       @return true if the Variable is fixed or false if not.
    */

    const bool isFixed() const
    {
      return theFixed;
    }


    // wrappers to expose is/setFixed as PropertySlots 
    void setFixed( IntCref aValue );
    const Int getFixed() const;


    /**
       Returns the concentration of this Variable.

       @return Concentration in M (mol/L).
    */

    const Real getConcentration() const
    {
      return getValue() / ( getSuperSystem()->getVolume() * N_A );
    }

    void registerStepper( StepperPtr aStepperPtr );

    static VariablePtr createInstance() { return new Variable; }

    virtual StringLiteral getClassName() const { return "Variable"; }

  protected:

    void makeSlots();

  protected:

    Real theValue;
    Real theVelocity;

    Real theTotalVelocity;

    Real theLastTime;

    StepperVector theStepperVector;
    // this is a list of indices of Steppers' VariableCache of this Variable.
    RealPtrVector theVelocityVector;

    bool theFixed;

  };



  class PositiveVariable
    :
    public Variable
  {

  public:


    PositiveVariable()
    {
      // do nothing
    }

    virtual ~PositiveVariable()
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
      const Real aPutativeValue( getValue() + aPutativeVelocity );

      if( aPutativeValue >= 0 )
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }

    static VariablePtr createInstance() { return new PositiveVariable; }
      
    virtual StringLiteral getClassName() const { return "PositiveVariable"; }

  };





#if 0 // unmaintained

  /**
     Variable class is used to represent state variables, such as
     amounts of molecular species in a compartment.

  */

  class SRMVariable 
    : 
    public Variable
  {
    //FIXME: for Accumulators:: to be deleted
    friend class Accumulator;
    void accumulatorSetValue( const Real aValue )
    {
      theValue = aValue;
    }


  public: // message slots

    void setAccumulatorClass( StringCref anAccumulatorClassname );
    const String getAccumulatorClass() const;

  public:

    SRMVariable();
    virtual ~SRMVariable();

    static VariablePtr createInstance() { return new SRMVariable; }

    /**
       Initializes this variable. 
       Called at startup.
    */
    virtual void initialize();


    /**
       integrate phase.
       Perform integration by a result calculated by integrator.
    */

    virtual void integrate( RealCref aTime );

    /**
       Set a value with no check. (i.e. isFixed() is ignored.)

       Use setValue() for usual purposes.

       This updates the accumulator immediately.

       @see setValue
    */

    virtual void loadValue( RealCref aValue );


    /**
       Get a value via save() method of the Accumulator.
    */

    virtual const Real saveValue();


    virtual StringLiteral getClassName() const { return "SRMVariable"; }

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


#endif /* __VARIABLE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



