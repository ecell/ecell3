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

#ifndef ___SUBSTANCE_H___
#define ___SUBSTANCE_H___
#include "libecs.hpp"
#include "PrimitiveType.hpp"
#include "Integrators.hpp"
#include "Entity.hpp"


namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  /**
     Substance class is used to represent molecular species.
  */

  class Substance : public Entity
  {
    friend class Integrator;
    friend class Accumulator;

  public: // message slots

    void setAccumulatorClass( UVariableVectorRCPtrCref aMessage );
    const UVariableVectorRCPtr getAccumulatorClass() const;

  public:

    Substance();
    virtual ~Substance();

    static SubstancePtr createInstance() { return new Substance; }

    virtual const PrimitiveType getPrimitiveType() const
    {
      return PrimitiveType( PrimitiveType::SUBSTANCE );
    }


    /**
       @return the number of molecules.
    */
    const Real getQuantity() const
    { 
      return theQuantity; 
    }

    /**
       Fix quantity of this Substance.
    */

    void setFixed()
    { 
      theFixed = true;
    }

    /**
       Unfix quantity of this Substance.
    */

    void clearFixed()
    { 
      theFixed = false;
    }

    /**
       @return true if the Substance is fixed or false if not.
    */
    bool isFixed() const                         
    { 
      return theFixed; 
    }

    /**
       Whether concentration of this substance can be calculated or not.
       It must have a supersystem which have volume other than zero to
       calculate concentration.
       \return true -> if concentration can be obtained. false -> if not.
    */
    // bool haveConcentration() const;

    /**
       Returns a concentration if it have.
       Invalid if haveConcentration() is false.
       @return Concentration in M (mol/L).
    */

    const Real getConcentration() const
    {
      if ( theConcentration < 0 ) 
	{
	  calculateConcentration(); 
	}

      return theConcentration;
    }

    /**
       Initializes this substance. 
       Called at startup.
    */
    void initialize();

    /**
       Clear phase.
       Then call clear() of the integrator.
    */
    void clear()
    { 
      theVelocity = 0; 
      theIntegrator->clear();
    }

    /**
       This is called one or several times in react phase.
       Time of call is determined by the type of the integrator.
    */
    void turn()
    {
      theIntegrator->turn();
    }

    /**
       integrate phase.
       Perform integration by a result calculated by integrator.
    */
    void integrate();

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

    /**
       @param v velocity in number of molecules to be added.
    */
    void addVelocity( RealCref aVelocity ) 
    {
      theVelocity += aVelocity; 
    }

    /**
       Returns activity value of a Substance object.
       The activity is current velocity.
       @see getActivityPerSecond
       @return activity value of Substance in Real.
    */
    Real getActivity();

    /**
       Set a quantity with no check. (i.e. isFixed() is ignored.)
       This updates the accumulator immediately.
       (e.g. loading cell state.) Use setQuantity() for usual purposes.

       @see setQuantity
    */
    void loadQuantity( RealCref aQuantity );

    /**
       This simply set the quantity of this Substance with check of isFixed().
       This updates the accumulator immediately.

       @see isFixed
    */
    void setQuantity( RealCref aQuantity )
    { 
      if( !isFixed() ) 
	{
	  loadQuantity( aQuantity ); 
	}
    }


    /**
       Get a quantity via save() method of the Accumulator.
    */
    const Real saveQuantity();


    virtual StringLiteral getClassName() const { return "Substance"; }

    /**
       set a class name string of user default accumulator
    */
    static void setUserDefaultAccumulatorName( StringCref name ) 
    { 
      USER_DEFAULT_ACCUMULATOR_NAME = name; 
    }

    /**
       a class name string of user default accumulator
    */
    static StringCref userDefaultAccumulatorName() 
    { 
      return USER_DEFAULT_ACCUMULATOR_NAME; 
    }

  protected:

    void setAccumulator( StringCref anAccumulatorClassname );
    void setAccumulator( AccumulatorPtr anAccumulator );
    void setIntegrator( IntegratorPtr anIntegrator ) 
    { 
      theIntegrator = anIntegrator; 
    }

    void makeSlots();

  private:

    void calculateConcentration() const;
    void mySetQuantity( RealCref aQuantity ) 
    { 
      theQuantity = aQuantity; 
      theConcentration = -1; 
    }

  public:

    /**
       A class name string of system default accumulator
    */
    const static String SYSTEM_DEFAULT_ACCUMULATOR_NAME;

  private:

    static String USER_DEFAULT_ACCUMULATOR_NAME;

    AccumulatorPtr theAccumulator;
    IntegratorPtr theIntegrator;

    Real theQuantity;
    Real theFraction;
    Real theVelocity;

    bool theFixed;

    mutable Real theConcentration;
  };

  /** @} */ //end of libecs_module 

} // namespace libecs

#endif /* ___SUBSTANCE_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



