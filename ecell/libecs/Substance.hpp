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
#include "Entity.hpp"


namespace libecs
{


  /**
     Substance class is used to represent molecular species.
  */

  class Substance : public Entity
  {
    friend class Integrator;
    friend class Accumulator;

  public: // message slots

    void setQuantity( MessageCref message );
    void setAccumulatorClass( MessageCref message );
    const Message getQuantity( StringCref keyword );
    const Message getConcentration( StringCref keyword );
    const Message getAccumulatorClass( StringCref keyword );

  public:

    Substance();
    virtual ~Substance();

    static SubstancePtr instance() { return new Substance; }

    virtual const PrimitiveType getPrimitiveType() const
    {
      return PrimitiveType( PrimitiveType::SUBSTANCE );
    }


    /**
       @return the number of molecules.
    */
    Real getQuantity() const                    
    { 
      return theQuantity; 
    }

    /**
       Fixes or unfixes this Substance.
       @param f Boolean value. true -> fix, false -> unfix.
    */
    void fix( bool f )                           
    { 
      theFixed = f; 
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
    Real getConcentration() 
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
    void clear();

    /**
       This is called one or several times in react phase.
       Time of call is determined by the type of the integrator.
    */
    void turn();

    /**
       Transit phase.
       Perform integration by a result calculated by integrator.
    */
    void transit();

    /**
       @return current velocity value in (number of molecules)/(step)
    */
    Real getVelocity() const
    { 
      return theVelocity; 
    }

    /**
       @param v velocity in number of molecules to be added.
    */
    void addVelocity( Real velocity ) 
    {
      theVelocity += velocity; 
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
    void loadQuantity( Real q );

    /**
       This simply set the quantity of this Substance with check of isFixed().
       This updates the accumulator immediately.

       @see isFixed
    */
    void setQuantity(Real q)    
    { 
      if( !isFixed() ) 
	{
	  loadQuantity( q ); 
	}
    }


    /**
       Get a quantity via save() method of the Accumulator.
    */
    Real saveQuantity();


    virtual const char* const getClassName() const { return "Substance"; }

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

    void setAccumulator( StringCref classname );
    void setAccumulator( AccumulatorPtr accumulator );
    void setIntegrator( IntegratorPtr integrator ) 
    { 
      theIntegrator = integrator; 
    }

    void makeSlots();

  private:

    void calculateConcentration();
    void mySetQuantity( Real q ) 
    { 
      theQuantity = q; 
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

    Real theConcentration;
  };


} // namespace libecs

#endif /* ___SUBSTANCE_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/



