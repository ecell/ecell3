//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2007 Keio University
//       Copyright (C) 2005-2007 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell System is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell System is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell System -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi <shafi@e-cell.org>,
// E-Cell Project.
//

#ifndef __VARIABLE_HPP
#define __VARIABLE_HPP

#include <utility>

#include "libecs.hpp"
#include "Entity.hpp"
#include "Interpolant.hpp"
#include "System.hpp"


namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


  /**
     Variable class represents state variables in the simulation model, such as
     amounts of molecular species in a compartment.

  */

  LIBECS_DM_CLASS( Variable, Entity )
  {

  public:

    LIBECS_DM_BASECLASS( Variable );

    LIBECS_DM_OBJECT( Variable, Variable )
      {
	INHERIT_PROPERTIES( Entity );
	
	PROPERTYSLOT_LOAD_SAVE( Real, Value,
				&Variable::setValue,
				&Variable::getValue,
				&Variable::loadValue,
				&Variable::saveValue );


	PROPERTYSLOT_SET_GET( Real,  DiffusionCoeff );

	PROPERTYSLOT_SET_GET( Integer,  Fixed );

	PROPERTYSLOT_NO_LOAD_SAVE( Real, Velocity,
				   NOMETHOD,
				   &Variable::getVelocity );

	PROPERTYSLOT_LOAD_SAVE( Real, MolarConc,
				&Variable::setMolarConc,
				&Variable::getMolarConc,
				&Variable::loadMolarConc,
				NOMETHOD );
	//	PROPERTYSLOT_NO_LOAD_SAVE( Real, MolarConc,
	//				   &Variable::setMolarConc,
	//				   &Variable::getMolarConc );

	PROPERTYSLOT_NO_LOAD_SAVE( Real, NumberConc,
				   &Variable::setNumberConc,
				   &Variable::getNumberConc );
      }


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
       Clear theVelocity by zero.
    */

    virtual const bool isIntegrationNeeded() const
    {
      return ! theInterpolantVector.empty();
    }

    /** 
	Integrate.
    */

    virtual void integrate( RealParam aTime )
    {
      if( isFixed() == false ) 
	{
	  updateValue( aTime );
	}
      else 
	{
	  theLastTime = aTime;
	}
    }


    /**

    This method is used internally by DifferentialStepper.

    @internal
    */

    void interIntegrate( RealParam aCurrentTime )
      {
	const Real anInterval( aCurrentTime - theLastTime );
	
	if ( anInterval > 0.0 )
	  {
	    Real aVelocitySum( calculateDifferenceSum( aCurrentTime,
						       anInterval ) );
	    loadValue( getValue() + aVelocitySum );
	  }
      }


    /**
       This simply sets the value of this Variable if getFixed() is false.

       @see getFixed()
    */

    virtual SET_METHOD( Real, Value )
    { 
      if ( !isFixed() ) 
	{
	  loadValue( value ); 
	}
    }


    // Currently this is non-virtual, but will be changed to a 
    // virtual function, perhaps in version 3.3.
    // virtual
    GET_METHOD( Real, Value )
    { 
      return saveValue();
    }

    void addValue( RealParam aValue )
    {
      setValue( getValue() + aValue );
    }

    void loadValue( RealParam aValue )
    {
      theValue = aValue;
    }

    const Real saveValue() const
    {
      return theValue;
    }

    /**
       @return current velocity value in (number of molecules)/(step)
    */

    GET_METHOD( Real, Velocity )
    {
      Real aVelocitySum( 0.0 );
      FOR_ALL( InterpolantVector, theInterpolantVector )
	{
	  InterpolantPtr const anInterpolantPtr( *i );
	  aVelocitySum += anInterpolantPtr->getVelocity( theLastTime );
	}

      return aVelocitySum;
    }

    /**

    A wrapper to set Fixed property by a bool value.

    */

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
    SET_METHOD( Integer, Fixed )
    { 
      theFixed = value != 0;
    }

    GET_METHOD( Integer, Fixed )
    { 
      return theFixed;
    }

    SET_METHOD( Real, DiffusionCoeff )
    { 
      theDiffusionCoeff = value;
    }

    GET_METHOD( Real, DiffusionCoeff )
    { 
      return theDiffusionCoeff;
    }

    /**
       Returns the molar concentration of this Variable.

       @return Concentration in M [mol/L].
    */

    GET_METHOD( Real, MolarConc )
    {
      // N_A_R = 1.0 / N_A
      return getNumberConc() * N_A_R;
    }

    /**
       Set the molar concentration of this Variable.

       @param value Concentration in M [mol/L].
    */

    SET_METHOD( Real, MolarConc )
    {
      setNumberConc( value * N_A );
    }

    /**
       Load the molar concentration of this Variable.

       This method uses loadNumberConc() instead of setNumberConc().

       @see setNumberConc()
    */

    LOAD_METHOD( Real, MolarConc )
    {
      loadNumberConc( value * N_A );
    }

    /**
       Returns the number concentration of this Variable.

       Unlike getMolarConc, this method just returns value / size.

       @return Concentration in [number/L].
    */

    GET_METHOD( Real, NumberConc )
    {
      return getValue() / getSizeOfSuperSystem();

      // This uses getSizeOfSuperSystem() private method instead of
      // getSuperSystem()->getSize() because otherwise it is 
      // impossible to inline this.
    }

    /**
       Set the number concentration of this Variable.

       @param value concentration in [number/L].
    */

    SET_METHOD( Real, NumberConc )
    {
      setValue( value * getSizeOfSuperSystem() );

      // This uses getSizeOfSuperSystem() private method instead of
      // getSuperSystem()->getSize() because otherwise it is 
      // impossible to inline this.
    }


    /**
       Load the number concentration of this Variable.

       This method can be called before the SIZE Variable of 
       the supersystem of this Variable is configured in
       Model::initialize().

       Thus this method gets the value of the SIZE Variable
       without relying on the System::getSizeVariable() method
       of the supersystem.

       @see loadMolarConc()
       @see System::getSizeVariable()
       @see System::configureSizeVariable()
       @see System::findSizeVariable()
    */

    LOAD_METHOD( Real, NumberConc );

    void registerInterpolant( InterpolantPtr const anInterpolant );
    //    void removeInterpolant( InterpolantPtr const anInterpolant );

    void setCreationTime(Real creationTime)
    {
      theLastTime = creationTime;
      creationTimeInitialized = true;
    }
    
  protected:

    const Real calculateDifferenceSum( RealParam aCurrentTime, 
				       RealParam anInterval ) const
    {
      Real aVelocitySum( 0.0 );
      FOR_ALL( InterpolantVector, theInterpolantVector )
	{
	  InterpolantPtr const anInterpolantPtr( *i );
	  aVelocitySum += anInterpolantPtr->getDifference( aCurrentTime,
							   anInterval );
	}

      return aVelocitySum;
    }


    void updateValue( RealParam aCurrentTime )
    {
      const Real anInterval( aCurrentTime - theLastTime );

      if( anInterval == 0.0 )
	{
	  return;
	}

      const Real aVelocitySum( calculateDifferenceSum( aCurrentTime, 
						       anInterval ) );
      loadValue( getValue() + aVelocitySum );

      theLastTime = aCurrentTime;
    }



    void clearInterpolantVector();

  private:

    const Real getSizeOfSuperSystem() const
      {
	return getSuperSystem()->getSizeVariable()->getValue();
      }


  protected:

    Real theValue;

    Real theLastTime;

    Real theDiffusionCoeff;

    InterpolantVector theInterpolantVector;

    bool theFixed;

    bool creationTimeInitialized;
  };


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
