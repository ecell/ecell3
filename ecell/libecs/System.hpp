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

#ifndef ___SYSTEM_H___
#define ___SYSTEM_H___
#include <map>

#include "libecs.hpp"

#include "Entity.hpp"
#include "Reactor.hpp"

namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


  // Maps used for entry lists
  DECLARE_MAP( const String, SubstancePtr, 
	       std::less<const String>, SubstanceMap );
  DECLARE_MAP( const String, ReactorPtr,   
	       std::less<const String>, ReactorMap );
  DECLARE_MAP( const String, SystemPtr,    
	       std::less<const String>, SystemMap );


  class System 
    : 
    public Entity
  {

  public:

    /** 
	A function type that returns a pointer to System.

	Every subclass must have this type of a function which returns
	an instance for the SystemMaker.
    */

    typedef SystemPtr (* AllocatorFuncPtr )();


    System();
    virtual ~System();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::SYSTEM );
    }

    virtual void initialize();

    virtual void integrate()
    {
      updateVolume();
    }


    /**
       Get a pointer to a Stepper object that this System belongs.

       @return A pointer to a Stepper object that this System belongs or
       NULL pointer if it is not set.
    */

    virtual StepperPtr getStepper() const 
    { 
      return theStepper; 
    }

    /**
       Register the Stepper object as a stepper for this System by an ID.

       @param anID Stepper ID.
    */

    void setStepperID( StringCref anID );


    /**
       Get an ID of the Stepper.

       @return an ID of the Stepper as a String.
    */

    const String getStepperID() const;

    /**
       Get the volume of this System in [L] (liter).

       @return Volume of this System.
    */

    virtual const Real getVolume() const
    {
      return theVolume;
    }

    /**
       Set a new volume of this System in [L] (liter).

       The volume is updated at the beginning of the next step.

       @param aVolume the new volume value.
     */

    virtual void setVolume( RealCref aVolume )
    {
      theVolumeBuffer = aVolume;
    }

    void updateVolume( )
    {
      theVolume = theVolumeBuffer;
      updateConcentrationFactor();
    }


    template <class C>
    const std::map<const String,C*> getMap() const
    {
      DEFAULT_SPECIALIZATION_INHIBITED();
    }

    SubstanceMapCref getSubstanceMap() const
    {
      return theSubstanceMap;
    }

    ReactorMapCref   getReactorMap() const
    {
      return theReactorMap;
    }

    SystemMapCref    getSystemMap() const
    {
      return theSystemMap;
    }


    /**
       Find a Reactor with given id in this System.  
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Reactor object in this System named @a id.
    */

    ReactorPtr getReactor( StringCref anID ) ;


    /**
       Find a Substance with given id in this System. 
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Substance object in this System named @a id.
    */

    SubstancePtr getSubstance( StringCref id );

    /**
       Find a System with given id in this System. 
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a System object in this System whose ID is id.
    */

    virtual SystemPtr getSystem( StringCref id );


    /**
       Register a Reactor object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    void registerReactor( ReactorPtr aReactor );
  

    /**
       Register a Substance object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    void registerSubstance( SubstancePtr aSubstance );
  

    /**
       Register a System object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    void registerSystem( SystemPtr aSystem );


    bool isRootSystem() const
    {
      return ( getSuperSystem() == this );
    }


    const Real getActivityPerSecond() const;

    virtual void setStepInterval( RealCref aStepInterval );
    virtual const Real getStepInterval() const;
    const Real getStepsPerSecond() const;

    virtual const SystemPath getSystemPath() const;

    void notifyChangeOfEntityList();

    /**
       This method returns 1 / ( volume * Avogadro's number ).

       Quantity * getConcentrationFactor() forms 
       concentration in M (molar).

       When calculating the concentration, using this is faster
       than getVolume() because the value is precalculated when
       setConcentration() is called.

    */

    RealCref getConcentrationFactor() const
    {
      return theConcentrationFactor;
    }


    virtual StringLiteral getClassName() const { return "System"; }
    static SystemPtr createInstance() { return new System; }

  public: // property slots

    const UVariableVectorRCPtr getSystemList() const;
    const UVariableVectorRCPtr getSubstanceList() const;
    const UVariableVectorRCPtr getReactorList() const;

  protected:

    void updateConcentrationFactor() 
    {
      theConcentrationFactor = 1 / ( getVolume() * N_A );
    }

    virtual void makeSlots();

  protected:

    StepperPtr theStepper;

  private:

    Real theVolume;

    Real theVolumeBuffer;

    Real theConcentrationFactor;

    ReactorMap   theReactorMap;
    SubstanceMap theSubstanceMap;
    SystemMap    theSystemMap;

    bool         theEntityListChanged;

  };


  template <>
  inline const std::map<const String,SubstancePtr> System::getMap() const
  {
    return theSubstanceMap;
  }

  template <>
  inline const std::map<const String,ReactorPtr>   System::getMap() const
  {
    return theReactorMap;
  }

  template <>
  inline const std::map<const String,SystemPtr>    System::getMap() const
  {
    return theSystemMap;
  }



  /*@}*/

} // namespace libecs

#endif /* ___SYSTEM_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
