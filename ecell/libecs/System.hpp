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

    class isRegularReactorItem;

  public:

    System();
    virtual ~System();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::SYSTEM );
    }

    virtual void initialize();

    /**
       Returns a pointer to a Stepper object that this System belongs.

       This overrides Entity::getStepper().

       @return A pointer to a Stepper object that this System belongs or
       NULL pointer if it is not set.
    */

    virtual StepperPtr getStepper() const 
    { 
      return theStepper; 
    }

    /**
       Instantiate a Stepper object of @a classname using theModel's
       StepperMaker object.  Register the Stepper object as a stepper for 
       this System.

       @param classname Classname of the Stepper that this System may have.
    */

    void setStepperID( StringCref anID );

    /**
       @return Volume of this System. Unit is [L].
    */

    virtual const Real getVolume() const
    {
      return theVolume;
    }

    /**
       Set a new volume for this System. 
       Make the new volume effective from beginning of next time step.
     */

    virtual void setVolume( RealCref aVolume )
    {
      theVolume = aVolume;
      updateConcentrationFactor();
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

    void setStepperID( UVariableVectorRCPtrCref aMessage );

    const UVariableVectorRCPtr getStepperID() const;

    const UVariableVectorRCPtr getSystemList() const;
    const UVariableVectorRCPtr getSubstanceList() const;
    const UVariableVectorRCPtr getReactorList() const;

  protected:

    void updateConcentrationFactor() 
    {
      theConcentrationFactor = 1 / ( theVolume * N_A );
    }

    virtual void makeSlots();

  protected:

    StepperPtr theStepper;

    ReactorMapConstIterator theFirstRegularReactorIterator;

  private:

    Real theVolume;

    Real theConcentrationFactor;

    ReactorMap   theReactorMap;
    SubstanceMap theSubstanceMap;
    SystemMap    theSystemMap;

    bool         theEntityListChanged;

  };

  /**
  Equivalent to Reactor::isRegularReactor except that
  this function object takes a reference to a ReactorMap::value_type.
  */
  class System::isRegularReactorItem
    : public std::unary_function< const ReactorMap::value_type,bool >
  {
  public:
    bool operator()( const ReactorMap::value_type r ) const
    {
      return Reactor::isRegularReactor::isRegularName( ( r.second )->getID() );
    }
  };

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
