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

#ifndef ___SYSTEM_HPP
#define ___SYSTEM_HPP
#include <map>

#include "libecs.hpp"

#include "Entity.hpp"

namespace libecs
{

  /** @addtogroup entities
   *@{
   */

  /** @file */


  // Maps used for entry lists
  DECLARE_MAP( const String, VariablePtr, 
	       std::less<const String>, VariableMap );
  DECLARE_MAP( const String, ProcessPtr,   
	       std::less<const String>, ProcessMap );
  DECLARE_MAP( const String, SystemPtr,    
	       std::less<const String>, SystemMap );


  class System 
    : 
    public Entity
  {

  public:

    DM_BASECLASS( System );


    System();
    virtual ~System();

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::SYSTEM );
    }

    virtual void initialize();

    /**
       Get a pointer to a Stepper object that this System belongs.

       @return A pointer to a Stepper object that this System belongs or
       NULL pointer if it is not set.
    */

    StepperPtr getStepper() const 
    { 
      return theStepper; 
    }

    /**
       Set a StepperID.

       This provides a default Stepper to Processes holded by this System.

       @param anID Stepper ID.
    */

    SET_METHOD( String, StepperID );


    /**
       Get the default StepperID in this System.

       @return an ID of the Stepper as a String.
    */

    GET_METHOD( String, StepperID );

    /**
       Get the volume of this System in [L] (liter).

       @return Volume of this System.
    */

    virtual GET_METHOD( Real, Volume ) = 0;

    GET_METHOD( Real, VolumeN_A )
    {
      return getVolume() * N_A;
    }

    /**
       Set a new volume of this System in [L] (liter).

       The volume is updated at the beginning of the next step.

       @param aVolume the new volume value.
     */

    virtual SET_METHOD( Real, Volume ) = 0;

    template <class C>
    const std::map<const String,C*,std::less<const String> >& getMap() const
    {
      DEFAULT_SPECIALIZATION_INHIBITED();
    }

    virtual VariableMapCref  getVariableMap() const = 0;

    virtual ProcessMapCref   getProcessMap() const = 0;

    SystemMapCref    getSystemMap() const
    {
      return theSystemMap;
    }


    /**
       Find a Process with given id in this System.  
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Process object in this System named @a id.
    */

    ProcessPtr getProcess( StringCref anID ) ;


    /**
       Find a Variable with given id in this System. 
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Variable object in this System named @a id.
    */

    VariablePtr getVariable( StringCref aSystemPath );

    /**
       Find a System pointed by the given SystemPath relative to
       this System.
       
       If aSystemPath is empty, this method returns this System.

       If aSystemPath is absolute ( starts with '/' ), this method
       calls getSystem() of the Model object, and returns the result.

       This method throws NotFound exception if it is not found.

       @param aSystemPath A SystemPath object.
       @return a borrowed pointer to a System object pointed by aSystemPath.
    */

    SystemPtr getSystem( SystemPathCref anID );


    /**
       Find a System with a given id in this System. 
       
       This method throws NotFound exception if it is not found.

       Unlike getSystem( SystemPath ) method, this method searches only
       within this System.  In the other words this method doesn't 
       conduct a recursive search.

       @param anID An ID string of a System.

       @return a borrowed pointer to a System object in this System
       whose ID is anID.
    */

    SystemPtr getSystem( StringCref id );


    /**
       Register a Process object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    virtual void registerProcess( ProcessPtr aProcess );
  

    /**
       Register a Variable object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    virtual void registerVariable( VariablePtr aVariable );
  

    /**
       Register a System object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    virtual void registerSystem( SystemPtr aSystem );

    /**
       Check if this is a root System.


       @return true if this is a Root System, false otherwise.
    */

    bool isRootSystem() const
    {
      return ( getSuperSystem() == this );
    }

    /**
       @see Entity::getSystePath()
    */

    virtual const SystemPath getSystemPath() const;


    /**
       Get a Model object to which this System belongs.

       @return a borrowed pointer to the Model.
    */

    ModelPtr getModel() const
    {
      return theModel;
    }

    void setModel( ModelPtr const aModel )
    {
      theModel = aModel;
    }

    void notifyChangeOfEntityList();


  public: // property slots

    const Polymorph getSystemList() const;
    const Polymorph getVariableList() const;
    const Polymorph getProcessList() const;

  protected:

    StepperPtr   theStepper;

  private:

    ModelPtr  theModel;

    SystemMap    theSystemMap;

    bool         theEntityListChanged;

  };


  class VirtualSystem 
    : 
    public System
  {

  public:

    LIBECS_DM_OBJECT( System, VirtualSystem );


    VirtualSystem();
    virtual ~VirtualSystem();

    /**
       Get the volume of this System in [L] (liter).

       @return Volume of this System.
    */

    virtual GET_METHOD( Real, Volume )
    {
      return getSuperSystem()->getVolume();
    }

    /**
       Set a new volume of this System in [L] (liter).

       The volume is updated at the beginning of the next step.

       @param aVolume the new volume value.
     */

    virtual SET_METHOD( Real, Volume )
    {
      getSuperSystem()->setVolume( value );
    }

    virtual void initialize();

    virtual void registerProcess( ProcessPtr aProcess );

    virtual VariableMapCref getVariableMap() const
    {
      return getSuperSystem()->getVariableMap();
    }

    virtual ProcessMapCref   getProcessMap() const
    {
      return theProcessMap;
    }

  private:

    ProcessMap   theProcessMap;

  };


  class LogicalSystem 
    : 
    public VirtualSystem
  {

  public:

    LIBECS_DM_OBJECT( System, LogicalSystem );

    LogicalSystem();
    virtual ~LogicalSystem();

    virtual void initialize();

    virtual VariableMapCref getVariableMap() const
    {
      return theVariableMap;
    }

    virtual void registerVariable( VariablePtr aVariable );

  private:

    VariableMap theVariableMap;

  };



  class CompartmentSystem 
    : 
    public LogicalSystem
  {

  public:

    LIBECS_DM_OBJECT( System, CompartmentSystem );

    CompartmentSystem();
    virtual ~CompartmentSystem();

    virtual GET_METHOD( Real, Volume )
    {
      return theVolume;
    }

    /**
       Set a new volume of this System in [L] (liter).

       The volume is updated at the beginning of the next step.

       @param aVolume the new volume value.
     */

    virtual SET_METHOD( Real, Volume )
    {
      theVolume = value;
    }

    virtual void initialize();
 
  private:

    Real theVolume;

  };




  template <>
  inline VariableMapCref System::getMap() const
  {
    return getVariableMap();
  }

  template <>
  inline ProcessMapCref   System::getMap() const
  {
    return getProcessMap();
  }

  template <>
  inline SystemMapCref    System::getMap() const
  {
    return getSystemMap();
  }



  /*@}*/

} // namespace libecs

#endif /* __SYSTEM_HPP */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
