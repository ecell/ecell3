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

    void setStepperID( StringCref anID );


    /**
       Get the default StepperID in this System.

       @return an ID of the Stepper as a String.
    */

    const String getStepperID() const;

    /**
       Get the volume of this System in [L] (liter).

       @return Volume of this System.
    */

    virtual const Real getVolume() const = 0;

    /**
       Set a new volume of this System in [L] (liter).

       The volume is updated at the beginning of the next step.

       @param aVolume the new volume value.
     */

    virtual void setVolume( RealCref aVolume ) = 0;

    template <class C>
    const std::map<const String,C*,std::less<const String> >& getMap() const
    {
      DEFAULT_SPECIALIZATION_INHIBITED();
    }

    virtual VariableMapCref getVariableMap() const = 0;

    virtual ProcessMapCref   getProcessMap() const = 0;

    virtual SystemMapCref    getSystemMap() const
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

    VariablePtr getVariable( StringCref id );

    /**
       Find a System with given id in this System. 
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a System object in this System whose ID is id.
    */

    virtual SystemPtr getSystem( StringCref id );


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


    bool isRootSystem() const
    {
      return ( getSuperSystem() == this );
    }

    virtual const SystemPath getSystemPath() const;

    void notifyChangeOfEntityList();


    virtual StringLiteral getClassName() const { return "System"; }
    //    static SystemPtr createInstance() { return new System; }

  public: // property slots

    const Polymorph getSystemList() const;
    const Polymorph getVariableList() const;
    const Polymorph getProcessList() const;

  protected:

    virtual void makeSlots();

  protected:

    StepperPtr   theStepper;

  private:

    SystemMap    theSystemMap;

    bool         theEntityListChanged;

  };


  class VirtualSystem 
    : 
    public System
  {

  public:

    VirtualSystem();
    virtual ~VirtualSystem();

    virtual void initialize();

    /**
       Get the volume of this System in [L] (liter).

       @return Volume of this System.
    */

    virtual const Real getVolume() const
    {
      return getSuperSystem()->getVolume();
    }

    virtual void registerProcess( ProcessPtr aProcess );

    /**
       Set a new volume of this System in [L] (liter).

       The volume is updated at the beginning of the next step.

       @param aVolume the new volume value.
     */

    virtual void setVolume( RealCref aVolume )
    {
      getSuperSystem()->setVolume( aVolume );
    }

    virtual VariableMapCref getVariableMap() const
    {
      return getSuperSystem()->getVariableMap();
    }

    virtual ProcessMapCref   getProcessMap() const
    {
      return theProcessMap;
    }

    virtual StringLiteral getClassName() const { return "VirtualSystem"; }
    static SystemPtr createInstance() { return new VirtualSystem; }

  protected:

    virtual void makeSlots();

  private:

    ProcessMap   theProcessMap;

  };


  class LogicalSystem 
    : 
    public VirtualSystem
  {

  public:

    LogicalSystem();
    virtual ~LogicalSystem();

    virtual void initialize();

    virtual VariableMapCref getVariableMap() const
    {
      return theVariableMap;
    }

    virtual void registerVariable( VariablePtr aVariable );

    virtual StringLiteral getClassName() const { return "LogicalSystem"; }
    static SystemPtr createInstance() { return new LogicalSystem; }

  protected:

    virtual void makeSlots();

  private:

    VariableMap theVariableMap;

  };



  class CompartmentSystem 
    : 
    public LogicalSystem
  {

  public:

    CompartmentSystem();
    virtual ~CompartmentSystem();

    virtual void initialize();

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
      theVolume = aVolume;
    }

    virtual StringLiteral getClassName() const { return "CompartmentSystem"; }
    static SystemPtr createInstance() { return new CompartmentSystem; }
 
 protected:

    virtual void makeSlots();

  private:

    Real theVolume;

    Real theVolumeBuffer;

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

#endif /* ___SYSTEM_H___ */


/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
