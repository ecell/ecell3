//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-Cell Simulation Environment package
//
//                Copyright (C) 1996-2002 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Kouichi Takahashi <shafi@e-cell.org>,
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#ifndef __SYSTEM_HPP
#define __SYSTEM_HPP

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


  LIBECS_DM_CLASS( System, Entity )
  {
    
  public:

    LIBECS_DM_BASECLASS( System );

    LIBECS_DM_OBJECT( System, System )
    {
      INHERIT_PROPERTIES( Entity );
      
      //    PROPERTYSLOT_SET_GET( Real,      Dimension );
      PROPERTYSLOT_SET_GET( String,    StepperID );
      
      PROPERTYSLOT_GET_NO_LOAD_SAVE( Real,      Size );

    }

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
       Get the size of this System in [L] (liter).

       @return Size of this System.
    */

    GET_METHOD( Real, Size );

    GET_METHOD( Real, SizeN_A )
    {
      return getSize() * N_A;
    }

    template <class C>
      const std::map<const String,C*,std::less<const String> >& getMap() const;
    //    {
    //      DEFAULT_SPECIALIZATION_INHIBITED();
    //    }

    VariableMapCref getVariableMap() const
    {
      return theVariableMap;
    }

    ProcessMapCref  getProcessMap() const
    {
      return theProcessMap;
    }

    SystemMapCref    getSystemMap() const
    {
      return theSystemMap;
    }


    /**
       Find a Process with given id in this System.  
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Process object in this System named @a id.
    */

    ProcessPtr getProcess( StringCref anID ) const;


    /**
       Find a Variable with given id in this System. 
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Variable object in this System named @a id.
    */

    VariablePtr getVariable( StringCref anID ) const;

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

    SystemPtr getSystem( SystemPathCref anID ) const;


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

    SystemPtr getSystem( StringCref id ) const;


    /**
       Register a Process object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    void registerProcess( ProcessPtr aProcess );
  

    /**
       Register a Variable object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    void registerVariable( VariablePtr aVariable );
  

    /**
       Register a System object in this System.

       This method steals ownership of the given pointer, and deletes
       it if there is an error.
    */

    void registerSystem( SystemPtr aSystem );

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

    VariableCptr const getSizeVariable() const
    {
      return theSizeVariable;
    }

    void notifyChangeOfEntityList();

    VariableCptr const findSizeVariable() const;

    void configureSizeVariable();

  public: // property slots

    GET_METHOD( Polymorph, SystemList );
    GET_METHOD( Polymorph, VariableList );
    GET_METHOD( Polymorph, ProcessList );

  protected:

    StepperPtr   theStepper;

  private:

    ModelPtr     theModel;

    VariableMap  theVariableMap;
    ProcessMap   theProcessMap;
    SystemMap    theSystemMap;

    VariableCptr  theSizeVariable;

    bool         theEntityListChanged;

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
