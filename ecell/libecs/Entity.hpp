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

#ifndef ___ENTITY_H___
#define ___ENTITY_H___

#include "libecs.hpp"
#include "EntityType.hpp"
#include "PropertyInterface.hpp"


namespace libecs
{

  /** @addtogroup entities The Entities.
      Entities.
      
      @ingroup libecs

   
      @{ 
   */ 

  /** @file */

  
  /**
     Entity class is a base class for all components in the cell model.

  */

  class Entity 
    : 
    public PropertyInterface
  {

  public:

    Entity(); 
    virtual ~Entity();


    /**
       Get a System to which this Entity belongs.

       @return a borrowed pointer to the super system.
    */

    SystemPtr getSuperSystem() const 
    {
      return theSuperSystem;
    }


    /**
       Get a FullID of this Entity.

       @return a FullID of this Entity.
    */

    const FullID getFullID() const;


    /**
       Get EntityType of this Entity.

       This method is overridden in Substance, Reactor and System classes.

       @return EntityType of this Entity object.
       @see EntityType
    */

    virtual const EntityType getEntityType() const
    {
      return EntityType( EntityType::ENTITY );
    }


    /**
       Get a Model object to which this Entity belongs.

       @return a borrowed pointer to the Model.
    */

    ModelPtr getModel() const
    {
      return theModel;
    }


    /**
       Get a SystemPath of this Entity.

       @note The SystemPath doesn't include ID of this Entity even if 
       this Entity is a System.

       @return a SystemPath of this Entity.
    */

    virtual const SystemPath getSystemPath() const;

    /**
       Returns a pointer to a Stepper object that this Entity belongs.

       The Stepper of an Entity is defined as a Stepper of a
       supersystem of the Entity.  As a exception, a System has a
       pointer to a Stepper as its member variable, thus this method
       returns the variable in System class.

       @return A pointer to a Stepper object that this Entity belongs or
       NULLPTR if it is not set.
    */

    virtual StepperPtr getStepper() const;


    /// \name Properties
    //@{

    /**
       Set an identifier of this Entity.

       @param anID an id of this Entry.
    */

    void setID( StringCref anID ) 
    { 
      theID = anID; 
    }

    /**
       Get an id string of this Entity.

       @return an id of this Entity.
    */

    const String getID() const
    { 
      return theID; 
    }

    /**
       Set name of this Entity.

       @param aName a name of this Entity.
    */

    void setName( StringCref aName ) 
    { 
      theName = aName;
    }

    /**
       Get a name of this Entity.

       @return a name of this Entity.
    */

    const String getName() const 
    { 
      return theName; 
    }

    /**
       Get a FullID of this Entity as String.

       @note Property name for this method is 'getFullID', not
       'getFullIDString.'

       @return a FullID string of this Entity.
    */

    const String getFullIDString() const;


    //@}


    /**
       @internal

       Set a supersystem of this Entity.  

       Usually no need to set this manually because a System object does
       this when an Entity is added to the System.

       @param supersystem a pointer to a System to which this object belongs.
    */

    void setSuperSystem( SystemPtr const supersystem ) 
    { 
      theSuperSystem = supersystem; 
    }

    void setModel( ModelPtr const aModel )
    {
      theModel = aModel;
    }


    /// @internal

    virtual StringLiteral getClassName() const { return "Entity"; }


  protected:

    virtual void makeSlots();

  private:

    // hide them
    Entity( EntityRef );
    EntityRef operator=( EntityRef );

  private:

    ModelPtr  theModel;
    SystemPtr theSuperSystem;
    String    theID;
    String    theName;
  };

  /*@}*/

} // namespace libecs

#endif /*  ___ENTITY_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
