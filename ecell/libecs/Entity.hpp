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

#ifndef ___ENTITY_H___
#define ___ENTITY_H___

#include "libecs.hpp"
#include "PrimitiveType.hpp"
#include "PropertyInterface.hpp"


namespace libecs
{

  /** @defgroup libecs_module The Libecs Module 
   * This is the libecs module 
   * @{ 
   */ 
  
  /**
     Entity class is a base class for all components in the cell model.
     Entity is-a PropertyInterface. 

  */

  class Entity 
    : 
    public PropertyInterface
  {

  public:

    Entity(); 
    virtual ~Entity();

    /**
       Set supersystem pointer of this Entity.  
       Usually no need to set this manually because a System object will 
       do this when an Entity is installed to the System.

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

    ModelPtr getModel() const
    {
      return theModel;
    }

    SystemPtr getSuperSystem() const 
    {
      return theSuperSystem;
    }

    /**
       Set an identifier of this Entity.

       @param entryname entryname of this Entry.
    */
    void setID( StringCref id ) 
    { 
      theID = id; 
    }

    /**
       @return entryname of this Entity.
    */
    const String getID() const
    { 
      return theID; 
    }

    /**
       Set name of this Entity.

       @param name name of this Entity.
    */
    void setName( StringCref name ) 
    { 
      theName = name; 
    }

    /**
       @return name of this Entity.
    */
    const String getName() const 
    { 
      return theName; 
    }

    /**
       @return FullID of this Entity.
    */

    const FullID getFullID() const;

    const String getFullIDString() const;

    virtual const PrimitiveType getPrimitiveType() const
    {
      return PrimitiveType( PrimitiveType::ENTITY );
    }

    /**
       Returns SystemPath of this Entity.

       The SystemPath doesn't include ID of this Entity even if 
       this Entity is a System.
    */

    virtual const SystemPath getSystemPath() const;

    /**
       Returns activity value (per delta-T) of this Entity.
       Override this in subclasses.  If there is no overriding method,
       this returns zero.

       @return activity of this Entity
       @see getActivityPerSecond()
    */
    virtual const Real getActivity() const;

    /**
       Returns activity value (per second).
       Default action of this method is to return getActivity() / delta-T,
       but this can be changed in subclasses.

       @return activity of this Entity per second
    */
    virtual const Real getActivityPerSecond() const;

    /**
       Returns a pointer to a Stepper object that this Entity belongs.

       The Stepper of an Entity is defined as a Stepper of a
       supersystem of the Entity.

       This is overridden in System class because Stepper of a System is 
       explicitly defined as a member variable, not as a supersystem's.

       @return A pointer to a Stepper object that this Entity belongs or
       NULL pointer if it is not set.
    */

    virtual StepperPtr getStepper() const;

    virtual PropertySlotPtr getPropertySlot( StringCref aPropertyName, 
					     EntityCptr aRequester );

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

  /** @} */ //end of libecs_module 

} // namespace libecs

#endif /*  ___ENTITY_H___ */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
