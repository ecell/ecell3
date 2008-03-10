//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2008 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

#include "libecs.hpp"
#include "EntityType.hpp"
#include "PropertiedClass.hpp"
#include "FullID.hpp"

#ifndef __LIBECS_ENTITY_DEFINED
#define __LIBECS_ENTITY_DEFINED

/**
   @addtogroup entities The Entities.
   Entities.
  
   @ingroup libecs
   @{
 */
/** @file */

namespace libecs {

DECLARE_VECTOR( EntityPtr, EntityVector );


/**
   Entity class is a base class for all components in the cell model.

*/


LIBECS_DM_CLASS( Entity, PropertiedClass )
{
public:
    LIBECS_DM_OBJECT_ABSTRACT( Entity )
    {
        INHERIT_PROPERTIES( PropertiedClass );
        PROPERTYSLOT_SET_GET( String, Name );
    }

    Entity();
    virtual ~Entity();

    /**
       Called right before the simulation gets kicked off.
     */
    virtual void initialize();

    /**
       Get the System where this Entity belongs.
       @return the borrowed pointer to the super system.
    */
    SystemPtr getEnclosingSystem() const
    {
        return theSuperSystem;
    }

    /**
       Get the FullID of this Entity.
       @return the FullID of this Entity.
    */
    const FullID getFullID() const;

    /**
       Get EntityType of this Entity.
       This method is overridden in Variable, Process and System classes.
       @return EntityType of this Entity object.
       @see EntityType
    */
    const EntityType& getEntityType() const
    {
        return EntityType::fromPropertiedClassKind(
                getPropertyInterface().getKind() );
    }
    /// \name Properties
    //@{

    /**
       Set an identifier of this Entity.

       @param anID an id of this Entry.
    */

    SET_METHOD( String, ID )
    {
        theID = value;
    }

    /**
       Get an id string of this Entity.

       @return an id of this Entity.
    */

    GET_METHOD( String, ID )
    {
        return theID;
    }

    /**
       Set name of this Entity.

       @param aName a name of this Entity.
    */

    SET_METHOD( String, Name )
    {
        theName = value;
    }

    /**
       Get a name of this Entity.

       @return a name of this Entity.
    */

    GET_METHOD( String, Name )
    {
        return theName;
    }
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

private:
    Entity( const Entity& ); // no copy construction
    Entity& operator=( Entity& ); // no assignment

private:
    System*   theSuperSystem;
    String    theID;
    String    theName;
};

} // namespace libecs
#endif /* __LIBECS_ENTITY_DEFINED */

#include "System.hpp"

#ifndef __LIBECS_ENTITY_MEMBER_DEFINED
#define __LIBECS_ENTITY_MEMBER_DEFINED

namespace libecs {

const FullID Entity::getFullID() const
{
    return FullID(
            getEntityType(),
            getEnclosingSystem() ?
                getEnclosingSystem()->getPath():
                SystemPath(),
            getID() );
}

} // namespace libecs

#endif /* __LIBECS_ENTITY_MEMBER_DEFINED */

/** @} */
/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
