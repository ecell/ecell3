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

#ifndef __LIBECS_ENTITY_DEFINED
#include "Entity.hpp"
#endif

#include <boost/shared_ptr.hpp>
#include "Happening.hpp"

/** @addtogroup entities
 *@{
 */

/** @file */
#ifndef __LIBECS_SYSTEM_DEFINED
#define __LIBECS_SYSTEM_DEFINED
namespace libecs
{

class Model;

LIBECS_DM_CLASS( System, Entity )
{
protected:
    // Maps used for entry lists
    DECLARE_UNORDERED_MAP(
                const String, Variable*, DEFAULT_HASHER( String ),
                VariableMap );
    DECLARE_UNORDERED_MAP(
                const String, Process*, DEFAULT_HASHER( String ),
                ProcessMap );
    DECLARE_UNORDERED_MAP(
                const String, System*, DEFAULT_HASHER( String ),
                SystemMap );
    typedef ::std::map<LocalID, Entity*> EntityMap;

    struct EntityEventObserver
    {
    public:
        virtual ~EntityEventObserver();
    };

    typedef Happening<boost::shared_ptr<EntityEventObserver>, Entity*> EntityEvent;

public:
    typedef VariableMap::const_iterator VariableIterator;
    typedef ProcessMap::const_iterator ProcessIterator;
    typedef SystemMap::const_iterator SystemIterator;
    typedef EntityMap::const_iterator EntityIterator;

    LIBECS_DM_BASECLASS( System );

    LIBECS_DM_OBJECT( System, System )
    {
        INHERIT_PROPERTIES( Entity );

        PROPERTYSLOT_SET_GET( String, StepperID );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Size );
    }

    virtual ~System();

    /** @see Entity::initialize */
    virtual void initialize();

    /**
       Retrieves the pointer to a Stepper object assigned to this System.
       @return the pointer to the assigned Stepper object. NULL if unassigned.
    */
    Stepper* getStepper() const
    {
        return theStepper;
    }

    /**
       Assigns a Stepper object to this System.
       @return the pointer to assign to the Stepper object.
    */
    void setStepper( Stepper* obj );

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

    template<typename T_>
    typename UNORDERED_MAP( const String, T_*, DEFAULT_HASHER(String) )::const_iterator
    getBelongings() const;

    /**
       Find a Process with given id in this System.
       This method returns null if not found.
       @return a borrowed pointer to a Process object
     */
    Process* getProcess( const String& id ) const;

    /**
       Find a Variable with given id in this System.
       This method returns null if not found.
       @return a borrowed pointer to a Variable object
     */
    Variable* getVariable( const String& anID ) const;

    /**
       Find an Entity with given id within this System.
       This method returns null if not found.
       @return a borrowed pointer to an Entity object
     */
    Entity* getEntity( const LocalID& localID ) const;

    /**
       Add a Entity object to this System.
       This method takes over the ownership of the given pointer,
       and deletes it if there is an error.
    */
    void add( Entity* anEntity );

    /**
       Check if this is a root System.
       @return true if this is a Root System, false otherwise.
    */
    bool isRootSystem() const
    {
        return !getEnclosingSystem();
    }

    /**
       @see Entity::getSystemPath()
    */
    virtual const SystemPath getPath() const;

    /**
       Get a Model object associated with this system.
       @return a borrowed pointer to the Model.
    */
    Model* getModel() const
    {
        return theModel;
    }

    /**
       Associates a Model object with this system
     */
    void setModel( Model* model )
    {
        if ( __libecs_ready )
            THROW_EXCEPTION( Exception, "Object is already initialized" );
        theModel = model;
    }

    Variable* getSizeVariable() const
    {
        return theSizeVariable;
    }

    void configureSizeVariable();

protected:
    void notifyChangeOfEntityList();

    const Variable* const findSizeVariable() const;

public: // property slots
    GET_METHOD( Polymorph, SystemList );
    GET_METHOD( Polymorph, VariableList );
    GET_METHOD( Polymorph, ProcessList );

protected:
    void addProcess( Process* aProcess );
    void addVariable( Variable* aVariable );
    void addSystem( System* aSystem );

public:
     EntityEvent entityAdded;

protected:
    Stepper*     theStepper;
    Model*       theModel;
    VariableMap  theVariableMap;
    ProcessMap   theProcessMap;
    SystemMap    theSystemMap;
    EntityMap    theEntityMap;
    Variable*    theSizeVariable;
    bool         theEntityListChanged;
};

template<>
System::VariableIterator
System::getBelongings<Variable>() const
{
    return theVariableMap.begin();
}

template<>
System::ProcessIterator
System::getBelongings<Process>() const
{
    return theProcessMap.begin();
}

template<>
System::SystemIterator System::getBelongings<System>() const
{
    return theSystemMap.begin();
}

} // namespace libecs

#endif /* __LIBECS_SYSTEM_DEFINED */

/*@}*/
/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
