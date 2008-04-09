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
#endif /* __LIBECS_ENTITY_DEFINED */

#include <boost/shared_ptr.hpp>
#include <boost/range/iterator_range.hpp>

#include "Happening.hpp"
#include "RangeConcatenator.hpp"

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
                LocalID, Variable*, LocalID::Hasher,
                VariableMap );
    DECLARE_UNORDERED_MAP(
                LocalID, Process*, LocalID::Hasher,
                ProcessMap );
    DECLARE_UNORDERED_MAP(
                LocalID, System*, LocalID::Hasher,
                SystemMap );

    typedef RangeConcatenator< ::boost::mpl::vector<
            VariableMap, ProcessMap, SystemMap >,
            ::std::pair< const LocalID, Entity* > > EntityMap;

public:
    struct EntityEventDescriptor
    {
        EntityEventDescriptor( System* _system, Entity* _entity,
                const LocalID& _id )
            : system( _system ), entity( _entity ), id( _id ) {} 

        System* system;
        Entity* entity;
        LocalID id;
    };

    struct EntityEventObserver
    {
        typedef EntityEventDescriptor Descriptor;
    };

    typedef Happening<boost::shared_ptr<EntityEventObserver>, EntityEventDescriptor> EntityEvent;

    typedef ::boost::iterator_range< VariableMap::iterator > VariablesRange;
    typedef ::boost::iterator_range< VariableMap::const_iterator > VariablesCRange;
    typedef ::boost::iterator_range< ProcessMap::iterator > ProcessesRange;
    typedef ::boost::iterator_range< ProcessMap::const_iterator > ProcessesCRange;

    typedef ::boost::iterator_range< SystemMap::iterator > SystemsRange;
    typedef ::boost::iterator_range< SystemMap::const_iterator > SystemsCRange;

    typedef ::boost::iterator_range< EntityMap::iterator > EntitiesRange;
    typedef ::boost::iterator_range< EntityMap::const_iterator > EntitiesCRange;

public:
    LIBECS_DM_BASECLASS( System );

    LIBECS_DM_OBJECT( System, System )
    {
        INHERIT_PROPERTIES( Entity );

        PROPERTYSLOT_SET_GET( String, StepperID );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Size );
    }

    virtual ~System();

    virtual void startup();

    virtual void initialize();

    virtual void postInitialize();

    /**
       Retrieves the pointer to a Stepper object assigned to this System.
       @return the pointer to the assigned Stepper object. NULL if unassigned.
    */
    Stepper* getStepper() const
    {
        return stepper_;
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
    ::boost::iterator_range< typename UNORDERED_MAP( LocalID, T_*, DEFAULT_HASHER(String) )::const_iterator > getBelongings() const;

    EntitiesCRange getBelongings() const;

    template<typename T_>
    ::boost::iterator_range< typename UNORDERED_MAP( LocalID, T_*, DEFAULT_HASHER(String) )::iterator > getBelongings();
    EntitiesRange getBelongings();

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
    Variable* getVariable( const String& id ) const;

    /**
       Find a System with given id in this System.
       This method returns null if not found.
       @return a borrowed pointer to a Variable object
     */
    System* getSystem( const String& id ) const;

    /**
       Find a System with given system path in this System.
       This method returns null if not found.
       @return a borrowed pointer to a Variable object
     */
    System* getSystem( const SystemPath& path ) const;

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
    void add( const String& id, Entity* anEntity );

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
    const SystemPath getPath() const;

    Variable* getSizeVariable() const
    {
        return sizeVariable_;
    }

    void configureSizeVariable();

protected:
    void notifyChangeOfEntityList();

    Variable* const findSizeVariable() const;

public: // property slots
    GET_METHOD( Polymorph, SystemList );
    GET_METHOD( Polymorph, VariableList );
    GET_METHOD( Polymorph, ProcessList );

protected:
    void addProcess( const String& id, Process* aProcess );
    void addVariable( const String& id, Variable* aVariable );
    void addSystem( const String& id, System* aSystem );

public:
    EntityEvent  entityAdded;
    EntityEvent  entityRemoved;

protected:
    Stepper*     stepper_;
    VariableMap  variables_;
    ProcessMap   processes_;
    SystemMap    systems_;
    mutable EntityMap*   entities_;
    Variable*    sizeVariable_;
    bool         entityListChanged_;
};

template<>
inline System::VariablesCRange
System::getBelongings<Variable>() const
{
    return VariablesCRange( variables_.begin(), variables_.end() );
}

template<>
inline System::ProcessesCRange
System::getBelongings<Process>() const
{
    return ProcessesCRange( processes_.begin(), processes_.end() );
}

template<>
inline System::SystemsCRange
System::getBelongings<System>() const
{
    return SystemsCRange( systems_.begin(), systems_.end() );
}

inline System::EntitiesCRange
System::getBelongings() const
{
    if ( !entities_ )
    {
        entities_ = new EntityMap( EntityMap::range_list_type(
                const_cast<VariableMap&>(variables_),
                const_cast<ProcessMap&>(processes_),
                const_cast<SystemMap&>(systems_) ) );
    }

    return EntitiesCRange( entities_->begin(), entities_->end() );
}

template<>
inline System::VariablesRange
System::getBelongings<Variable>()
{
    return VariablesRange( variables_.begin(), variables_.end() );
}

template<>
inline System::ProcessesRange
System::getBelongings<Process>()
{
    return ProcessesRange( processes_.begin(), processes_.end() );
}

template<>
inline System::SystemsRange
System::getBelongings<System>()
{
    return SystemsRange( systems_.begin(), systems_.end() );
}

inline System::EntitiesRange
System::getBelongings()
{
    if ( !entities_ )
    {
        entities_ = new EntityMap( EntityMap::range_list_type(
                variables_, processes_, systems_ ) );
    }

    return EntitiesRange( entities_->begin(), entities_->end() );
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
