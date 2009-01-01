//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2009 Keio University
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

#ifndef __SYSTEM_HPP
#define __SYSTEM_HPP

#include "libecs/Defs.hpp"
#include "libecs/Entity.hpp"

namespace libecs
{

LIBECS_DM_CLASS( System, Entity )
{
public:
    // Maps used for entry lists
    DECLARE_MAP( const String, Variable*, std::less<const String>, VariableMap );
    DECLARE_MAP( const String, Process*, std::less<const String>, ProcessMap );
    DECLARE_MAP( const String, System*, std::less<const String>, SystemMap );

public:
    LIBECS_DM_BASECLASS( System );

    LIBECS_DM_OBJECT( System, System )
    {
        INHERIT_PROPERTIES( Entity );
        PROPERTYSLOT_SET_GET( String, StepperID );
        PROPERTYSLOT_GET_NO_LOAD_SAVE( Real, Size );
    }

    System();
    virtual ~System();

    virtual void dispose();

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
    Stepper* getStepper() const 
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

    template <class T_>
    const std::map<const String, T_*, std::less<const String> >& getMap() const;

    VariableMapCref getVariableMap() const
    {
        return theVariableMap;
    }

    ProcessMapCref getProcessMap() const
    {
        return theProcessMap;
    }

    SystemMapCref getSystemMap() const
    {
        return theSystemMap;
    }


    /**
       Find a Process with given id in this System.    
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Process object in this System named @a id.
    */
    Process* getProcess( String const& anID ) const;


    /**
       Find a Variable with given id in this System. 
       
       This method throws NotFound exception if it is not found.

       @return a borrowed pointer to a Variable object in this System named @a id.
    */
    Variable* getVariable( String const& anID ) const;

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
    System* getSystem( SystemPathCref anID ) const;


    /**
       Find a System with a given id in this System. 
       
       This method throws NotFound exception if it is not found.

       Unlike getSystem( SystemPath ) method, this method searches only
       within this System.    In the other words this method doesn't 
       conduct a recursive search.

       @param anID An ID string of a System.

       @return a borrowed pointer to a System object in this System
       whose ID is anID.
    */
    System* getSystem( String const& id ) const;


    /**
       Register a Process object to this System.

       This method steals ownership of the given pointer.
    */
    void registerEntity( Process* aProcess );


    /**
       Unregister the specified Process object from this System.
     */
    void unregisterEntity( Process* aProcess );

    /**
       Register a Variable object to this System.

       This method steals ownership of the given pointer.
    */
    void registerEntity( Variable* aVariable );


    /**
       Unregister the specified Process object from this System.
     */
    void unregisterEntity( Variable* aProcess );


    /**
       Register a System object to this System.

       This method steals ownership of the given pointer.
    */
    void registerEntity( System* aSystem );


    /**
       Unregister the specified System object from this System.
     */
    void unregisterEntity( System* aProcess );


    /**
       Register an Entity object to this System.

       This method steals ownership of the given pointer.
     */
    void registerEntity( Entity* anEntity );


    /**
       Unregister the Entity specified by anEntityType and anID
       from this System.
       @param anEntityType The type of the entity,
       @param anID         The ID of the entity.
     */
    void unregisterEntity( EntityType const& anEntityType, String const& anID  );


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

    Variable const* getSizeVariable() const;

    void notifyChangeOfEntityList();

    Variable const* findSizeVariable() const;

    void configureSizeVariable();

public: // property slots
    GET_METHOD( Polymorph, SystemList );
    GET_METHOD( Polymorph, VariableList );
    GET_METHOD( Polymorph, ProcessList );

private:
    void unregisterEntity( SystemMap::iterator const& );

    void unregisterEntity( ProcessMap::iterator const& );

    void unregisterEntity( VariableMap::iterator const& );

protected:
    String          theStepperID;
    Stepper*        theStepper;

private:
    VariableMap     theVariableMap;
    ProcessMap      theProcessMap;
    SystemMap       theSystemMap;

    Variable const* theSizeVariable;
};


template <>
inline System::VariableMapCref System::getMap() const
{
    return getVariableMap();
}

template <>
inline System::ProcessMapCref  System::getMap() const
{
    return getProcessMap();
}

template <>
inline System::SystemMapCref   System::getMap() const
{
    return getSystemMap();
}

} // namespace libecs

#endif /* __SYSTEM_HPP */
