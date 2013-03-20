//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2012 Keio University
//       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

#include <boost/range/iterator_range.hpp>

#include "libecs/Defs.hpp"
#include "libecs/Entity.hpp"

namespace libecs
{

class Variable;
class Process;
class Stepper;

LIBECS_DM_CLASS( System, Entity )
{
public:
    // Maps used for entry lists
    typedef std::map< String, Variable*, std::less< String > > VariableMap;
    typedef std::map< String, Process*, std::less< String > > ProcessMap;
    typedef std::map< String, System*, std::less< String > > SystemMap;

    typedef boost::iterator_range< VariableMap::iterator > Variables;
    typedef boost::iterator_range< ProcessMap::iterator > Processes;
    typedef boost::iterator_range< SystemMap::iterator > Systems;

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

    virtual EntityType getEntityType() const
    {
        return EntityType( EntityType::SYSTEM );
    }

    /**
       Pre-initializes this system prior to preinitialize().
    */
    virtual void prepreinitialize();

    /**
       Pre-initializes this system. 
    */
    virtual void preinitialize();

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
    boost::iterator_range< typename std::map< String, T_*, std::less< String > >::iterator > getEntities() const;

    Variables getVariables() const
    {
        return Variables(
            const_cast< System* >( this )->theVariableMap.begin(),
            const_cast< System* >( this )->theVariableMap.end() );
    }

    Processes getProcesses() const
    {
        return Processes(
            const_cast< System* >( this )->theProcessMap.begin(),
            const_cast< System* >( this )->theProcessMap.end() );
    }

    Systems getSystems() const
    {
        return Systems(
            const_cast< System* >( this )->theSystemMap.begin(),
            const_cast< System* >( this )->theSystemMap.end() );
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
    System* getSystem( SystemPath const& anID ) const;


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
       Register a Variable object to this System.

       This method steals ownership of the given pointer.
    */
    void registerEntity( Variable* aVariable );


    /**
       Register a Process object to this System.

       This method steals ownership of the given pointer.
    */
    void registerEntity( Process* aProcess );


    /**
       Register a System object to this System.

       This method steals ownership of the given pointer.
    */
    void registerEntity( System* aSystem );


    /**
       Register an Entity object to this System.

       This method steals ownership of the given pointer.
     */
    void registerEntity( Entity* anEntity );


    /**
       Unregister the specified Entity object from this System.
     */
    void unregisterEntity( Entity* anEntity );


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
    virtual SystemPath getSystemPath() const;

    Variable const* getSizeVariable() const;

    void notifyChangeOfEntityList();

    Variable const* findSizeVariable() const;

    void configureSizeVariable();

    virtual void detach();

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
inline System::Variables System::getEntities< Variable >() const
{
    return getVariables();
}

template <>
inline System::Processes System::getEntities< Process >() const
{
    return getProcesses();
}

template <>
inline System::Systems   System::getEntities< System >() const
{
    return getSystems();
}

} // namespace libecs

#endif /* __SYSTEM_HPP */
