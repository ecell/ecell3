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

#ifndef __MODEL_HPP
#define __MODEL_HPP

#include <map>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/range/iterator_range.hpp>
#include "AssocVector.h"

#include "libecs.hpp"
#include "FullID.hpp"
#include "CastUtils.hpp"
#include "PropertiedClass.hpp"
#include "System.hpp"
#include "RangeConcatenator.hpp"

/** @addtogroup model The Model.

    The model.

    @ingroup libecs
 */
/** @{ */
/** @file */


namespace libecs
{
class SimulationContext;
class Entity;
class Stepper;
class Variable;
class Process;
class PropertiedObjectMaker;

/**
   Model class represents a simulation model.

   Model has a list of Steppers and a pointer to the root system.

*/
LIBECS_DM_CLASS( Model, PropertiedClass )
{
public:
    typedef ::Loki::AssocVector< String, Stepper* > StepperMap;
    typedef ::boost::transform_iterator<
            select2nd< StepperMap::value_type >,
                StepperMap::iterator > StepperIterator;
    typedef ::boost::transform_iterator<
            select2nd< StepperMap::value_type >,
                StepperMap::const_iterator> StepperCIterator;
    typedef ::boost::iterator_range< StepperIterator > SteppersRange;
    typedef ::boost::iterator_range< StepperCIterator > SteppersCRange;

    typedef ::Loki::AssocVector< FullID, Process* > ProcessMap;
    typedef ::boost::iterator_range< ProcessMap::iterator >       ProcessMapRange;
    typedef ::boost::iterator_range< ProcessMap::const_iterator > ProcessMapCRange;

    typedef ::Loki::AssocVector< FullID, System* > SystemMap;
    typedef ::boost::iterator_range< SystemMap::iterator >       SystemMapRange;
    typedef ::boost::iterator_range< SystemMap::const_iterator > SystemMapCRange;

    typedef ::Loki::AssocVector< FullID, Variable* > VariableMap;
    typedef ::boost::iterator_range< VariableMap::iterator >       VariableMapRange;
    typedef ::boost::iterator_range< VariableMap::const_iterator > VariableMapCRange;

    typedef RangeConcatenator< ::boost::mpl::vector<
            VariableMap, ProcessMap, SystemMap >,
            ::std::pair< const LocalID, Entity* > > EntityMap;
    typedef ::boost::iterator_range< EntityMap::iterator >       EntityMapRange;
    typedef ::boost::iterator_range< EntityMap::const_iterator > EntityMapCRange;

private:
    class EntityEventObserver: public System::EntityEventObserver
    {
    public:
        EntityEventObserver() {}

        void setModel( Model* model )
        {
            model_ = model;
        }

        void entityAdded( Descriptor desc );

        void entityRemoved( Descriptor desc );

    private:
        Model* model_;
    };
    friend class EntityEventObserver;

public:
    virtual ~Model();

    virtual void startup();

    virtual void initialize();

    virtual void postInitialize();

    virtual void interrupt( TimeParam time );

    /**
       Creates a new Entity object and register it in an appropriate System
       in  the Model.
       @param className
       @param fullID
       @param aName
    */
    template<typename T_>
    T_* createEntity( const String& className );

    /**
       Adds an entity to the model
     */
    void addEntity( const FullID& fullID, Entity* entity );

    /**
       This method finds an Entity object pointed by the FullID.
       @param fullID a FullID of the requested Entity.
       @return A borrowed pointer to an Entity specified by the FullID.
    */
    const Entity* getEntity( const FullID& fullID, bool throwIfNotFound = true ) const;

    /**
       This method finds an Entity object pointed by the FullID.
       @param fullID a FullID of the requested Entity.
       @return A borrowed pointer to an Entity specified by the FullID.
    */
    Entity* getEntity( const FullID& fullID, bool throwIfNotFound = true );

    /**
       This method finds a System object pointed by the SystemPath.
       @param systemPath a SystemPath of the requested System.
       @return A borrowed pointer to a System.
    */
    System* getSystem( const SystemPath& systemPath, bool throwIfNotFound = true );


    /**
       This method finds a System object pointed by the SystemPath.
       @param systemPath a SystemPath of the requested System.
       @return A borrowed pointer to a System.
    */
    const System* getSystem( const SystemPath& systemPath, bool throwIfNotFound = true ) const;

    /**
       Create a stepper with an ID and a classname.

       @param className  a classname of the Stepper to create.

       @param anID        a Stepper ID string of the Stepper to create.

    */
    Stepper* createStepper( const String& className );

    /**
       Adds a stepper to the model
     */
    void addStepper( const String& id, Stepper* );

    /**
       Get a stepper by an ID.

       @param anID a Stepper ID string of the Stepper to get.
       @return a borrowed pointer to the Stepper.
    */
    Stepper* getStepper( const String& anID );

    /**
       Get a stepper by an ID.

       @param anID a Stepper ID string of the Stepper to get.
       @return a borrowed pointer to the Stepper.
    */
    const Stepper* getStepper( const String& anID ) const;

    /**
       Get the StepperMap of this Model.

       @return the const reference of the StepperMap.
    */
    SteppersCRange getSteppers() const
    {
        return SteppersCRange(
            StepperCIterator(
                steppers_.begin(),
                select2nd<StepperMap::value_type>()),
            StepperCIterator(
                steppers_.end(),
                select2nd<StepperMap::value_type>()));
    }

    /**
       Get the StepperMap of this Model.

       @return the const reference of the StepperMap.
    */
    SteppersRange getSteppers()
    {
        return SteppersRange(
            StepperIterator(
                steppers_.begin(),
                select2nd<StepperMap::value_type>()),
            StepperIterator(
                steppers_.end(),
                select2nd<StepperMap::value_type>()));
    }

    /**
       Get the SystemMap of this Model.

       @return the const reference of the SystemMap.
    */
    SystemMapCRange getSystemMap() const
    {
        return SystemMapCRange( systems_.begin(), systems_.end());
    }

    /**
       Get the EntityMap of this Model.

       @return the const reference of the EntityMap.
    */
    SystemMapRange getSystemMap()
    {
        return SystemMapRange( systems_.begin(), systems_.end());
    }

    /**
       Get the ProcessMap of this Model.

       @return the const reference of the ProcessMap.
    */
    ProcessMapCRange getProcessMap() const
    {
        return ProcessMapCRange( processes_.begin(), processes_.end());
    }

    /**
       Get the ProcessMap of this Model.

       @return the const reference of the ProcessMap.
    */
    ProcessMapRange getProcessMap()
    {
        return ProcessMapRange( processes_.begin(), processes_.end());
    }

    /**
       Get the VariableMap of this Model.

       @return the const reference of the VariableMap.
    */
    VariableMapCRange getVariables() const
    {
        return VariableMapCRange( variables_.begin(), variables_.end());
    }

    /**
       Get the VariableMap of this Model.

       @return the const reference of the VariableMap.
    */
    VariableMapRange getVariables()
    {
        return VariableMapRange( variables_.begin(), variables_.end());
    }

    /**
       Get the VariableMap of this Model.

       @return the const reference of the VariableMap.
    */
    EntityMapCRange getEntities() const
    {
        if ( !entities_ )
        {
            entities_ = new EntityMap( EntityMap::range_list_type(
                    const_cast<VariableMap&>(variables_),
                    const_cast<ProcessMap&>(processes_),
                    const_cast<SystemMap&>(systems_) ) );
        }
        return EntityMapCRange( entities_->begin(), entities_->end() );
    }

    /**
       Get the VariableMap of this Model.

       @return the const reference of the VariableMap.
    */
    EntityMapRange getEntities()
    {
        if ( !entities_ )
        {
            entities_ = new EntityMap( EntityMap::range_list_type(
                    variables_, processes_, systems_ ) );
        }
        return EntityMapRange( entities_->begin(), entities_->end() );
    }

    FullID getFullIDOf( const Entity* ent ) const;

    /**
       Get the RootSystem.

       @return a borrowed pointer to the RootSystem.
    */
    System* getRootSystem() const
    {
        return rootSystem_;
    }

    const SimulationContext* getSimulationContext() const
    {
        return simulationContext_;
    }

    SimulationContext* getSimulationContext()
    {
        return simulationContext_;
    }

    void setSimulationContext( SimulationContext* ctx )
    {
        simulationContext_ = ctx;
    }

private:
    SimulationContext*     simulationContext_;
    System*                rootSystem_;
    StepperMap             steppers_;
    SystemMap              systems_;
    ProcessMap             processes_;
    VariableMap            variables_;
    mutable EntityMap*     entities_;
    PropertiedObjectMaker* propertiedObjectMaker_;
    EntityEventObserver    observer_;
};

} // namespace libecs

#include "PropertiedObjectMaker.hpp"

namespace libecs {

template<typename T_>
inline T_* Model::createEntity( const String& className )
{
    T_* ent( propertiedObjectMaker_->make<T_>( className ) );
    ent->setModel( this );
    ent->startup();
    return ent;
}

} // namespace libecs

/** @}*/


#endif /* __STEPPERLEADER_HPP */
/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
