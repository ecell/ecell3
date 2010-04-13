//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell System
//
//       Copyright (C) 1996-2010 Keio University
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

#ifndef __ENTITY_HPP
#define __ENTITY_HPP

#include "libecs/Defs.hpp"
#include "libecs/EntityType.hpp"
#include "libecs/EcsObject.hpp"
#include "libecs/PropertyInterface.hpp"
#include "libecs/LoggerBroker.hpp"

namespace libecs
{
    class System;

    /**
       Entity class is a base class for all components in the cell model.
    */
    LIBECS_DM_CLASS( Entity, EcsObject )
    {
        friend class LoggerBroker;
    public:

        LIBECS_DM_OBJECT_ABSTRACT( Entity ) 
        {
            INHERIT_PROPERTIES( EcsObject );
            PROPERTYSLOT_SET_GET( String, Name );
        }

        Entity(); 
        virtual ~Entity();

        /**
           Get a System to which this Entity belongs.

           @return a borrowed pointer to the super system.
        */
        System* getSuperSystem() const 
        {
            return theSuperSystem;
        }

        /**
           Get a FullID of this Entity.

           @return a FullID of this Entity.
        */
        FullID getFullID() const;

        /**
           Get EntityType of this Entity.

           This method is overridden in Variable, Process and System classes.

           @return EntityType of this Entity object.
           @see EntityType
        */
        virtual EntityType getEntityType() const
        {
            return EntityType( EntityType::ENTITY );
        }

        /**
           Get a SystemPath of this Entity.

           @note The SystemPath doesn't include ID of this Entity even if 
           this Entity is a System.

           @return a SystemPath of this Entity.
        */
        virtual SystemPath getSystemPath() const;


        /**
           @name Properties
           @{
         */

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


        /**
           Get a string representation of this Entity as String.

           @return a description string of this Entity.
        */
        virtual String asString() const;
        /** @} */


        /**
           Get loggers associated to this Entity
         */
        LoggerBroker::LoggersPerFullID getLoggers() const;

        /**
           @internal

           Set a supersystem of this Entity.    

           Usually no need to set this manually because a System object does
           this when an Entity is added to the System.

           @param supersystem a pointer to a System to which this object belongs.
        */
        void setSuperSystem( System* supersystem ) 
        { 
            theSuperSystem = supersystem; 
        }

        /**
          Detach this entity from Model
        */
        virtual void detach();

    protected:
    
        void setLoggerMap( LoggerBroker::PerFullIDMap* anLoggerMap )
        {
            theLoggerMap = anLoggerMap;
        }

    private:

        // hide them
        Entity( Entity& );
        Entity& operator=( Entity& );

    private:
        System*                        theSuperSystem;
        LoggerBroker::PerFullIDMap*    theLoggerMap;
        String                         theID;
        String                         theName;
    };

} // namespace libecs

#endif /* __ENTITY_HPP */
