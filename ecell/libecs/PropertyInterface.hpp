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

#ifndef __PROPERTYINTERFACE_HPP
#define __PROPERTYINTERFACE_HPP

#include <map>

#include "dmtool/DMObject.hpp"

#include "libecs.hpp"
#include "Defs.hpp"
#include "Exceptions.hpp"

namespace libecs
{

  /** @addtogroup property The Inter-object Communication.
   *  The Interobject Communication.
   *@{

  */

  /** @file */
  

  // probably better to replace by AssocVector.
  DECLARE_MAP( const String, PropertySlotPtr, 
	       std::less<const String>, PropertySlotMap );

  /**
     Common base class for classes with PropertySlots.

     Properties is a group of methods which can be accessed via (1)
     PropertySlots and (2) set/getProperty() methods, in addition to
     normal C++ method calls.

     @note  Subclasses of PropertyInterface MUST call their own makeSlots()
     to create their property slots in their constructors.
     (virtual functions don't work in constructors)

     @todo class-static slots?

     @see PropertySlot

  */

  class PropertyInterface
  {
  public:

    PropertyInterface();
    virtual ~PropertyInterface();


    /**
       Set a value of a property slot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName a name of the property.
       @param aValue the value to set as a PolymorphVector.
       @throw NoSlot 
    */

    void setProperty( StringCref aPropertyName, PolymorphCref aValue );

    /**
       Get a property value from this object via a PropertySlot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName a name of the property.
       @return the value as a PolymorphVector.
       @throw NoSlot
    */

    const Polymorph getProperty( StringCref aPropertyName ) const;


    /**
       Get a PropertySlot by name.

       @param aPropertyName the name of the PropertySlot.

       @return a borrowed pointer to the PropertySlot with the name.
    */

    virtual PropertySlotPtr getPropertySlot( StringCref aPropertyName ) const
    {
      return getPropertySlotMap().find( aPropertyName )->second;
    }



    const Polymorph getPropertyList() const;

    const Polymorph getPropertyAttributes( StringCref aPropertyName ) const;


    /// @internal 

    //FIXME: can be a protected member?

    PropertySlotMapCref getPropertySlotMap() const 
    {
      return thePropertySlotMap;
    }

    const String getClassNameString() const { return getClassName(); }

    virtual StringLiteral getClassName() const { return "PropertyInterface"; }


    /// @internal

    template <typename Type>
    void nullSet( const Type& )
    {
      THROW_EXCEPTION( AttributeError, "Not setable." );
    }

    /// @internal

    template <typename Type>
    const Type nullGet() const
    {
      THROW_EXCEPTION( AttributeError, "Not getable." );
    }

  protected:

    static PropertySlotMakerPtr getPropertySlotMaker();

    void registerSlot( StringCref aName, PropertySlotPtr aPropertySlotPtr );
    void removeSlot( StringCref aName );

  private:

    PropertySlotMap thePropertySlotMap;
    
    //    LoggerVector theLoggerVector;

  };


#define GET_METHOD( TYPE, NAME )\
const TYPE get ## NAME() const

#define SET_METHOD( TYPE, NAME )\
void set ## NAME( TYPE ## Cref value )

#define GET_METHOD_DEF( TYPE, NAME, CLASS )\
const TYPE CLASS::get ## NAME() const

#define SET_METHOD_DEF( TYPE, NAME, CLASS )\
void CLASS::set ## NAME( TYPE ## Cref value )


#define SIMPLE_GET_METHOD( TYPE, NAME )\
GET_METHOD( TYPE, NAME )\
{\
  return NAME;\
} //

#define SIMPLE_SET_METHOD( TYPE, NAME )\
SET_METHOD( TYPE, NAME )\
{\
  NAME = value;\
} //

#define SIMPLE_SET_GET_METHOD( TYPE, NAME )\
SIMPLE_SET_METHOD( TYPE, NAME )\
SIMPLE_GET_METHOD( TYPE, NAME )

#define ECELL_DM_OBJECT\
 StringLiteral getClassname() { return XSTR( _ECELL_CLASSNAME ); }\
 static _ECELL_TYPE* createInstance() { return new _ECELL_CLASSNAME ; }


#define LIBECS_DM_OBJECT( TYPE, CLASSNAME )\
public:\
 StringLiteral getClassname() { return XSTR( CLASSNAME ); }\
DM_OBJECT( TYPE, CLASSNAME )

#define DEFINE_PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD )\
CREATE_PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD )\

#define CREATE_PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD )\
    registerSlot( # NAME,\
		  getPropertySlotMaker()->\
		  createPropertySlot( *this, Type2Type<TYPE>(),\
				      SETMETHOD,\
				      GETMETHOD\
				      ) )

#define CREATE_PROPERTYSLOT_SET_GET( TYPE, NAME, CLASS )\
CREATE_PROPERTYSLOT( TYPE, NAME,\
                     & CLASS::set ## NAME,\
                     & CLASS::get ## NAME )

#define CREATE_PROPERTYSLOT_SET( TYPE, NAME, CLASS )\
CREATE_PROPERTYSLOT( TYPE, NAME,\
                     & CLASS::set ## NAME,\
                     NULLPTR )


#define CREATE_PROPERTYSLOT_GET( TYPE, NAME, CLASS )\
CREATE_PROPERTYSLOT( TYPE, NAME,\
                     NULLPTR,\
                     & CLASS::get ## NAME )



  /*@}*/
  
} // namespace libecs

#endif /* __PROPERTYINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
