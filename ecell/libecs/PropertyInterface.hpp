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

#include "libecs.hpp"

#include "Defs.hpp"
#include "Message.hpp"


namespace libecs
{

  /** \defgroup property The Inter-object Communication.
   *  The Interobject Communication.
   *@{
  */


  
  DECLARE_MAP( const String, PropertySlotPtr, 
	       std::less<const String>, PropertySlotMap );

  /**
     Common base class for classes which receive Messages.

     \note  Subclasses of PropertyInterface MUST call their own makeSlots()
     to create their property slots in their constructors.
     (virtual functions don't work in constructors)

     \todo class-static slots?

     \see Message
     \see PropertySlot

  */

  class PropertyInterface
  {
  public:

    enum PropertyAttribute
      {
	SETABLE =    ( 1 << 0 ),
	GETABLE =    ( 1 << 1 ),
	CUMULATIVE = ( 1 << 2 )
      };


    PropertyInterface();
    virtual ~PropertyInterface();


    /**
       Send a message to this object via a PropertySlot.

       \param aMessage a Message object
       \throw NoSlot 
    */

    void setMessage( MessageCref aMessage );

    /**
       Get a message from this object via a PropertySlot.

       \param aPropertyName a name of the PropertySlot.
       \return the Message from this object.
       \throw NoSlot
    */

    const Message getMessage( StringCref aPropertyName ) const;


    /**
       Get a PropertySlot by name.

       \param aPropertyName the name of the PropertySlot.

       \return a borrowed pointer to the PropertySlot with the name.
    */

    virtual PropertySlotPtr getPropertySlot( StringCref aPropertyName )
    {
      return thePropertySlotMap[ aPropertyName ];
    }


    /**
       Create and register PropertySlots of this object.

       This method should be defined in subclasses, if it has new
       PropertySlots defined.  

       This method must be called from the constructor only once.
    */

    virtual void makeSlots();


    /** \name Properties

    Properties is a group of methods which can be accessed via (1)
    PropertySlots and (2) set/getMessage() methods, in addition to
    normal C++ method calls.

    */
    //@{

    const UVariableVectorRCPtr getPropertyList() const;
    const UVariableVectorRCPtr getPropertyAttributes() const;

    //@}



    /// \internal 

    //FIXME: can be protected?

    PropertySlotMapCref getPropertySlotMap() const 
    {
      return thePropertySlotMap;
    }

    const String getClassNameString() const { return getClassName(); }

    virtual StringLiteral getClassName() const { return "PropertyInterface"; }


    /// \internal

    template <typename Type>
    void nullSet( const Type& )
    {
      THROW_EXCEPTION( AttributeError, "Not setable." );
    }

    /// \internal

    template <typename Type>
    const Type nullGet() const
    {
      THROW_EXCEPTION( AttributeError, "Not getable." );
    }

  protected:

    static PropertySlotMakerPtr getPropertySlotMaker();

    void registerSlot( PropertySlotPtr );
    void removeSlot( StringCref keyword );

  private:

    PropertySlotMap thePropertySlotMap;

  };

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
