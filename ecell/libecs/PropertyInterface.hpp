//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//        This file is part of E-CELL Simulation Environment package
//
//                Copyright (C) 1996-2001 Keio University
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

#include "Message.hpp"
#include "PropertySlot.hpp"


namespace libecs
{

  DECLARE_MAP( const String, PropertySlotPtr, 
	       std::less<const String>, PropertyMap );

  /**
     Common base class for classes which receive Messages.

     NOTE:  Subclasses of PropertyInterface MUST call their own makeSlots()
     to create their property slots in their constructors.
     (virtual functions don't work in constructors)

     FIXME: class-static slots?

     @see Message
     @see PropertySlot
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

    void setMessage( MessageCref );
    const Message getMessage( StringCref ) const;

    PropertyMapCref getPropertySlotMap() const 
    {
      return thePropertyMap;
    }

    virtual void makeSlots();

    virtual const char* const className() const { return "PropertyInterface"; }

  public: // message slots

    const UVariableVectorRCPtr getPropertyList() const;
    const UVariableVectorRCPtr getPropertyAttributes() const;


    /**

     createPropertySlot template method provides a standard way 
     to create a new slot.  It is template so that it can accept methods
     of class T (the template parameter class).

    */

    template<class T>
    void
    createPropertySlot( StringCref name,
			T& object,
			ClassPropertySlot<T>::SetUVariableVectorMethodPtr set,
			ClassPropertySlot<T>::GetUVariableVectorMethodPtr get )
    {
      appendSlot( new UVariableVectorPropertySlot<T>( name, 
						      object, 
						      set, 
						      get ) );
    }

    template<class T>
    void
    createPropertySlot( StringCref name,
			T& object,
			ClassPropertySlot<T>::SetRealMethodPtr set,
			ClassPropertySlot<T>::GetRealMethodPtr get )
    {
      appendSlot( new RealPropertySlot<T>( name, 
					   object, 
					   set, 
					   get ) );
    }

    template<class T>
    void
    createPropertySlot( StringCref name,
			T& object,
			ClassPropertySlot<T>::SetStringMethodPtr set,
			ClassPropertySlot<T>::GetStringMethodPtr get )
    {
      appendSlot( new StringPropertySlot<T>( name, 
					     object, 
					     set, 
					     get ) );
    }

    void appendSlot( PropertySlotPtr );
    void deleteSlot( StringCref keyword );

  private:

    PropertyMap thePropertyMap;

  };
  




} // namespace libecs

#endif /* __PROPERTYINTERFACE_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
