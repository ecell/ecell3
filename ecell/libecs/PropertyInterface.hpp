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

//#include <map>

#include "AssocVector.h"

#include "libecs.hpp"
#include "PropertySlot.hpp"
#include "PropertySlotProxy.hpp"
#include "PropertiedClass.hpp"

namespace libecs
{

  /** @addtogroup property The Inter-object Communication.
   *  The Interobject Communication.
   *@{

  */

  /** @file */

  class PropertyInterfaceBase
  {
  public:

    ~PropertyInterfaceBase()
    {
      ; // do nothing
    }

    static void throwNoSlot( StringCref aClassName, StringCref aPropertyName );

  protected:

    PropertyInterfaceBase()
    {
      ; // do nothing
    }

  };


  template
  <
    class T
  >
  class PropertyInterface
    :
    public PropertyInterfaceBase
  {

  public:

    typedef PropertySlot<T> PropertySlot_;
    DECLARE_TYPE( PropertySlot_, PropertySlot );

    DECLARE_ASSOCVECTOR_TEMPLATE( String, PropertySlot*,
				  std::less<const String>, PropertySlotMap );

    PropertyInterface()
    {
      T::initializeProperties( Type2Type<T>() );
    }

    ~PropertyInterface()
    {
      // This object is never deleted.
      /*
	for( PropertySlotMapIterator i( thePropertySlotMap.begin() ); 
	i != thePropertySlotMap.end() ; ++i )
	{
	delete i->second;
	}
      */
    }

    /**
       Get a PropertySlot by name.

       @param aPropertyName the name of the PropertySlot.

       @return a borrowed pointer to the PropertySlot with that name.
    */

    static PropertySlotPtr getPropertySlot( StringCref aPropertyName )
    {
      PropertySlotMapConstIterator i( findPropertySlot( aPropertyName ) );

      if( i == thePropertySlotMap.end() )
	{
	  throwNoSlot( "This class", aPropertyName );
	}

      return i->second;
    }

    static PropertySlotProxyPtr 
    createPropertySlotProxy( T& anObject,
			     StringCref aPropertyName )
    {
      try
	{
	  PropertySlotPtr aPropertySlot( getPropertySlot( aPropertyName ) );
	  return new ConcretePropertySlotProxy<T>( anObject, *aPropertySlot );
	}
      catch( NoSlotCref )
	{
	  throwNoSlot( anObject.getClassName(), aPropertyName );
	}
    }


    /**
       Set a value of a property slot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName the name of the property.
       @param aValue the value to set as a Polymorph.
       @throw NoSlot 
    */

    static void setProperty( T& anObject, StringCref aPropertyName, 
			     PolymorphCref aValue )
    {
      PropertySlotMapConstIterator 
	aPropertySlotMapIterator( findPropertySlot( aPropertyName ) );
      
      if( aPropertySlotMapIterator != thePropertySlotMap.end() )
	{
	  aPropertySlotMapIterator->second->setPolymorph( anObject, aValue );
	}
      else
	{
	  anObject.defaultSetProperty( aPropertyName, aValue );
	}
    }
    

    /**
       Get a property value from this object via a PropertySlot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName the name of the property.
       @return the value as a Polymorph.
       @throw NoSlot
    */

    static const Polymorph getProperty( const T& anObject,
					StringCref aPropertyName )
    {
      PropertySlotMapConstIterator 
	aPropertySlotMapIterator( findPropertySlot( aPropertyName ) );
      
      if( aPropertySlotMapIterator != thePropertySlotMap.end() )
	{
	  return aPropertySlotMapIterator->second->getPolymorph( anObject );
	}
      else
	{
	  return anObject.defaultGetProperty( aPropertyName );
	}
    }


    static void loadProperty( T& anObject, StringCref aPropertyName, 
			      PolymorphCref aValue )
    {
      getPropertySlot( aPropertyName )->loadPolymorph( anObject, aValue );
    }
    
    static const Polymorph saveProperty( const T& anObject,
					 StringCref aPropertyName )
    {
      return getPropertySlot( aPropertyName )->savePolymorph( anObject );
    }

    static const Polymorph getPropertyList()
    {
      PolymorphVector aVector;
      // aVector.reserve( thePropertySlotMap.size() );
      
      for( PropertySlotMapConstIterator i( thePropertySlotMap.begin() ); 
	   i != thePropertySlotMap.end() ; ++i )
	{
	  aVector.push_back( i->first );
	}
      
      return aVector;
    }

    
    static void 
    registerPropertySlot( StringCref aName, PropertySlotPtr aPropertySlotPtr )
    {
      if( findPropertySlot( aName ) != thePropertySlotMap.end() )
	{
	  // it already exists. take the latter one.
	  delete thePropertySlotMap[ aName ];
	  thePropertySlotMap.erase( aName );
	}

      thePropertySlotMap[ aName ] = aPropertySlotPtr;
    }


    /*
    static void removePropertySlot( StringCref aName );
    {
      if( thePropertySlotMap.find( aName ) == thePropertySlotMap.end() )
	{
	  THROW_EXCEPTION( NoSlot,
			   getClassName() + String( ":no slot for keyword [" ) +
			   aName + String( "] found.\n" ) );
	}
      
      delete thePropertySlotMap[ aName ];
      thePropertySlotMap.erase( aName );
    }
    */

    static PropertySlotMapCref getPropertySlotMap()
    {
      return thePropertySlotMap;
    }


  private:

    static PropertySlotMapConstIterator 
    findPropertySlot( StringCref aPropertyName )
    {
      return thePropertySlotMap.find( aPropertyName );
    }

  private:

    static PropertySlotMap        thePropertySlotMap;

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
