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

#ifndef __PROPERTIEDCLASS_HPP
#define __PROPERTIEDCLASS_HPP

#include "dmtool/DMObject.hpp"

#include "libecs.hpp"

namespace libecs
{

  //
  // Macros for DM class definition.
  //

#define LIBECS_DM_CLASS( CLASSNAME, BASE )\
  DECLARE_CLASS( CLASSNAME );\
  class CLASSNAME\
   :\
  public BASE 

//,\
//  private PropertyInterface<CLASSNAME>


#define LIBECS_DM_OBJECT_ABSTRACT( CLASSNAME )\
  LIBECS_DM_OBJECT_DEF( CLASSNAME );\
  LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME );\
  LIBECS_DM_DEFINE_PROPERTIES()


#define LIBECS_DM_OBJECT( CLASSNAME, DMTYPE )\
  DM_OBJECT( CLASSNAME, DMTYPE );\
  LIBECS_DM_OBJECT_DEF( CLASSNAME );\
  LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME );\
  LIBECS_DM_DEFINE_PROPERTIES()


#define LIBECS_DM_BASECLASS( CLASSNAME )\
  DM_BASECLASS( CLASSNAME )     


#define LIBECS_DM_INIT( CLASSNAME, DMTYPE )\
  DM_INIT( CLASSNAME, DMTYPE )\
  LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )


#define LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )\
  libecs::PropertyInterface<CLASSNAME>::PropertySlotMap\
    libecs::PropertyInterface<CLASSNAME>::thePropertySlotMap;\
  libecs::PropertyInterface<CLASSNAME>\
    CLASSNAME::thePropertyInterface

//  static PropertyInitializer<CLASSNAME> a ## CLASSNAME ## PropertyInitializer

  ///@internal
#define LIBECS_DM_OBJECT_DEF( CLASSNAME )\
  typedef CLASSNAME _LIBECS_CLASS_;\
  virtual StringLiteral getClassName() const { return XSTR( CLASSNAME ); } //

  ///@internal
#define LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME )\
private:\
 static PropertyInterface<CLASSNAME> thePropertyInterface;\
public:\
 virtual PropertySlotBasePtr getPropertySlot( StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.getPropertySlot( aPropertyName );\
 }\
 virtual void setProperty( StringCref aPropertyName, PolymorphCref aValue )\
 {\
  thePropertyInterface.setProperty( *this, aPropertyName, aValue );\
 }\
 virtual const Polymorph getProperty( StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.getProperty( *this, aPropertyName );\
 }\
 virtual void loadProperty( StringCref aPropertyName, PolymorphCref aValue )\
 {\
  thePropertyInterface.loadProperty( *this, aPropertyName, aValue );\
 }\
 virtual const Polymorph saveProperty( StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.saveProperty( *this, aPropertyName );\
 }\
 virtual const Polymorph getPropertyList() const\
 {\
  return thePropertyInterface.getPropertyList();\
 }\
 virtual PropertySlotProxyPtr\
 createPropertySlotProxy( StringCref aPropertyName )\
 {\
  return thePropertyInterface.createPropertySlotProxy( *this, aPropertyName );\
 } //


  //
  // Macros for property definitions
  //

#define INHERIT_PROPERTIES( BASECLASS )\
    BASECLASS::initializeProperties( Type2Type<TT>() )

#define PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD )\
  PropertyInterface<TT>::registerPropertySlot( # NAME,\
         new ConcretePropertySlot<TT,TYPE>( SETMETHOD, GETMETHOD ) );

#define PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD,\
				LOADMETHOD, SAVEMETHOD )\
  PropertyInterface<TT>::registerPropertySlot( # NAME,\
         new LoadSaveConcretePropertySlot<TT,TYPE>( SETMETHOD, GETMETHOD,\
						    LOADMETHOD, SAVEMETHOD ) )

#define PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD )\
        PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD,\
				NULLPTR, NULLPTR )

#define PROPERTYSLOT_SET_GET( TYPE, NAME )\
  PROPERTYSLOT( TYPE, NAME,\
                       & _LIBECS_CLASS_::set ## NAME,\
                       & _LIBECS_CLASS_::get ## NAME )

#define PROPERTYSLOT_SET( TYPE, NAME )\
  PROPERTYSLOT( TYPE, NAME,\
                       & _LIBECS_CLASS_::set ## NAME,\
                       NULLPTR )


#define PROPERTYSLOT_GET( TYPE, NAME )\
  PROPERTYSLOT( TYPE, NAME,\
                       NULLPTR,\
                       & _LIBECS_CLASS_::get ## NAME )

#define PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( TYPE, NAME )\
  PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME,\
                             & _LIBECS_CLASS_::set ## NAME,\
                             & _LIBECS_CLASS_::get ## NAME )

#define PROPERTYSLOT_SET_NO_LOAD_SAVE( TYPE, NAME )\
  PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME,\
                             & _LIBECS_CLASS_::set ## NAME,\
                             NULLPTR )

#define PROPERTYSLOT_GET_NO_LOAD_SAVE( TYPE, NAME )\
  PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME,\
                             NULLPTR,\
                             & _LIBECS_CLASS_::get ## NAME )


  ///@internal
#define LIBECS_DM_DEFINE_PROPERTIES()\
  template<class TT>\
  static void initializeProperties( Type2Type<TT> )\



  // 
  // Macros for property method declaration / definitions.
  //


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


  /** @addtogroup property The Inter-object Communication.
   *  The Interobject Communication.
   *@{

  */

  /** @file */
  
  // probably better to replace by AssocVector.
  //  DECLARE_MAP( const String, PropertySlotPtr, 
  //	       std::less<const String>, PropertySlotMap );

  /**
     Common base class for classes with PropertySlots.

     @see PropertySlot

  */

  class PropertiedClass
  {

  public:

    LIBECS_DM_DEFINE_PROPERTIES()
    {
      ; // empty, but this must be here.
    }


    PropertiedClass()
    {
      ; // do nothing
    }

    virtual ~PropertiedClass()
    {
      ; // do nothing
    }

    virtual PropertySlotBasePtr 
    getPropertySlot( StringCref aPropertyName ) const = 0;

    virtual void 
    setProperty( StringCref aPropertyName, PolymorphCref aValue ) = 0;

    virtual const Polymorph 
    getProperty( StringCref aPropertyName ) const = 0;

    virtual void 
    loadProperty( StringCref aPropertyName, PolymorphCref aValue ) = 0;

    virtual const Polymorph 
    saveProperty( StringCref aPropertyName ) const = 0;

    virtual const Polymorph getPropertyList() const = 0;

    virtual void defaultSetProperty( StringCref aPropertyName, 
				     PolymorphCref aValue );

    virtual const Polymorph 
    defaultGetProperty( StringCref aPorpertyName ) const;

    const Polymorph 
    getPropertyAttributes( StringCref aPropertyName ) const;
    
    void registerLogger( LoggerPtr aLogger );

    void removeLogger( LoggerPtr aLogger );

    LoggerVectorCref getLoggerVector() const
    {
      return theLoggerVector;
    }

    const String getClassNameString() const { return getClassName(); }

    virtual StringLiteral getClassName() const = 0;


  protected:

    /// @internal

    template <typename Type>
    void nullSet( const Type& )
    {
      throwNotSetable();
    }

    /// @internal

    template <typename Type>
    const Type nullGet() const
    {
      throwNotGetable();
    }

  private:

    static void throwNotSetable();
    static void throwNotGetable();

  protected:

    LoggerVector theLoggerVector;

  };


#define NULLSET_SPECIALIZATION( TYPE )\
  template <> void PropertiedClass::nullSet<TYPE>( const TYPE& )

  NULLSET_SPECIALIZATION( Real );
  NULLSET_SPECIALIZATION( Int );
  NULLSET_SPECIALIZATION( String );
  NULLSET_SPECIALIZATION( Polymorph );

#define NULLGET_SPECIALIZATION( TYPE )\
  template <> const TYPE PropertiedClass::nullGet<TYPE>() const

  NULLGET_SPECIALIZATION( Real );
  NULLGET_SPECIALIZATION( Int );
  NULLGET_SPECIALIZATION( String );
  NULLGET_SPECIALIZATION( Polymorph );


  /*@}*/
  
} // namespace libecs

#endif /* __PROPERTIEDCLASS_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
