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
// modify it under the terms of the GNU General Public // License as published by the Free Software Foundation; either
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

#ifndef __PROPERTIEDCLASS_HPP
#define __PROPERTIEDCLASS_HPP

#include "libecs.hpp"

#if defined( WIN32 )
// a bit hackish, but works.
class LIBECS_API ModuleMaker;
#endif /* WIN32 */

#include "dmtool/DMObject.hpp"

namespace libecs
{

  //
  // Macros for DM class definition.
  //

#define LIBECS_DM_CLASS( CLASSNAME, BASE )\
  DECLARE_CLASS( CLASSNAME );\
  class DM_IF CLASSNAME\
   :\
  public BASE 


#define LIBECS_DM_OBJECT_ABSTRACT( CLASSNAME )\
  LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME );\
  LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME );\
  LIBECS_DM_DEFINE_PROPERTIES()


#define LIBECS_DM_OBJECT( CLASSNAME, DMTYPE )\
  DM_OBJECT( CLASSNAME, DMTYPE );\
  LIBECS_DM_OBJECT_DEF( CLASSNAME, DMTYPE );\
  LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME );\
  LIBECS_DM_DEFINE_PROPERTIES()


#define LIBECS_DM_BASECLASS( CLASSNAME )\
  DM_BASECLASS( CLASSNAME )     


#define LIBECS_DM_INIT( CLASSNAME, DMTYPE )\
  DM_INIT( CLASSNAME, DMTYPE )\
  LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )


  // This macro does two things:
  // (1) template specialization of PropertyInterface for CLASSNAME
  //     is necessary here to instantiate thePropertySlotMap static variable.
  // (2) thePropertyInterface static variable must also be instantiated.

#define LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )\
  template class libecs::PropertyInterface<CLASSNAME>;\
  libecs::PropertyInterface<CLASSNAME> CLASSNAME::thePropertyInterface

  ///@internal
#define LIBECS_DM_OBJECT_DEF( CLASSNAME, DMTYPE )\
  typedef DMTYPE _LIBECS_DMTYPE_;\
  LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME )

#define LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME )\
  typedef CLASSNAME _LIBECS_CLASS_;\
  virtual libecs::StringLiteral getClassName() const { return XSTR( CLASSNAME ); } //

  ///@internal
#define LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME )\
private:\
 static libecs::PropertyInterface<CLASSNAME> thePropertyInterface;\
public:\
 virtual libecs::PropertySlotBasePtr getPropertySlot( libecs::StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.getPropertySlot( aPropertyName );\
 }\
 virtual void setProperty( libecs::StringCref aPropertyName, libecs::PolymorphCref aValue )\
 {\
  thePropertyInterface.setProperty( *this, aPropertyName, aValue );\
 }\
 virtual const libecs::Polymorph getProperty( libecs::StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.getProperty( *this, aPropertyName );\
 }\
 virtual void loadProperty( libecs::StringCref aPropertyName, libecs::PolymorphCref aValue )\
 {\
  thePropertyInterface.loadProperty( *this, aPropertyName, aValue );\
 }\
 virtual const libecs::Polymorph saveProperty( libecs::StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.saveProperty( *this, aPropertyName );\
 }\
 virtual const libecs::Polymorph getPropertyList() const\
 {\
  return thePropertyInterface.getPropertyList( *this );\
 }\
 virtual libecs::PropertySlotProxyPtr\
 createPropertySlotProxy( libecs::StringCref aPropertyName )\
 {\
  return thePropertyInterface.createPropertySlotProxy( *this, aPropertyName );\
 }\
 virtual const libecs::Polymorph\
 getPropertyAttributes( libecs::StringCref aPropertyName ) const\
 {\
  return thePropertyInterface.getPropertyAttributes( *this, aPropertyName );\
 } \
static libecs::PolymorphMapCref getClassInfo( void )\
{\
  return thePropertyInterface.getInfoMap();\
}\
static const void* getClassInfoPtr()\
{\
  return reinterpret_cast<const void*>(&thePropertyInterface.getInfoMap());\
}//


  //
  // Macros for property definitions
  //

#define INHERIT_PROPERTIES( BASECLASS )\
    BASECLASS::initializePropertyInterface( libecs::Type2Type<TT>() );\
    CLASS_INFO( "Baseclass", # BASECLASS )
    
#define CLASS_DESCRIPTION( DESCRIPTION )\
    CLASS_INFO( "Description", DESCRIPTION )

#define NOMETHOD NULLPTR

#define CLASSINFO_TRUE 1
#define CLASSINFO_FALSE 0

#define METHODFLAG( METHODPTR, NULLVALUE ) \
 METHODFLAG2( METHODPTR, NULLVALUE )

#define METHODFLAG2( METHODPTR, NULLVALUE ) \
 ( # METHODPTR  == # NULLVALUE ? CLASSINFO_FALSE : CLASSINFO_TRUE )
  /**
	 macro for setting class Info string
	 Info is expected as PropertyName, Value both Strings
	Property descriptor strings 
  */

#define CLASS_INFO( FIELDNAME, FIELDVALUE) \
 libecs::PropertyInterface<TT>::setInfoField( \
  libecs::String( FIELDNAME ), \
  libecs::String( FIELDVALUE ) )


  /** 
	  macro for setting Property class info
	 PropertyName, Type, set_flag, get_flag, save_flag, load_flag
  */
#define CLASSPROPERTY_INFO( PROPERTYNAME, TYPE, SETMETHOD, GETMETHOD, SAVEMETHOD, LOADMETHOD ) \
 libecs::PropertyInterface<TT>::setPropertyInfoField( \
  libecs::String( PROPERTYNAME ), \
  libecs::String( libecs::PropertiedClass::TypeName<TYPE>::name ), \
  METHODFLAG(SETMETHOD, NULLPTR ), \
  METHODFLAG( GETMETHOD, NULLPTR ), \
  METHODFLAG( SAVEMETHOD, NULLPTR ), \
  METHODFLAG( LOADMETHOD, NULLPTR ) )




#define PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD )\
  libecs::PropertyInterface<TT>::registerPropertySlot( # NAME,\
         new libecs::ConcretePropertySlot<TT,TYPE>( SETMETHOD, GETMETHOD ) );\
CLASSPROPERTY_INFO( # NAME, TYPE, SETMETHOD, GETMETHOD, SETMETHOD, GETMETHOD )

#define PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD,\
				LOADMETHOD, SAVEMETHOD )\
  libecs::PropertyInterface<TT>::registerPropertySlot( # NAME,\
         new libecs::LoadSaveConcretePropertySlot<TT,TYPE>( SETMETHOD, GETMETHOD,\
						    LOADMETHOD, SAVEMETHOD ) );\
CLASSPROPERTY_INFO( # NAME, TYPE, SETMETHOD, GETMETHOD, SAVEMETHOD, LOADMETHOD )


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

  // Info


  ///@internal
#define LIBECS_DM_DEFINE_PROPERTIES()\
  template<class TT>\
  static void initializePropertyInterface( libecs::Type2Type<TT> )



  // 
  // Macros for property method declaration / definitions.
  //

#define SET_SLOT( TYPE, METHODNAME )\
  void METHODNAME( libecs::Param<TYPE>::type value )

#define GET_SLOT( TYPE, METHODNAME )\
  const TYPE METHODNAME() const

#define SET_SLOT_DEF( TYPE, METHODNAME, CLASS )\
  SET_SLOT( TYPE, CLASS::METHODNAME )

#define GET_SLOT_DEF( TYPE, METHODNAME, CLASS )\
  GET_SLOT( TYPE, CLASS::METHODNAME )


#define SET_METHOD( TYPE, NAME )\
  SET_SLOT( TYPE, set ## NAME )

#define GET_METHOD( TYPE, NAME )\
  GET_SLOT( TYPE, get ## NAME )

#define LOAD_METHOD( TYPE, NAME )\
  SET_SLOT( TYPE, load ## NAME )

#define SAVE_METHOD( TYPE, NAME )\
  GET_SLOT( TYPE, save ## NAME )


#define SET_METHOD_DEF( TYPE, NAME, CLASS )\
  SET_SLOT_DEF( TYPE, set ## NAME, CLASS )

#define GET_METHOD_DEF( TYPE, NAME, CLASS )\
  GET_SLOT_DEF( TYPE, get ## NAME, CLASS )

#define LOAD_METHOD_DEF( TYPE, NAME, CLASS )\
  SET_SLOT_DEF( TYPE, load ## NAME, CLASS )

#define SAVE_METHOD_DEF( TYPE, NAME, CLASS )\
  GET_SLOT_DEF( TYPE, save ## NAME, CLASS )



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

  template<typename T> class PropertyInterface;

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

  class LIBECS_API PropertiedClass
  {
  public:
    template<typename T> struct TypeName
    {
      static const char name[];
    };

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

    virtual const Polymorph 
    getPropertyAttributes( StringCref aPropertyName ) const = 0;

    virtual void defaultSetProperty( StringCref aPropertyName, 
				     PolymorphCref aValue );
    
    virtual const Polymorph 
    defaultGetProperty( StringCref aPorpertyName ) const;
    
    virtual const Polymorph defaultGetPropertyList() const;
    
    virtual const Polymorph 
    defaultGetPropertyAttributes( StringCref aPropertyName ) const;

    void registerLogger( LoggerPtr aLogger );

    void removeLogger( LoggerPtr aLogger );

    LoggerVectorCref getLoggerVector() const
    {
      return theLoggerVector;
    }

    const String getClassNameString() const { return getClassName(); }

    virtual StringLiteral getClassName() const = 0;

  public:

    /// @internal

    template <typename Type>
    void nullSet( typename Param<Type>::type )
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



  // these specializations of nullSet/nullGet are here to avoid spreading
  // inline copies of them around.  This reduces sizes of DM .so files a bit.

#define NULLSET_SPECIALIZATION( TYPE )\
  template <> LIBECS_API void libecs::PropertiedClass::nullSet<TYPE>( Param<TYPE>::type )

  NULLSET_SPECIALIZATION( Real );
  NULLSET_SPECIALIZATION( Integer );
  NULLSET_SPECIALIZATION( String );
  NULLSET_SPECIALIZATION( Polymorph );

#define NULLGET_SPECIALIZATION( TYPE )\
  template <> LIBECS_API const TYPE libecs::PropertiedClass::nullGet<TYPE>() const

  NULLGET_SPECIALIZATION( Real );
  NULLGET_SPECIALIZATION( Integer );
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
