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
#include "Exceptions.hpp"

//
// Macros for DM class definition.
//

#define LIBECS_DM_CLASS( CLASSNAME, BASE )\
DECLARE_CLASS( CLASSNAME );\
typedef BASE __LIBECS_BASECLASSOF_ ## CLASSNAME; \
class DM_IF CLASSNAME: public BASE 


#define LIBECS_DM_OBJECT_ABSTRACT( CLASSNAME )\
  LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME );\
  LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME );\
  LIBECS_DM_DEFINE_PROPERTIES()


#define LIBECS_DM_OBJECT( CLASSNAME, DMTYPE )\
  LIBECS_DM_OBJECT_DEF( CLASSNAME, DMTYPE );\
  LIBECS_DM_DEFINE_PROPERTIES()


#define LIBECS_DM_BASECLASS( CLASSNAME )

#define LIBECS_DM_INIT( CLASSNAME, DMTYPE )\
  DM_INIT( CLASSNAME )\
  LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )


  // This macro does two things:
  // (1) template specialization of PropertyInterface for CLASSNAME
  //     is necessary here to instantiate thePropertySlotMap static variable.
  // (2) thePropertyInterface static variable must also be instantiated.

#define LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )\
  libecs::ConcretePropertyInterface<CLASSNAME> \
      CLASSNAME::thePropertyInterface( \
          #CLASSNAME, \
          ::libecs::Type2PropertiedClassKind< ::libecs::DMTYPE >::value )

  ///@internal
#define LIBECS_DM_OBJECT_DEF( CLASSNAME, DMTYPE )\
  typedef DMTYPE _LIBECS_DMTYPE_;\
  LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME ); \
  LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME ); \
  DM_OBJECT( CLASSNAME );

#define LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME )\
  typedef __LIBECS_BASECLASSOF_ ## CLASSNAME _LIBECS_BASE_CLASS_; \
  typedef CLASSNAME _LIBECS_CLASS_; \
  typedef DynamicModuleBase< CLASSNAME > Module;

  ///@internal
#define LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME )\
private:\
    static ::libecs:: ConcretePropertyInterface<CLASSNAME> thePropertyInterface;\
public:\
    virtual const ::libecs::PropertyInterface& getPropertyInterface() const { \
       return thePropertyInterface; \
    } \
    static const void* getClassInfo( const std::string& kind )\
    {\
        if ( kind == "PropertyInterface" ) { \
            return &thePropertyInterface; \
        } \
        return 0; \
    }\
public:\
  CLASSNAME( const Module& mod ): _LIBECS_BASE_CLASS_( reinterpret_cast< const _LIBECS_BASE_CLASS_::Module &>( mod ) ) {}
//
// Macros for property definitions
//

#define INHERIT_PROPERTIES( BASECLASS )\
    CLASS_INFO( "Baseclass", # BASECLASS )
    
#define CLASS_DESCRIPTION( DESCRIPTION )\
    CLASS_INFO( "Description", DESCRIPTION )

#define NOMETHOD NULLPTR

#define CLASSINFO_TRUE 1
#define CLASSINFO_FALSE 0

/**
 macro for setting class Info string
 Info is expected as PropertyName, Value both Strings
 Property descriptor strings 
 */

#define CLASS_INFO( FIELDNAME, FIELDVALUE) \
    thePropertyInterface.setInfoField( \
        ::libecs::String( FIELDNAME ), ::libecs::String( FIELDVALUE ) )

#define PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD,\
				LOADMETHOD, SAVEMETHOD )\
  thePropertyInterface.registerPropertySlot( \
         new ::libecs::ConcretePropertySlot<_LIBECS_CLASS_,TYPE>( \
            ::libecs::String( #NAME ), \
            ( ::libecs::ConcretePropertySlot<_LIBECS_CLASS_, TYPE>::SetMethod ) SETMETHOD, \
            ( ::libecs::ConcretePropertySlot<_LIBECS_CLASS_, TYPE>::GetMethod ) GETMETHOD, \
            ( ::libecs::ConcretePropertySlot<_LIBECS_CLASS_, TYPE>::LoadMethod ) LOADMETHOD, \
            ( ::libecs::ConcretePropertySlot<_LIBECS_CLASS_, TYPE>::SaveMethod ) SAVEMETHOD  ) )


#define PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD ) \
    PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD, \
                            NULLPTR, NULLPTR )

#define PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD )\
    PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD,\
                            &::libecs::PropertiedClass::nullLoad, \
                            &::libecs::PropertiedClass::nullSave )

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
  static void initializePropertyInterface( ::libecs::PropertyInterface& thePropertyInterface )

// 
// Macros for property method declaration / definitions.
//

#define SET_SLOT( TYPE, METHODNAME )\
  void METHODNAME( ::libecs::Param<TYPE>::type value )

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

/** @addtogroup property The Inter-object Communication.
 *  The Interobject Communication.
 *@{
 */

namespace libecs {

/** @file */

class PropertyInterface;  

/**
   Common base class for classes with PropertySlots.

   @see PropertySlot

*/

class LIBECS_API PropertiedClass
{
public:
    typedef void _LIBECS_BASE_CLASS_;
    typedef PropertyInterface ConcretePropertyInterface;
    typedef DynamicModuleBase<PropertiedClass> Module;

public:
    LIBECS_DM_DEFINE_PROPERTIES();

    PropertiedClass( const Module& mod )
        : theModule( mod ), __libecs_ready( false )
    {
      ; // do nothing
    }

    virtual ~PropertiedClass();

    virtual void __libecs_init__();

    PropertySlot* getPropertySlot( const String& aPropertyName ) const;

    /**
       Set a value of a property slot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName the name of the property.
       @param aValue the value to set as a Polymorph.
       @throw NoSlot
    */
    void setProperty( const String& aPropertyName, const Polymorph& aValue );

    /**
       Get a property value from this object via a PropertySlot.

       This method checks if the property slot exists, and throws
       NoSlot exception if not.

       @param aPropertyName the name of the property.
       @return the value as a Polymorph.
       @throw NoSlot
    */
    const Polymorph getProperty( const String& aPropertyName ) const;

    void loadProperty( const String& aPropertyName, const Polymorph& aValue );

    Polymorph saveProperty( const String& aPropertyName ) const;

    virtual void defaultSetProperty( const String& aPropertyName, 
				     const Polymorph& aValue );
    
    virtual const Polymorph 
    defaultGetProperty( const String& aPorpertyName ) const;
    
    virtual const Polymorph defaultGetPropertyList() const;
    
    virtual const Polymorph 
    defaultGetPropertyAttributes( const String& aPropertyName ) const;

    virtual const PropertyInterface& getPropertyInterface() const;

    void registerLogger( LoggerPtr aLogger );

    void removeLogger( LoggerPtr aLogger );

    const LoggerVector& getLoggerVector() const
    {
        return theLoggerVector;
    }

    const String& getClassName() const;

    static void initializePropertyInterface( const PropertyInterface& )
    {
        ; // do nothing
    }

    PropertySlotProxy createPropertySlotProxy( const String& aPropertyName );

public:
    /// @internal

    template <typename Type>
    void nullSet( typename Param<Type>::type );

    /// @internal
    template <typename Type>
    const Type nullGet() const;

    // @internal
    void nullLoad( Param<Polymorph>::type );

    /// @internal
    const Polymorph nullSave() const;

    virtual const String& asString() const;

    void throwException( const Exception& exc )
    {
        Exception modified(
            exc.getMethod(),
            asString() + ": " + exc.getMessage() );
        const_cast< Exception & >( exc ).swap( modified );
        throw exc;
    }

protected:
    static ConcretePropertyInterface thePropertyInterface;
    LoggerVector theLoggerVector;
    const Module& theModule;
    bool __libecs_ready;
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
  
} // namespace libecs

/*@}*/

#include "PropertyInterface.hpp"

#endif /* __PROPERTIEDCLASS_HPP */

/*
  Do not modify
  $Author$
  $Revision$
  $Date$
  $Locker$
*/
