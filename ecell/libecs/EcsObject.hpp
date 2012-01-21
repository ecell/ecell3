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

#include "libecs/PropertyAttributes.hpp"

#ifndef __ECSOBJECT_HPP
#define __ECSOBJECT_HPP

#include "libecs/Defs.hpp"
#include "libecs/Handle.hpp"

#include "dmtool/DMObject.hpp"

class DynamicModuleInfo;

namespace libecs
{

/**
   Define a EcsObject
   @param CLASSNAME the name of the class to declare.
   @param BASE      the base class
 */
#define LIBECS_DM_CLASS( CLASSNAME, BASE )\
    class DM_IF CLASSNAME: public BASE

/**
   Define a EcsObject with one extra interface
   @param CLASSNAME the name of the class to declare.
   @param BASE      the base class
 */
#define LIBECS_DM_CLASS_EXTRA_1( CLASSNAME, BASE, IF1 )\
    class DM_IF CLASSNAME: public BASE, public IF1

/**
   Define a EcsObject with a mix-in class
   @param CLASSNAME the name of the class to declare.
   @param BASE      the base class
   @param MIXIN     the class mixed-in to CLASSNAME
 */
#define LIBECS_DM_CLASS_MIXIN( CLASSNAME, BASE, MIXIN )\
    class DM_IF CLASSNAME: public BASE, public MIXIN< CLASSNAME >


/**
   @internal
 */
#define LIBECS_DM_INIT_PROP_INTERFACE()\
    static void initializePropertyInterface( libecs::PropertyInterfaceBase* aPropertyInterface )

/**
   Defines the property interface initializer that should be previously
   declared in LIBECS_DM_OBJECT macro.

   @param CLASSNAME
 */
#define LIBECS_DM_INIT_PROP_INTERFACE_DEF( CLASSNAME )\
    void CLASSNAME::initializePropertyInterface( libecs::PropertyInterfaceBase* aPropertyInterface )



/**
   Put in front of the property definition block of an abstract EcsObject
   @param CLASSNAME the name of the abstract EcsObject class
 */
#define LIBECS_DM_OBJECT_ABSTRACT( CLASSNAME )\
    LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME )\
    LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME )\
    LIBECS_DM_INIT_PROP_INTERFACE()


/**
   Put in front of the property definition block of a mix-in for an EcsObject
   @param TEMPLATENAME the name of the template to be mixed in.
   @param BASE the name of the EcsObject class for the template to mix in.
 */
#define LIBECS_DM_OBJECT_MIXIN( TEMPLATENAME, BASE )\
    LIBECS_DM_OBJECT_DEF_ABSTRACT( BASE );\
    typedef TEMPLATENAME _LIBECS_MIXIN_CLASS_; \
    LIBECS_DM_INIT_PROP_INTERFACE()


/**
   Put in front of the property definition block of an EcsObject
   @param CLASSNAME the name of the EcsObject class
 */
#define LIBECS_DM_OBJECT( CLASSNAME, DMTYPE )\
    DM_OBJECT( CLASSNAME );\
    LIBECS_DM_OBJECT_DEF( CLASSNAME, DMTYPE );\
    LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME );\
    LIBECS_DM_INIT_PROP_INTERFACE()


/**
   Used when defining a root class (Process, Variable, System, Stepper)
   @internal
   @param CLASSNAME the name of the EcsObject class
 */
#define LIBECS_DM_BASECLASS( CLASSNAME )\
    DM_BASECLASS( CLASSNAME )         


/**
   Define the class stub.
   @param CLASSNAME the name of the EcsObject class
   @param DMTYPE    the root class of the class to declare.
                    (Process, Variable, System, Stepper)
 */
#define LIBECS_DM_INIT( CLASSNAME, DMTYPE )\
    DM_INIT( CLASSNAME )\
    LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE )


/**
   Defines methods for the class stub.

   This macro does two things:
   (1) template specialization of PropertyInterface for CLASSNAME
       is necessary at this point to instantiate thePropertySlotMap static
       variable.
   (2) thePropertyInterface static variable must also be instantiated.

   @param CLASSNAME the name of the EcsObject class
   @param DMTYPE    the root class of the class to declare.
                    (Process, Variable, System, Stepper)
 */
#define LIBECS_DM_INIT_STATIC( CLASSNAME, DMTYPE ) \
    template class libecs::PropertyInterface< CLASSNAME >;\
    char CLASSNAME::thePropertyInterface[ sizeof( libecs::PropertyInterface<CLASSNAME> ) ]; \
    char CLASSNAME::theDMTypeName[] = #DMTYPE; \
    libecs::PropertyInterface< CLASSNAME >& CLASSNAME::_getPropertyInterface() \
    { \
        return *reinterpret_cast< libecs::PropertyInterface< CLASSNAME >* >( thePropertyInterface ); \
    } \
    void CLASSNAME::initializeModule() \
    { \
        new(thePropertyInterface) libecs::PropertyInterface<CLASSNAME>( #CLASSNAME, theDMTypeName ); \
    } \
    void CLASSNAME::finalizeModule() \
    { \
        reinterpret_cast< libecs::PropertyInterface<CLASSNAME>* >( thePropertyInterface )->~PropertyInterface<CLASSNAME>(); \
    } \
    libecs::PropertySlotBase const* CLASSNAME::getPropertySlot( libecs::String const& aPropertyName ) const \
    { \
        return _getPropertyInterface().getPropertySlot( aPropertyName ); \
    } \
    void CLASSNAME::setProperty( libecs::String const& aPropertyName, libecs::Polymorph const& aValue ) \
    { \
        return _getPropertyInterface().setProperty( *this, aPropertyName, aValue ); \
    } \
    libecs::Polymorph CLASSNAME::getProperty( libecs::String const& aPropertyName ) const \
    { \
        return _getPropertyInterface().getProperty( *this, aPropertyName ); \
    } \
    void CLASSNAME::loadProperty( libecs::String const& aPropertyName, libecs::Polymorph const& aValue ) \
    { \
        return _getPropertyInterface().loadProperty( *this, aPropertyName, aValue ); \
    } \
    libecs::Polymorph CLASSNAME::saveProperty( libecs::String const& aPropertyName ) const \
    { \
        return _getPropertyInterface().saveProperty( *this, aPropertyName ); \
    } \
    std::vector< libecs::String > CLASSNAME::getPropertyList() const \
    { \
        return _getPropertyInterface().getPropertyList( *this ); \
    } \
    libecs::PropertySlotProxy* CLASSNAME::createPropertySlotProxy( libecs::String const& aPropertyName ) \
    { \
        return _getPropertyInterface().createPropertySlotProxy( *this, aPropertyName ); \
    } \
    libecs::PropertyAttributes \
    CLASSNAME::getPropertyAttributes( libecs::String const& aPropertyName ) const \
    { \
        return _getPropertyInterface().getPropertyAttributes( *this, aPropertyName ); \
    } \
    libecs::PropertyInterfaceBase const& CLASSNAME::getPropertyInterface() const \
    {\
        return _getPropertyInterface(); \
    } \
    DynamicModuleInfo const* CLASSNAME::getClassInfoPtr()\
    {\
        return static_cast<const DynamicModuleInfo*>( &_getPropertyInterface() );\
    }

/**
   Expanded to typedefs of the class itself (_LUBECS_CLASS_) and the root class
   (_LIBECS_DMTYPE_).
   @param CLASSNAME the name of the class to declare.
   @param DMTYPE    the root class of the class to declare.
                    (Process, Variable, System, Stepper)
 */ 
#define LIBECS_DM_OBJECT_DEF( CLASSNAME, DMTYPE )\
    typedef DMTYPE _LIBECS_DMTYPE_;\
    LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME )

/**
   Expanded to a typedef of the class itself (_LUBECS_CLASS_).
   @param CLASSNAME the name of the class to declare.
 */ 
#define LIBECS_DM_OBJECT_DEF_ABSTRACT( CLASSNAME )\
    typedef CLASSNAME _LIBECS_CLASS_;

/**
   Declare methods for the class stub.
   @param CLASSNAME the name of the EcsObject class
 */
#define LIBECS_DM_EXPOSE_PROPERTYINTERFACE( CLASSNAME )\
private:\
    static char thePropertyInterface[sizeof(libecs::PropertyInterface<CLASSNAME>)]; \
    static char theDMTypeName[]; \
    static libecs::PropertyInterface<CLASSNAME>& _getPropertyInterface(); \
public:\
    static void initializeModule(); \
    static void finalizeModule(); \
    static DynamicModuleInfo const* getClassInfoPtr(); \
    virtual libecs::PropertySlotBase const* getPropertySlot( libecs::String const& aPropertyName ) const; \
    virtual void setProperty( libecs::String const& aPropertyName, libecs::Polymorph const& aValue ); \
    virtual libecs::Polymorph getProperty( libecs::String const& aPropertyName ) const; \
    virtual void loadProperty( libecs::String const& aPropertyName, libecs::Polymorph const& aValue ); \
    virtual libecs::Polymorph saveProperty( libecs::String const& aPropertyName ) const; \
    virtual std::vector< libecs::String > getPropertyList() const; \
    virtual libecs::PropertySlotProxy* createPropertySlotProxy( libecs::String const& aPropertyName ); \
    virtual libecs::PropertyAttributes getPropertyAttributes( libecs::String const& aPropertyName ) const; \
    virtual libecs::PropertyInterfaceBase const& getPropertyInterface() const;

/**
   Used within a property definition block to inherit the properties of the
   base class.
   @param BASECLASS the name of the base class from which the properties are
                    inherited.
 */
#define INHERIT_PROPERTIES( BASECLASS )\
        BASECLASS::initializePropertyInterface( aPropertyInterface );\
        CLASS_INFO( "Baseclass", # BASECLASS )


/**
   Used within a property definition block to define a class info entry
   that describes the function of the class.
   @param DESCRIPTION the description of the class.
 */
#define CLASS_DESCRIPTION( DESCRIPTION )\
        CLASS_INFO( "Description", DESCRIPTION )

#define NOMETHOD 0

#define CLASSINFO_TRUE 1
#define CLASSINFO_FALSE 0

/**
   @internal
 */
#define METHODFLAG( METHODPTR, NULLVALUE ) \
    METHODFLAG2( METHODPTR, NULLVALUE )

/**
   @internal
 */
#define METHODFLAG2( METHODPTR, NULLVALUE ) \
    ( ( METHODPTR ) == NULLVALUE ? CLASSINFO_FALSE : CLASSINFO_TRUE )

/**
   Define a class info entry.
   @param FIELDNAME the name of the field.
   @param FIELDVALUE the value of the field.
*/
#define CLASS_INFO( FIELDNAME, FIELDVALUE) \
    aPropertyInterface->setInfoField( FIELDNAME, libecs::Polymorph( FIELDVALUE ) )


/** 
   Expanded to the function pointer of a getter function.
   @param TYPE the type of the property.
   @param PTR  the pointer to the function.
*/
#define PROPERTYSLOT_GET_METHOD_PTR( TYPE, PTR ) \
    static_cast< typename libecs::ConcretePropertySlot< _LIBECS_CLASS_, TYPE >::GetMethodPtr>( PTR )


/** 
   Expanded to the function pointer of a setter function.
   @param TYPE the type of the property.
   @param PTR  the pointer to the function.
*/
#define PROPERTYSLOT_SET_METHOD_PTR( TYPE, PTR ) \
    static_cast< typename libecs::ConcretePropertySlot< _LIBECS_CLASS_, TYPE >::SetMethodPtr>( PTR )


/** 
   Expanded to the function pointer of a save handler function.
   @param TYPE the type of the property.
   @param PTR  the pointer to the function.
*/
#define PROPERTYSLOT_SAVE_METHOD_PTR( TYPE, PTR ) \
    static_cast< typename libecs::ConcretePropertySlot< _LIBECS_CLASS_, TYPE >::GetMethodPtr>( PTR )


/** 
   Expanded to the function pointer of a load handler function.
   @param TYPE the type of the property.
   @param PTR  the pointer to the function.
*/
#define PROPERTYSLOT_LOAD_METHOD_PTR( TYPE, PTR ) \
    static_cast< typename libecs::ConcretePropertySlot< _LIBECS_CLASS_, TYPE >::SetMethodPtr>( PTR )


/**
   Define a property slot.

   @param TYPE the type of the property.
   @param NAME the name of the property.
   @param SETMETHOD the pointer to the setter.
   @param GETMETHOD the pointer to the getter.
 */
#define PROPERTYSLOT( TYPE, NAME, SETMETHOD, GETMETHOD )\
    aPropertyInterface->registerPropertySlot( \
        new libecs::ConcretePropertySlot< _LIBECS_CLASS_ ,TYPE >( \
            #NAME, SETMETHOD, GETMETHOD ) );

/**
   Define a property slot.

   @param TYPE the type of the property.
   @param NAME the name of the property.
   @param SETMETHOD the pointer to the setter.
   @param GETMETHOD the pointer to the getter.
   @param LOADMETHOD the pointer to the load handler.
   @param SAVEMETHOD the pointer to the save handler.
 */
#define PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD, \
                                LOADMETHOD, SAVEMETHOD ) \
    aPropertyInterface->registerPropertySlot( \
         new LoadSaveConcretePropertySlot< _LIBECS_CLASS_, TYPE >( \
            #NAME, SETMETHOD, GETMETHOD, LOADMETHOD, SAVEMETHOD ) );


/**
   Define a property slot that are not savable nor loadable.

   @param TYPE the type of the property.
   @param NAME the name of the property.
   @param SETMETHOD the pointer to the setter.
   @param GETMETHOD the pointer to the getter.
 */
#define PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD )\
    PROPERTYSLOT_LOAD_SAVE( TYPE, NAME, SETMETHOD, GETMETHOD, \
                            0, 0 )

/**
   Define a property slot such that its getter and setter functions are
   Class::getNAME() or Class::setNAME().

   @param TYPE the type of the property.
   @param NAME the name of the property.
 */
#define PROPERTYSLOT_SET_GET( TYPE, NAME )\
    PROPERTYSLOT( TYPE, NAME, & _LIBECS_CLASS_::set ## NAME, \
                              & _LIBECS_CLASS_::get ## NAME )

/**
   Define a write-only property slot such that its setter function is
   Class::setNAME().

   @param TYPE the type of the property.
   @param NAME the name of the property.
 */
#define PROPERTYSLOT_SET( TYPE, NAME )\
    PROPERTYSLOT( TYPE, NAME, & _LIBECS_CLASS_::set ## NAME, 0 )


/**
   Define a read-only property slot such that its getter function is
   Class::getNAME().

   @param TYPE the type of the property.
   @param NAME the name of the property.
 */
#define PROPERTYSLOT_GET( TYPE, NAME )\
    PROPERTYSLOT( TYPE, NAME, 0, & _LIBECS_CLASS_::get ## NAME )


/**
   Define a property slot such that its getter and setter functions are
   Class::getNAME() and Class::setName(), and is not savable nor loadable.

   @param TYPE the type of the property.
   @param NAME the name of the property.
 */
#define PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( TYPE, NAME )\
    PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME, & _LIBECS_CLASS_::set ## NAME,\
                                           & _LIBECS_CLASS_::get ## NAME )

/**
   Define a write-only property slot such that its setter function is
   Class::setName(), and is not savable nor loadable.

   @param TYPE the type of the property.
   @param NAME the name of the property.
 */
#define PROPERTYSLOT_SET_NO_LOAD_SAVE( TYPE, NAME )\
    PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME, & _LIBECS_CLASS_::set ## NAME, 0 )


/**
   Define a read-only property slot such that its getter function is
   Class::getName(), and is not savable nor loadable.

   @param TYPE the type of the property.
   @param NAME the name of the property.
 */
#define PROPERTYSLOT_GET_NO_LOAD_SAVE( TYPE, NAME )\
    PROPERTYSLOT_NO_LOAD_SAVE( TYPE, NAME, 0, &_LIBECS_CLASS_::get ## NAME )



// 
// Macros for property method declaration / definitions.
//
/**
   Expand to a function signature that can be used both for declaration and
   definition of a setter method.

   @param TYPE the type of the slot.
   @param METHODNAME the name of the method.
 */
#define SET_SLOT( TYPE, METHODNAME )\
    void METHODNAME( libecs::Param<TYPE>::type value )

/**
   Expand to a function signature that can be used both for declaration and
   definition of a getter method.

   @param TYPE the type of the slot.
   @param METHODNAME the name of the method.
 */
#define GET_SLOT( TYPE, METHODNAME )\
    TYPE METHODNAME() const

/**
   Expand to the method definition starter of a setter.

   @param TYPE the type of the slot.
   @param METHODNAME the name of the method.
   @param CLASS the class name that contains the method.
 */
#define SET_SLOT_DEF( TYPE, METHODNAME, CLASS )\
    SET_SLOT( TYPE, CLASS::METHODNAME )

/**
   Expand to the method definition starter of a getter.

   @param TYPE the type of the slot.
   @param METHODNAME the name of the method.
   @param CLASS the class name that contains the method.
 */
#define GET_SLOT_DEF( TYPE, METHODNAME, CLASS )\
    GET_SLOT( TYPE, CLASS::METHODNAME )

/**
   Expand to the signature of the setter function whose name is NAME
   prefixed with "set".

   @param TYPE the type of the slot.
   @param NAME the name of the slot.
 */
#define SET_METHOD( TYPE, NAME )\
    SET_SLOT( TYPE, set ## NAME )

/**
   Expand to the signature of the getter function whose name is NAME
   prefixed with "get".

   @param TYPE the type of the slot.
   @param NAME the name of the slot.
 */
#define GET_METHOD( TYPE, NAME )\
    GET_SLOT( TYPE, get ## NAME )

/**
   Expand to the signature of the setter function whose name is NAME
   prefixed with "load".

   @param TYPE the type of the slot.
   @param NAME the name of the slot.
 */
#define LOAD_METHOD( TYPE, NAME )\
    SET_SLOT( TYPE, load ## NAME )

/**
   Expand to the signature of the getter function whose name is NAME
   prefixed with "save".

   @param TYPE the type of the slot.
   @param NAME the name of the slot.
 */
#define SAVE_METHOD( TYPE, NAME )\
    GET_SLOT( TYPE, save ## NAME )


/**
   Expand to the method definition starter of a setter whose name is
   NAME prefixed with "set".

   @param TYPE the type of the slot.
   @param NAME the name of the method.
   @param CLASS the class name that contains the method.
 */
#define SET_METHOD_DEF( TYPE, NAME, CLASS )\
    SET_SLOT_DEF( TYPE, set ## NAME, CLASS )

/**
   Expand to the method definition starter of the getter whose name is
   NAME prefixed with "get"

   @param TYPE the type of the slot.
   @param NAME the name of the method.
   @param CLASS the class name that contains the method.
 */
#define GET_METHOD_DEF( TYPE, NAME, CLASS )\
    GET_SLOT_DEF( TYPE, get ## NAME, CLASS )

/**
   Expand to the method definition starter of a setter whose name is
   NAME prefixed with "load".

   @param TYPE the type of the slot.
   @param NAME the name of the method.
   @param CLASS the class name that contains the method.
 */
#define LOAD_METHOD_DEF( TYPE, NAME, CLASS )\
    SET_SLOT_DEF( TYPE, load ## NAME, CLASS )

/**
   Expand to the method definition starter of the getter whose name is
   NAME prefixed with "save"

   @param TYPE the type of the slot.
   @param NAME the name of the method.
   @param CLASS the class name that contains the method.
 */
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

template< typename T > class PropertyInterface;
class PropertyInterfaceBase;

/**
   Common base class for classes with PropertySlots.

   @see PropertySlot
*/

class LIBECS_API EcsObject
{
public:

    LIBECS_DM_INIT_PROP_INTERFACE()
    {
        ; // empty, but this must be here.
    }


    EcsObject()
        : theModel( 0 ), theHandle( Handle::INVALID_HANDLE_VALUE ),
          disposed_( false )
    {
        ; // do nothing
    }

    virtual ~EcsObject()
    {
        dispose();
    }

    virtual void dispose();

    virtual void detach();

    /**
       Get a Model object to which this object belongs.

       @return a borrowed pointer to the Model.
    */
    Model* getModel() const
    {
        return theModel;
    }

    /**
       Associate a Model object with this object.

       @internal
       @param aModel the Model object to associate with this object.
     */
    void setModel( Model* const aModel )
    {
        theModel = aModel;
    }

    /**
       Return true if the object is already disposed.
       @return true if the object is disposed, false otherwise.
     */
    bool isDisposed() const
    {
        return disposed_;
    }

    /**
       Return the PropertySlot for aPropertyName

       @param aPropertyName the name of the property.
       @return the pointer to the PropertySlot.
     */
    virtual PropertySlotBase const* 
    getPropertySlot( String const& aPropertyName ) const = 0;


    /**
       Set the value of the property specified by aPropertyName
       to aValue.

       @param aPropertyName the name of the property.
       @param aValue the value to set the property to.
       @return the value of the property.
     */
    virtual void 
    setProperty( String const& aPropertyName, Polymorph const& aValue ) = 0;


    /**
       Return the Polymorph'ed value for aPropertyName

       @param aPropertyName the name of the property.
       @return the value of the property.
     */
    virtual Polymorph 
    getProperty( String const& aPropertyName ) const = 0;

    virtual void 
    loadProperty( String const& aPropertyName, Polymorph const& aValue ) = 0;

    virtual Polymorph saveProperty( String const& aPropertyName ) const = 0;

    virtual std::vector< String > getPropertyList() const = 0;

    virtual PropertyAttributes
    getPropertyAttributes( String const& aPropertyName ) const = 0;

    virtual PropertyInterfaceBase const& getPropertyInterface() const = 0; 
    virtual void defaultSetProperty( String const& aPropertyName, 
                                     Polymorph const& aValue );
    
    virtual Polymorph defaultGetProperty( String const& aPorpertyName ) const;

    virtual std::vector< String > defaultGetPropertyList() const;

    virtual PropertyAttributes
    defaultGetPropertyAttributes( String const& aPropertyName ) const;

    LIBECS_DEPRECATED String const& getClassName() const;

    virtual String asString() const;

    Handle const& getHandle() const
    {
        return theHandle;
    }

    /// @internal
    void setHandle( Handle const& aHandle )
    {
        theHandle = aHandle;
    } 

    /// @internal
    template <typename Type>
    void nullSet( typename Param< Type >::type );

    /// @internal
    template <typename Type>
    Type nullGet() const;

private:

    void throwNotSetable() const;
    void throwNotGetable() const;

protected:
    Model*   theModel;
    Handle   theHandle;
    bool     disposed_;
};

template <typename Type>
inline void EcsObject::nullSet( typename Param< Type >::type )
{
    throwNotSetable();
}

template <typename Type>
inline Type EcsObject::nullGet() const
{
    throwNotGetable();
}


// these specializations of nullSet/nullGet are here to avoid spreading
// inline copies of them around.    This reduces sizes of DM .so files a bit.

#define NULLGETSET_SPECIALIZATION( TYPE )\
    template <> LIBECS_API void EcsObject::nullSet<TYPE>( Param< TYPE >::type ); \
    template <> LIBECS_API TYPE EcsObject::nullGet<TYPE>() const

NULLGETSET_SPECIALIZATION( Real );
NULLGETSET_SPECIALIZATION( Integer );
NULLGETSET_SPECIALIZATION( String );
NULLGETSET_SPECIALIZATION( Polymorph );

#undef NULLGETSET_SPECIALIZATION

} // namespace libecs

#endif /* __ECSOBJECT_HPP */
