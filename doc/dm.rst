===========================
Creating New Object Classes
===========================

This section describes how to define your own object classes for use in
the simulation.

About Dynamic Modules
=====================

Dynamic Module (DM) is a file containing an object class, especially C++
class, which can be loaded and instantiated by the application. APP uses
this mechanism to provide users a way of defining and adding new classes
to appear in simulation models without recompiling the whole system.
Because the classes are defined in forms of native codes, this is the
most efficient way of adding a new code or object class in terms of
space and speed.

In APP, subclasses of PROCESS, VARIABLE, SYSTEM and STEPPER classes can
be dynamically loaded by the system.

In addition to standard DMs distributed with APP, user-defined DM files
can be created from C++ source code files ('.cpp' files) with the
``ecell3-dmc`` command. The compiled files usually take a form of shared
library ('.so') files.

Defining a new class
====================

A new object class can be defined by writing a C++ source code file with
some special usage of C++ macros.

Here is a boilarplate template of a DM file, with which you should feel
familiar if you have a C++ experience. Replace ``DMTYPE``,
``CLASSNAME``, and ``BASECLASS`` according to your case.

::

    #include <libecs/libecs.hpp>
    #include <libecs/.hpp>

    USE_LIBECS;

    LIBECS_DM_CLASS( ,  )
    {
    public:
        LIBECS_DM_OBJECT( ,  )
        {
          // ( Property definition of this class comes here. )
        }

        () {}// A constructor without an argument
        () {}// A destructor
    };

    LIBECS_DM_INIT( ,  );

``DMTYPE``, ``CLASSNAME`` and ``BASECLASS``
-------------------------------------------

First of all you have to decide basic attributes of the class you are
going to define; such as a DM type (PROCESS, VARIABLE, SYSTEM, or
STEPPER), a class name, and a base class.

-  ``DMTYPE``

   ``DMTYPE`` is one of DM base classes defined in APP PROCESS, STEPPER,
   VARIABLE, and SYSTEM.

-  ``CLASSNAME``

   ``CLASSNAME`` is a name of the object class.

   This must be a valid C++ class name, and should end with the
   ``DMTYPE`` name. For example, if you are going to define a new
   PROCESS class and want to name it Foo, the class name may look like
   FooProcess.

-  ``BASECLASS``

   The class your class inherits from.

   This may or may not be the same as the ``DMTYPE
       ``, depending on whether it is a direct descendant of the DM base
   class.

Filename
--------

The name of the source file must be the same as the ``CLASSNAME`` with a
trailing '.cpp' suffix. For example, if the ``CLASSNAME`` is FooProcess,
the file name must be ``FooProcess.cpp``.

The source code can be divided into header and source files (such as
``FooProcess.hpp`` and ``FooProcess.cpp``), but at least the
``LIBECS_DM_INIT`` macro must be placed in the source file of the class
(``FooProcess.cpp``).

Include Files
-------------

At least the libecs header file (``libecs/libecs.hpp``) and a header
file of the base class (such as ``libecs/.hpp``) must be included in the
head of the file.

DM Macros
---------

You may notice that the template makes use of some special macros:
``USE_LIBECS``, ``LIBECS_DM_CLASS``, ``LIBECS_DM_OBJECT``, and
``LIBECS_DM_INIT``.

``USE_LIBECS`` declares use of libecs library, which is the core library
of APP, in this file after the line.

``LIBECS_DM_CLASS``

``LIBECS_DM_OBJECT( ,
           )`` should be placed on the top of the class definition part
(immediately after '{' of the class). This macro declares that this is a
DM class. This macro makes it dynamically instantiable, and
automatically defines getClassName() method. Note that this macro
specifies public: field inside, and thus anything comes after this is
placed in public. For clarity it is a good idea to always write public:
explicitly after this macro.

::

     LIBECS_DM_OBJECT( DMTYPE, CLASSNAME )
              public:

``LIBECS_DM_INIT( ,
     )`` exports the class ``CLASSNAME`` as a DM class of type
``DMTYPE``. This must come after the definition (not just a declaration)
of the class to be exported with a ``LIBECS_DM_OBJECT`` call.

Constructor And Destructor
--------------------------

DM objects are always instantiated by calling the constructor with no
argument. The destructor is defined virtual in the base class.

Types And Declarations
----------------------

Basic types
~~~~~~~~~~~

The following four basic types are available to be used in your code if
you included ``libecs/libecs.hpp`` header file and called the
``USE_LIBECS`` macro.

-  ``Real``

   A real number. Usually implemented as a double precision floating
   point number. It is a 64-bit float on Linux/IA32/gcc platform.

-  ``Integer``

   A signed integer number. This is a 64-bit ``long int`` on
   Linux/IA32/gcc.

-  ``UnsignedInteger``

   An unsigned integer number. This is a 64-bit ``unsigned long int`` on
   Linux/IA32/gcc.

-  STRING

   A string equivalent to std::string class of the C++ standard library.

-  POLYMORPH

   POLYMORPH is a sort of universal type (actually a class) which can
   \*become\* and \*be made from\* any of ``Real``, ``Integer``,
   ``String``, and ``PolymorphVector``, which is a mixed list of these
   three types of objects. See the next section for details.

These types are recommended to be used over other C++ standard types
such as ``double``, ``int`` and ``char*``.

Pointer and reference types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each types, the following typedefs are available.

-  ``TYPEPtr``

   Pointer type. (== ``TYPE*``)

-  ``TYPECptr``

   Const pointer type. (== ``const TYPE*``)

-  ``TYPERef``

   Reference type. (== ``TYPE&``)

-  ``TYPECref``

   Const reference type. (== ``const TYPE&``)

For example, ``RealCref`` is equivalent to write ``const Real&``. Using
these typedefs is recommended.

To declare a new type, use ``DECLARE_TYPE`` macro. For example,

::

    DECLARE_TYPE( double, Real );

is called inside the system so that ``RealCref`` can be used as ``const
       double&``.

Similary, DECLARE\_CLASS can be used to enable the typedefs for a class.
Example:

::

    DECLARE_CLASS( Process );

enables ``ProcessCref`` ``ProcessPtr`` etc.. Most classes defined in
libecs have these typedefs.

Limits and other attributes of types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get limits and precisions of these numeric types, use
std::numeric\_limits<> template class in the C++ standard library. For
instance, to get a maximum value that can be represented by the ``Real``
type, use the template class like this:

::

    #include <limits>
    numeric_limits<Real>::max();

See the C++ standard library reference manual for more.

Polymorph class
---------------

A POLYMORPH object can be constructed from and converted to any of
``Real``, ``Integer``, ``String``, types and POLYMORPHVECTOR class.

Construct a Polymorph
~~~~~~~~~~~~~~~~~~~~~

To construct a POLYMORPH object, simply call a constructor with a value:

::

    Polymorph anIntegerPolymorph( 1 );
    Polymorph aRealPolymorph( 3.1 );
    Polymorph aStringPolymorph( "2.13e2" );

A POLYMORPH object can be constructed (or copied) from a POLYMORPH:

::

    Polymorph aRealPolymorph2( aRealPolymorph );

Getting a value of a Polymorph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The value of the POLYMORPH objects can be retrieved in any type by using
as<>() template method.

::

    anIntegerPolymorph.as<Real>();    // == 1.0
    aRealPolymorph.as<String>(); // == "3.1"
    aStringPolymorph.as<Integer>();  // == 213

    **Note**

    If an overflow occurs when converting a very big ``Real`` value to
    ``Integer``, a ValueError exception?? is thrown. (NOT IMPLEMENTED
    YET)

Examining and changing the type of Polymorph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

getType(), changeType()

PolymorphVector
~~~~~~~~~~~~~~~

POLYMORPHVECTOR is a list of POLYMORPH objects.

Other C++ statements
--------------------

The only limitation is the ``DM_INIT`` macro, which exports a class as a
DM class, can appear only once in a compilation unit which forms a
single shared library file.

Except for that, there is no limitation as far as the C++ compiler
understands it. There can be any C++ statements inside and outside of
the class definition including; other class definitions, nested classes,
typedefs, static functions, namespaces, and even template<>.

Be careful, however, about namespace corruptions. You may want to use
private C++ namespaces and static functiont when a class or a function
declared outside the DM class is needed.

PropertySlot
============

What is PropertySlot
--------------------

PROPERTYSLOT is a pair of methods to access (get) and mutate (set) an
*object property*, associated with the name of the property. Values of
the object property can either be stored in a member variable of the
object, or dynamically created when the methods are called.

All of the four DM base classes, PROCESS, VARIABLE, SYSTEM and STEPPER
can have a set of PROPERTYSLOTs, or object *properties*. In other words,
these classes inherit PROPERTYINTERFACE common base class.

What is PropertySlot for?
~~~~~~~~~~~~~~~~~~~~~~~~~

PROPERTYSLOTs can be used from model files (such as EM files) as a means
of giving parameter values to each objects in the simulation model (such
as ENTITY and STEPPER objects). It can also be ways of dynamic
communications between objects during the simulation.

Type of PropertySlot
~~~~~~~~~~~~~~~~~~~~

A type of a PROPERTYSLOT is any one of these four types:

-  ``Real``

-  ``Integer``

-  ``String``

-  ``Polymorph``

How to define a PropertySlot
----------------------------

To define a PROPERTYSLOT on an object class, you have to:

1. Define set and/or get method(s).

2. If necessary, define a member variable to store the property value.

3. Register the method(s) as a PROPERTYSLOT.

Set method and get method
~~~~~~~~~~~~~~~~~~~~~~~~~

A PROPERTYSLOT is a pair of object methods, *set method* and *get
method*, associated with a property name. Either one of the methods can
be ommited. If there is a set method defined for a PROPERTYSLOT, the
PROPERTYSLOT is said to be *setable*. If there is a get method, it is
*getable.*

A set method must have the following signature to be recognized by the
system.

::

    void CLASS::* ( const T&)

And a get method must look like this:

::

    const T CLASS::* ( void ) const

where ``T`` is a property type and ``CLASS`` is the object class that
the PROPERTYSLOT belongs to.

Don't worry, you don't need to memorize these prototypes. The following
four macoros can be used to declare and define set/get methods of a
specific type and a property name.

-  ``SET_METHOD( ,  )``

   -  *Expansion:*

      ::

          void set( const &value )

   -  *Usage:* ``SET_METHOD`` macro is used to declare or define a
      property set method, of which the property type is ``TYPE`` and
      the property name is ``NAME``, in a class definition. The given
      property value is available as the ``value`` argument variable.

   -  *Example:*

      This code:

      ::

          class FooProcess
          {
              SET_METHOD( Real, Flux )
              {
                  theFlux = value;
              }

              Real theFlux;
          };

      will expand to the following C++ program.

      ::

          class FooProcess
          {
              void setFlux( const Real& value )
              {
                  theFlux = value;
              }

              Real theFlux;
          };

      In this example, the given property value is stored in the member
      variable ``theFlux``.

-  ``GET_METHOD( ,  )``

   -  *Expansion:*

      ::

          const  get() const

   -  *Usage:* ``GET_METHOD`` macro is used to declare or define a
      property get method, of which the property type is ``TYPE`` and
      the property name is ``NAME``, in a class definition. Definition
      of the method must return the value of the property as a ``TYPE``
      object.

   -  *Example:*

      This code:

      ::

          class FooProcess
          {
              GET_METHOD( Real, Flux )
              {
                  return theFlux;
              }

              Real theFlux;
          };

      will expand to the following C++ program.

      ::

          class FooProcess
          {
              const Real getFlux() const
              {
                  return theFlux;
              }

              Real theFlux;
          };

-  ``SET_METHOD_DEF( , ,  )``

   -  *Expansion:*

      ::

          void ::set( const &value )

   -  *Usage:* ``SET_METHOD_DEF`` macro is used to define a property set
      method outside class scope.

   -  *Example:*

      ``SET_METHOD_DEF`` macro is usually used in conjunction with
      ``SET_METHOD`` macro. For instance, the following code declares a
      property setter method with ``SET_METHOD`` in the class
      definition, and later defines the actual body of the method using
      ``SET_METHOD_DEF``.

      ::

          class FooProcess
          {
              virtual SET_METHOD( Real, Flux );

              Real theFlux;
          };

          SET_METHOD_DEF( Real, Flux, FooProcess )
          {
              theFlux = value;
          }

      The definition part will expand to the following C++ program.

      ::

          void FooProcess::setFlux( const Real& value )
          {
              theFlux = value;
          }

-  ``GET_METHOD_DEF( , ,  )``

   -  *Expansion:*

      ::

          const  ::get() const

   -  *Usage:* ``GET_METHOD_DEF`` macro is used to define a property get
      method outside class scope.

   -  *Example:* See the example of ``SET_METHOD_DEF`` above.

If the property is both setable and getable, and is simply stored in a
member variable, the following macro can be used.

::

    SIMPLE_SET_GET_METHOD( ,  )

This assumes there is a variable with the same name as the property name
(``NAME``), and expands to a code that is equivalent to:

::

    SET_METHOD( ,  )
    {
       = value;
    }

    GET_METHOD( ,  )
    {
      return ;
    }

Registering PropertySlots
~~~~~~~~~~~~~~~~~~~~~~~~~

To register a PROPERTYSLOT on a class, one of these macros in the
``LIBECS_DM_OBJECT`` macro of the target class:

-  ``PROPERTYSLOT_SET_GET( ,  )``

   Use this if the property is both setable and getable, which means
   that the class defines both set method and get method.

   For example, to define a property 'Flux' of type ``Real`` on the
   FooProcess class, write like this in the public area of the class
   definition:

   ::

       public:

         LIBECS_DM_OBJECT( ,  )
         {
           PROPERTYSLOT_SET_GET( ,  );
         }

   This registers these methods:

   ::

       void FooProcess::setFlux( const Real& );

   and

   ::

       const Real FooProcess::getFlux() const;

   as the set and get methods of 'Flux' property of the class
   FooProcess, respectively. Signatures of the methods must match with
   the prototypes defined in the previous section. ``LIBECS_DM_OBJECT``
   can have any number of properties. It can also be empty.

-  ``PROPERTYSLOT_SET( ,  )``

   This is almost the same as ``PROPERTYSLOT_SET_GET``, but this does
   not register get method. Use this if only a set method is available.

-  ``PROPERTYSLOT_GET( ,  )``

   This is almost the same as ``PROPERTYSLOT_SET_GET``, but this does
   not register set method. Use this if only a get method is available.

-  ``PROPERTYSLOT( , , ,  )``

   If the name of either get or set method is different from the default
   format (set``NAME``\ () or get\ ``NAME``\ ()), then use this macro
   with explicitly specifying the pointers to the methods.

   For example, the following use of the macro registers setFlux2() and
   anotherGetMethod() methods of Flux property of the class FooProcess:

   ::

       PROPERTYSLOT( Flux, Real, 
                     &FooProcess::setFlux2,
                     &FooProcess::anotherGetMethod );

If more than one PROPERTYSLOTs with the same name are created on an
object, the last is taken.

Load / save methods
~~~~~~~~~~~~~~~~~~~

In addition to set and get methods, load and save methods can be
defined. Load methods are called when the model is loaded from the model
file. Similarly, save methods are called when the state of the model is
saved to a file by saveModel() method of the simulator.

Unless otherwise specified, load and save methods default to set and get
methods. This default definition can be changed by using the following
some macros.

-  ``PROPERTYSLOT_LOAD_SAVE( , , , , ,  )``

   This macros is the most generic way to set the property methods; all
   of set method, get method, load method ans save method can be
   specified independently. If the ``LOAD_METHOD`` is ``NOMETHOD``, it
   is said to be not *loadable*, and it is not *savable* if
   ``SAVE_METHOD`` is ``NOMETHOD``.

-  ``PROPERTYSLOT_NO_LOAD_SAVE( , , ,  )``

   Usage of this macro is the same as ``PROPERTYSLOT`` in the previous
   section, but this sets both ``LOAD_METHOD`` and ``SAVE_METHOD`` to
   ``NOMETHOD``.

   That is, this macro is equivalent to writing:

   ::

-  ``PROPERTYSLOT_SET_GET_NO_LOAD_SAVE( , , ,  )``

   ``PROPERTYSLOT_SET_NO_LOAD_SAVE( , ,  )``

   ``PROPERTYSLOT_GET_NO_LOAD_SAVE( , ,  )``

   Usage of these macros are the same as: ``PROPERTYSLOT_SET_GET``,
   ``PROPERTYSLOT_SET``, and ``PROPERTYSLOT_GET``, except that load and
   save methods are not set instead of default to set and get methods.

Inheriting properties of base class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases you may also want to use properties of base class. To
inherit the baseclass properties, use ``INHERIT_PROPERTIES( )``
macro. This macro is usually placed before any property
definition macros (such as ``PROPERTY_SET_GET()``).

::

    LIBECS_DM_OBJECT( ,  )
    {
        INHERIT_PROPERTIES(  );
      
        PROPERTYSLOT_SET_GET( ,  );
    }

Here ``PROPERTY_BASECLASS`` is usually the same as ``BASECLASS``. An
exception is when the ``BASECLASS`` does not make use of
``LIBECS_DM_OBJECT()`` macro. In this case, choose the nearest baseclass
in the class hierarachy that uses ``LIBECS_DM_OBJECT()`` for
``PROPERTY_BASECLASS``.

Using PropertySlots In Simulation
---------------------------------

(1) Static direct access (using native C++ method) bypassing the
PROPERTYSLOT, (2) dynamically-bound access via a PROPERTYSLOT object,
(3) dynamically-bound access via PROPERTYINTERFACE.

Defining a new Process class
============================

To define a new PROCESS class, at least the following two methods need
to be defined.

-  initialize()

-  fire()

initialize() is called when the simulation state needs to be reset. Note
that reset can happen anytime during the session, not just at the
beginning; especially when the reintegration of the state is requested.
fire() is called when the reaction takes place. You have to update the
VARIABLEs referred to by your PROCESS according to VARIABLEREFERENCE.

The PROCESS's VARIABLEREFERENCEs are stored in
``theVariableReferenceVector`` member variable, sorted by coefficient.
Hence references that have negative coefficients are followed by those
of zero coefficients, and so by those of positive coefficients. You can
get the offset from which the "zero" or positive references start
through getZeroVariableReferenceOffset() or
getPositiveVariableReferenceOffset(). If you want to look up for a
specific VARIABLEREFERENCE by name, use getVariableReference().

::

    #include <libecs.hpp>
    #include <Process.hpp>

    USE_LIBECS;

    LIBECS_DM_CLASS( SimpleProcess, Process )
    {
    public:
        LIBECS_DM_OBJECT( SimpleFluxProcess, Process )
        {
            PROPERTYSLOT_SET_GET( Real, k );
        }

        SimpleProcess(): k( 0.0 )
        {
        }

        SIMPLE_SET_GET_METHOD( Real, k );

        virtual void initialize()
        {
            Process::initialize();
            S0 = getVariableReference( "S0" );
        }

        virtual void fire()
        {
            // concentration gets reverted to the number of molecules
            // according to the volume of the System where the Process belongs.
            setFlux( k * S0.getMolarConc() * getSuperSystem()->getSize() * N_A );
        }

    protected:
        Real k;
        VariableReference const& S0;
    };

    LIBECS_DM_INIT( SimpleProcess, Process );

Defining a new Stepper class
============================

Defining a new Variable class
=============================

Defining a new System class
===========================

