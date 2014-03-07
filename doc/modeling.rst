===================
Modeling with ECELL
===================

By reading this chapter, you can get information about:

How an ECELL's simulation model is organized.
How to create a simulation model.
How to write a model file in EM format.

Objects In The Model
====================

ECELL's simulation model is fully object-oriented. That is, the
simulation model is actually a set of *objects* connected each other.
The objects have *properties*, which determine characteristics of the
objects (such as a reaction rate constant if the object represent a
chemical reaction) and relationships between the objects.

Types Of The Objects
--------------------

A simulation model of APP consists of the following types of objects.

-  Usually more than one ENTITY objects

-  One or more STEPPER object(s)

ENTITY objects define the structure of the simulation model and
represented phenomena (such as chemical reactions) in the model. STEPPER
objects implement specific simulation algorithms.

Entity objects
~~~~~~~~~~~~~~

The ENTITY class has three subclasses:

-  VARIABLE

   This class of objects represent state variables. A VARIABLE object
   holds a scalar real-number value. A set of values of all VARIABLE
   objects in a simulation model defines the state of the model at a
   certain point in time.

-  PROCESS

   This class of objects represent phenomena in the simulation model
   that result in changes in the values of one or more VARIABLE objects.
   The way of change of the VARIABLE values can be either discrete or
   continuous.

-  SYSTEM

   This class of objects define overall structure of the model. A SYSTEM
   object can contain sets of these three types of ENTITY, VARIABLE,
   PROCESS, and SYSTEM objects. A SYSTEM can contain other SYSTEMs, and
   can form a tree-like structure.

Stepper objects
~~~~~~~~~~~~~~~

A model must have one or more STEPPER object(s). Each PROCESS and SYSTEM
object must be connected with a STEPPER object in the same model. In
other words, STEPPER objects in the model have non-overlapping sets of
PROCESS and SYSTEM objects.

STEPPER is a class which implement a specific simulation algorithm. If
the model has more than one STEPPER objects, the system conducts a
multi-stepper simulation. In addition to the lists of PROCESS and SYSTEM
objects, a STEPPER has a list of VARIABLE objects that can be read or
written by its PROCESS objects. It also has a time step interval as a
positive real-number. The system schedules STEPPER objects according to
the step intervals, and updates the current time.

When called by the system, a STEPPER object integrates values of related
VARIABLE objects to the current time (if the model has a differential
component), calls zero, one or more PROCESS objects connected with the
STEPPER in an order determined by its implementation of the algorithm,
and determines the next time step interval. See the following chapters
for details of the simulation procedure.

Object Identifiers
------------------

APP uses several types of identifier strings to specify the objects,
such as the ENTITY and STEPPER objects, in a simulation model.

ID (ENTITYID and STEPPERID)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ENTITY and STEPPER object has an *ID*. ID is a character string of
arbitrary length starting from an alphabet or '\_' with succeeding
alphabet, '\_', and numeric characters. APP treats IDs in a
case-sensitive way.

If the ID is used to indicate a STEPPER object, it is called a
STEPPERID. The ID points to an ENTITY object is refered to as ENTITYID,
or just *ID*.

(need EBNF here)

Examples: ``_P3``, ``ATP``, ``GlucoKinase``

SystemPath;
~~~~~~~~~~~

The SYSTEMPATH identifies a SYSTEM from the tree-like hierarchy of
SYSTEM objects in a simulation model. It has a form of ENTITYID strings
joined by a character '/' (slash). As a special case, the SYSTEMPATH of
the root system is ``/``. For instance, if there is a SYSTEM ``A``, and
``A`` has a subsystem ``B``, a SYSTEMPATH ``/A/B`` specifies the SYSTEM
object ``B``. It has three parts: (1) the root system (``/``), (2) the
SYSTEM ``A`` directly under the root system, and (3) the SYSTEM ``B``
just under ``A``.

A SYSTEMPATH can be relative. The relative SYSTEMPATH does not point at
a SYSTEM object unless the current SYSTEM is given. A SYSTEMPATH is
relative if (1) it does not start with the leading ``/`` (the root
system), or (2) it contains '``.``\ ' (the current system) or '``..``\ '
(the super-system).

Examples: ``/A/B``, ``../A``, ``.``, ``/CELL/ER1/../CYTOSOL``

FullID
~~~~~~

A FULLID (FULLy qualified IDentifier) identifies a unique ENTITY object
in a simulation model. A FULLID comprises three parts, (1) a ENTITYTYPE,
(2) a SYSTEMPATH, and (3) an ENTITYID, joined by a character '``:``\ '
(colon).

::

    ::

The ENTITYTYPE is one of the following class names:

-  SYSTEM

-  PROCESS

-  VARIABLE

For example, the following FULLID points to a PROCESS object of which
ENTITYID is '``P``\ ', in the SYSTEM '``CELL``\ ' immediately under the
root system (``/``). Process:/CELL:P

FullPN
~~~~~~

FULLPN (FULLy qualified Property Name) specifies a unique *property*
(see the next section) of an ENTITY object in the simulation model. It
has a form of a FULLID and the name of the property joined by a
character '``:``\ ' (colon).

::

    :

or,

::

    :::

The following FULLPN points to 'Value' property of the VARIABLE object
``Variable:/CELL:S``. Variable:/CELL:S:Value

Object Properties
-----------------

ENTITY and STEPPER objects have *properties*. A property is an attribute
of a certain object associated with a name. Its value can be get from
and set to the object.

Types of object properties
~~~~~~~~~~~~~~~~~~~~~~~~~~

A value of a property has a *type*, which is one of the followings.

-  REAL number

   (ex. ``3.33e+10``, ``1.0``)

-  INTEGER number

   (ex. ``3``, ``100``)

-  STRINGTYPE

   STRINGTYPE has two forms: quoted and not quoted. A quoted STRINGTYPE
   can contain any ASCII characters except the quotation characters ('
   or "). Quotations can be omitted if the string has a form of a valid
   object identifier (ENTITYID, STEPPERID, SYSTEMPATH, FULLID, or
   FULLPN).

   If the STRINGTYPE is triple-quoted (by ``'''`` or ``"""``), it can
   contain new-line characters. (The current version still has some
   problems processing this.)

   (ex. ``_C10_A``, ``Process:/A/B:P1``, ``"It can
             include spaces if double-quoted."``,
   ``'single-quote is available too, if you want to
             use "double-quotes" inside.'``)

-  List

   The list can contain REAL, INTEGER, and STRINGTYPE values. This list
   can also contain other lists, that is, the list can be nested. A list
   must be surrounded by brackets (``[`` and ``]``), and the elements
   must be separated by space characters. In some cases outermost
   brackets are omitted (such as in EM files, see below).

   (ex. ``[ A 10 [ 1.0 "a string" 1e+10 ]
             ]`` )

Dynamic type adaptation of property values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system automatically convert the type of the property value if it is
different from what the object in the simulator (such as PROCESS and
VARIABLE) expects to get. That is, the system does not necessary raise
an error if the type of the given value differs from the type the
backend object accepts. The system tries to convert the type of the
value given in the model file to the requested type by the objects in
the simulator. The conversion is done by the objects in the simulator,
when it gets a property value. See also the following sections.

The conversion is done in the following manner.

-  From a numeric value (REAL or INTEGER)

   -  To a STRINGTYPE

      The number is simply converted to a character string. For example,
      a number 12.3 is converted to a STRINGTYPE ``'12.3'``.

   -  To a list

      A numeric value can be converted to a length-1 list which has that
      number as the first item. For example, 12.3 is equivalent to '[
      12.3 ]'.

-  From a STRINGTYPE

   -  To a numeric value (REAL or INTEGER)

      The initial portion of the STRINGTYPE is converted to a numeric
      value. The number can be represented either in a decimal form or a
      hexadecimal form. Leading white space characters are ignored.
      'INF' and 'NAN' (case-insensitive) are converted to an infinity
      and a NaN (not-a-number), respectively. If the initial portion of
      the STRINGTYPE cannot be converted to a numeric value, it is
      interpreted as a zero (0.0 or 0). This conversion procedure is
      equivalent to C functions ``strtol`` and ``strtod``, according to
      the destined type.

   -  To a list

      A STRINGTYPE can be converted to a length-1 list which has that
      STRINGTYPE as the first item. For example, 'string' is equivalent
      to '[ 'string' ]'.

-  From a list

   -  To a numeric or a STRINGTYPE value

      It simply takes the first item of the list. If necessary the taken
      value is further converted to the destined types.

    **Note**

    When converting from a REAL number to an INTEGER, or from a
    STRINGTYPE to a numeric value, overflow and underflow can occur
    during the conversion. In this case an exception (TYPE??) is raised
    when the backend object attempts the conversion.

E-Cell Model (EM) File Basics
=============================

Now you know the ECELL's simulation model consists of what types of
objects, and the objects have their properties. The next thing to
understand is how the simulation model is organized: the structure of
the model. But wait, learn the syntax of the ECELL model (EM) file
before proceeding to the next section would help you very much to
understand the details of the structure of the model, because most of
the example codes are in EM.

What Is EM?
-----------

In APP, the standard file format of model description and exchange is
XML-based EML (E-Cell Model description Language). Although EML is an
ideal means of integrating E-Cell with other software components such as
GUI model editors and databases, it is very tedious for human users to
write and edit by hand.

E-Cell Model (EM) is a file format with a programming language-like
syntax and a powerful embedded EMPY preprocessor, which is designed to
be productive and intuitive especially when handled by text editors and
other text processing programs. Semantics of EM and EML files are almost
completely equivalent to each other, and going between these two formats
is meant to be possible with no loss of information (some exceptions are
comments and directions to the preprocessor in EM). The file suffix of
EM files is ".em".

Why and when use EM?
~~~~~~~~~~~~~~~~~~~~

Although E-Cell Modeling Environment (which is under development) will
provide means of more sophisticated, scalable and intelligent model
construction on the basis of EML, learning syntax and semantics of EM
may help you get the idea of how object model inside ECELL is organized
and how it is driven to conduct simulations. Furthermore, owing to the
nature of the plain programming language-like syntax, EM can be used as
a simple and intuitive tool to communicate with other ECELL users. In
fact, this manual uses EM to illustrate how the model is constructed in
ECELL

EM files can be viewed as EML generator scripts.

EM At A Glance
--------------

Before getting into the details of EM syntax, let's have a look at a
tiny example. It's very simple, but you do not need to understand
everything for the moment.

::

    Stepper ODEStepper( ODE_1 ) 
    { 
            # no property 
    } 
     
    System System( / ) 
    { 
            StepperID       ODE_1;

            Variable Variable( SIZE )
            {
                    Value   1e-18; 
            }
     
            Variable Variable( S ) 
            { 
                    Value   10000; 
            } 
     
            Variable Variable( P ) 
            { 
                    Value   0; 
            } 

            Process MassActionFluxProcess( E ) 
            { 
                    Name  "A mass action from S to P."
                    k     1.0; 

                    VariableReferenceList [ S0 :.:S -1 ] 
                                          [ P0 :.:P 1 ];
            } 
     
    } 

This example is a model of a mass-action differential equation. In this
example, the model has a STEPPER ``ODE_1`` of class ODEStepper, which is
a generic ordinary differential equation solver. The model also has the
root system (``/``). The root sytem has the StepperID property, and four
ENTITY objects, VARIABLEs ``SIZE``, ``S`` and ``P``, and the PROCESS
``E``. ``SIZE`` is a special name of the VARIABLE, that determines the
size of the compartment. If the compartment is three-dimensional, it
means the volume of the compartment in [L] (liter). That value is used
to calculate concentrations of other VARIABLEs. These ENTITY objects
have their property values of several different types. For example,
``StepperID`` of the root system is the string without quotes
(``ODE_1``). The initial value given to Value property of the VARIABLE
``S`` is an integer number ``10000`` (and this is automatically
converted to a real number ``10000.0`` when the VARIABLE gets it because
the type of the Value property is REAL). Name property of the PROCESS
``E`` is the quoted string ``"A mass action from S to P"``, and 'k' of it is the real number
``1.0``. VariableReferenceList property of ``E`` is the list of two
lists, which contain strings (such as ``S0``), and numbers (such as
``-1``). The list contain relative FULLIDs (such as ``:.:S``) without
quotes.

General Syntax Of EM
--------------------

Basically an EM is (and thus an EML is) a list of just one type of
directives: *object instantiation*. As we have seen, ECELL's simulation
models have only two types of 'objects'; STEPPER and ENTITY. After
creating an object, property values of the object must be set. Therefore
the object instantiation has two steps: (1) creating the object and (2)
setting properties.

General form of object instantiation statements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following is the general form of definition (instantiation) of an
object in EM:

::

    TYPE CLASSNAME( ID )
    """INFO ()"""
    { 
            PROPERTY_NAME_1 PROPERTY_VALUE_1;
            PROPERTY_NAME_2 PROPERTY_VALUE_2;
            ...
            PROPERTY_NAME_n PROPERTY_VALUE_n;
    } 

where:

-  TYPE

   The type of the object, which is one of the followings:

   -  STEPPER

   -  VARIABLE

   -  PROCESS

   -  SYSTEM

-  ID

   This is a *StepperID* if the object type is STEPPER. If it is SYSTEM,
   put a SYSTEMPATH here. Fill in an ENTITYID if it is a VARIABLE or a
   PROCESS.

-  CLASSNAME

   The classname of this object. This class must be a subclass of the
   baseclass defined by *TYPE*. For example, if the *TYPE* is PROCESS,
   *CLASSNAME* must be a subclass of PROCESS, such as
   MassActionFluxProcess.

-  INFO

   An annotation for this object. This field is optional, and is not
   used in the simulation. A quoted single-line ("string") or a
   multi-line string ("""multi-line string""") can be put here.

-  PROPERTY

   An object definition has zero or more properties.

   The property starts with an unquoted property name string, followed
   by a property value, and ends with a semi-colon (``;``). For example,
   if the property name is Concentration and the value is ``10.0``, it
   may look like: Concentration 10.0;

   REAL, INTEGER, STRINGTYPE, and List are allowed as property value
   types (See the Object Properties section above).

   If the value is a List, outermost brackets are omitted. For example,
   to put a list

   ::

       [ 10 "string" [ LIST ] ]

   into a property slot ``Foo``, write a line in the object definition
   like this: Foo 10 "string" [ LIST ];

       **Note**

       All property values are lists, even if it is a scalar REAL
       number. Remember a number '1.0' is interconvertible with a
       length-1 list '[ 1.0 ]'. Therefore the system can correctly
       interpret property values without the brackets.

       In other words, if the property value is bracketed, for example,
       the following property value

       ::

           Foo [ 10 [ LIST ] ];

       is interpreted by the system as a length-1 List

       ::

           [ [ 10 [ LIST ] ] ]

       of which the first item is a list

       ::

           [ 10 [ LIST ] ]

       This may or may not be what you intend to have.

Macros And Preprocessing
------------------------

Before converting to EML, ``ecell3-em2eml`` command invokes the EMPY
program to preprocess the given EM file.

By using EMPY, you can embed any PYTHON expressions and statements after
'@' in an EM file. Put a PYTHON expression inside '@( python expression
)', and the macro will be replated with an evaluation of the expression.
If the expression is very simple, '()' can be ommited. Use '@{ pytyon
statements }' to embed PYTHON statements. For example, the following
code:

::

    @(AA='10')
    @AA

is expanded to:

::

    10

Of course the statement can be multi-line. This code

::

    @{
      def f( str ):
          return str + ' is true.'
    }

    @f( 'Video Games Boost Visual Skills' )

is expanded to

::

    Video Games Boost Visual Skills is true.

EMPY can also be used to include other files. The following line is
replaced with the content of the file ``foo.em`` immediately before the
EM file is converted to an EML:

::

    @include( 'foo.em' )

Use ``-E`` option of ``ecell3-em2eml`` command to see what happens in
the preprocessing. With this option, it outputs the result of the
preprocessing to standard output and stops without creating an EML file.

It has many more nice features. See the appendix A for the full
description of the EMPY program.

Comments
--------

The comment character is a sharp '#'. If a line contains a '#' outside a
quoted-string, anything after the character is considered a comment, and
not processed by the ``ecell3-em2eml`` command.

This is processed differently from the EMPY comments (@#). This comment
character is processed by the EMPY as a usual character, and does not
have an effect on the preprocessor. That is, the part of the line after
'#' is not ignored by EMPY preprocessor. To comment out an EMPY macro,
the EMPY comment (@#) must be used.

Structure Of The Model
======================

Top Level Elements
------------------

Usually an EM has one or more STEPPER and one or more SYSTEM statements.
These statements are top-level elements of the file. General structure
of an EM file may look like this:

::

    STEPPER_0
    STEPPER_1
    ...
    STEPPER_n

    SYSTEM_0 # the root system ( '/' )
    SYSTEM_1
    ...
    SYSTEM_m

``STEPPER_?`` is a STEPPER statement and ``SYSTEM_?`` is a SYSTEM
statement.

Systems
-------

The root system
~~~~~~~~~~~~~~~

The model must have a SYSTEM with a SYSTEMPATH '``/``\ '. This SYSTEM is
called the *root system* of the model.

::

    System System( / )
    {
        # ...
    }

The class of the root system is always System, no matter what class you
specify. This is because the simulator creates the root sytem when it
starts up, before loading the model file. That is, the statement does
not actually create the root system object when loading the EML file,
but just set its property values. Consequently the class name specified
in the EML is ignored. The model file must always have this root system
statement, even if you have no property to set.

Constructing the system tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model has more than one SYSTEM objects, it must form a tree which
starts from the root system (/). For example, the following is *not* a
valid EM.

::

    System System( / )
    {
    }

    System System( /CELL0/MITOCHONDRION0 )
    {
    }

This is invalid because these two SYSTEM objects, ``/`` and
``/CELL0/MITOCHONDRION0`` are not connected to each other, nor form a
single tree. Adding another SYSTEM, ``/CELL0``, makes it valid.

::

    System System( / )
    {
    }

    System System( /CELL0 )
    {
    }

    System System( /CELL0/MITOCHONDRION0 )
    {
    }

Of course a SYSTEM can have arbitrary number of sub-systems.

::

    System System( / )
    {
    }

    System System( /CELL1 ) {}
    System System( /CELL2 ) {}
    System System( /CELL3 ) {}
    # ...

    **Note**

    In future versions, the system will support composing a model from
    multiple model files (EMs or EMLs). This is not the same as the EM's
    file inclusion by EMPY preprocessor.

Sizes of the Systems
~~~~~~~~~~~~~~~~~~~~

If you want to define the size of a SYSTEM, create a VARIABLE with an ID
'``SIZE``\ '. If the SYSTEM models a three-dimensional compartment, the
``SIZE`` here means the volume of that compartment. The unit of the
volume is [L] (liter). In the next example, size of the root system is
``1e-18``.

::

    System System( / )
    {
        Variable Variable( SIZE )    # the size (volume) of this compartment
        {
            Value   1e-18;
        }
    }

If a System has no '``SIZE``\ ' VARIABLE, then it shares the ``SIZE``
VARIABLE with its supersystem. The root system always has its SIZE
VARIABLE. If it is not given by the model file, then the simulator
automatically creates it with the default value 1.0. The following
example has four SYSTEM objects, and two of them (``/`` and
``/COMPARTMENT``) have their own ``SIZE`` variables. Remaining two
(``/SUBSYSTEM`` and its subsystem ``/SUBSYSTEM/SUBSUBSYSTEM``) share the
``SIZE`` VARIABLE with the root system.

::

    System System( / )                       # SIZE == 1.0 (default)
    {
        # no SIZE
    }

    System System( /COMPARTMENT )            # SIZE == 2.0e-15
    {
        Variable Variable( SIZE )
        {
            Value 2.0e-15
        }
    }

    System System( /SUBSYSTEM )              # SIZE == SIZE of the root sytem
    {
        # no SIZE
    }

    System System( /SUBSYSTEM/SUBSUBSYSTEM ) # SIZE == SIZE of the root system
    {
        # no SIZE
    }

    **Note**

    Behavior of the system when zero or negative number is set to SIZE
    is undefined.

    **Note**

    Currently, the unit of the SIZE is (10 cm)^\ *d*, where d is
    dimension of the SYSTEM. If d is 3, it is (10 cm)^3 == liter. This
    specification is still under discussion, and is subject to change in
    future versions.

Variables And Processes
-----------------------

A SYSTEM statement has zero, one or more VARIABLE and PROCESS statements
in addition to its properties.

::

    System System( / )
    {
        # ... properties of this System itself comes here..

        Variable Variable( V0 ) {}
        Variable Variable( V1 ) {}
        # ...
        Variable Variable( Vn ) {}

        Process SomeProcess( P0 )  {}
        Process SomeProcess( P1 )  {}
        # ...
        Process OtherProcess( Pm ) {}
    }

Do not put a SYSTEM statement inside SYSTEM.

Connecting Steppers With Entity Objects
---------------------------------------

Any PROCESS and VARIABLE object in the model must be connected with a
STEPPER by setting its StepperID property. If the StepperID of a PROCESS
is omitted, it defaults to that of its supersystm (the SYSTEM the
PROCESS belongs to). StepperID of SYSTEM cannot be omitted.

In the following example, the root sytem is connected to the STEPPER
``STEPPER0``, and the PROCESS ``P0`` and ``P1`` belong to STEPPERs
``STEPPER0`` and ``STEPPER1``, respectively.

::

    Stepper SomeClassOfStepper( STEPPER0 )    {}
    Stepper AnotherClassOfStepper( STEPPER1 ) {}

    System System( / )  # connected to STEPPER0
    {
        StepperID     STEPPER0;

        Process AProcess( P0 )     # connected to STEPPER0
        {
            # No StepperID specified.
        }

        Process AProcess( P1 )     # connected to STEPPER1
        {
            StepperID     STEPPER1;
        }
    }

Connections between STEPPERs and VARIABLEs are automatically determined
by the system, and cannot be specified manually. See the next section.

Connecting Variable Objects With Processes
------------------------------------------

A PROCESS object changes values of VARIABLE object(s) according to a
certain procedure, such as the law of mass action. What VARIABLE objects
the PROCESS works on cannot be determined when it is programmed, but it
must be specified by the modeler when the PROCESS takes part in the
simulation. VariableReferenceList property of the PROCESS relates some
VARIABLE objects with the PROCESS.

VariableReferenceList is a list of *VARIABLEREFERENCEs*. A
VARIABLEREFERENCE, in turn, is usually a list of the following four
elements:

::

    [     ]

The last two fields can be omitted:

::

    [    ]

or,

::

    [   ]

These elements have the following meanings.

1. Reference name

   This field gives a local name inside the PROCESS to this
   VARIABLEREFERENCE. Some PROCESS classes use this name to identify
   particular instances of VARIABLEREFERENCE.

   Currently, this reference name must be set for all
   VARIABLEREFERENCEs, even if the PROCESS does not use the name at all.

   Lexical rule for this field is the same as the ENTITYID; leading
   alphabet or '\_' with trailing alphabet, '\_', and numeric
   characters.

2. FULLID

   This FULLID specifies the VARIABLE that this VARIABLEREFERENCE points
   to.

   The SYSTEMPATH of this FULLID can be relative. Also, ENTITYTYPE can
   be omitted. That is, writing like this is allowed:

   ::

       :.:S0

   instead of

   ::

       Variable:/CELL:S0

   , if the PROCESS exists in the SYSTEM ``/CELL``.

3. Coefficient (*optional*)

   This coefficient is an integer value that defines weight of the
   connection between the PROCESS and the VARIABLE that this
   VARIABLEREFERENCE points to.

   If this value is a non-zero integer, then this VARIABLEREFERENCE is
   said to be a *mutator VARIABLEREFERENCE*, and the PROCESS can change
   the value of the VARIABLE. If the value is zero, this
   VARIABLEREFERENCE is not a mutator, and the PROCESS should not change
   the value of the VARIABLE.

   If the PROCESS represents a chemical reaction, this value is usually
   interpreted by the PROCESS as a stoichiometric constant. For example,
   if the coefficient is -1, the value of the VARIABLE is decreased by 1
   in a single occurence of the forward reaction.

   If omitted, *this field defaults to zero*.

4. *isAccessor* flag (*optional*)

   This is a binary flag; set either 1 (true) or 0 (false). If this
   *isAccessor* flag is false, it indicates that the behavior of PROCESS
   is not affected by the VARIABLE that this VARIABLEREFERENCE points
   to. That is, the PROCESS never reads the value of the VARIABLE. The
   PROCESS may or may not change the VARIABLE regardless of the value of
   this field.

   Some PROCESS objects automatically sets this information, if it knows
   it never changes the value of the VARIABLE of this VARIABLEREFERENCE.
   Care should be taken when you set this flag manually, because many
   PROCESS classes do not check this flag when actually read the value
   of the VARIABLE.

   *The default is 1 (true).* This field is often omitted.

       **Note**

       In multi-stepper simulations, this information sometimes helps
       the system to run efficiently. If the system knows, for example,
       all PROCESS objects in the STEPPER ``A`` do not change any
       VARIABLE connected to the other STEPPER ``B``, it can give ``B``
       more chance to have larger stepsizes, rather than always checking
       whether STEPPER ``A`` changed some of the VARIABLE objects. This
       flag is mainly used when there are more than one STEPPERs.

Consider a reaction PROCESS in the root system, ``R``, consumes the
VARIABLE ``S`` and produces the VARIABLE ``P``, taking ``E`` as the
enzyme. This class of PROCESS requires to give the enzyme as a
VARIABLEREFERENCE of name ``ENZYME``. All the VARIABLE objects are in
the root system. In EM, VariableReferenceList of this PROCESS may appear
like this:

::

    System System( / )
    {
        # ...
        Variable Variable( S ) {}
        Variable Variable( P ) {}
        Variable Variable( E ) {}

        Process SomeReactionProcess( R )
        {
            # ...
            VariableReferenceList [ S0     :.:S -1 ]
                                  [ P0     :.:P  1 ]
                                  [ ENZYME :.:E  0 ];

        }
    }

Modeling Schemes
================

ECELL is a multi-algorithm simulator. It can run any kind of simulation
algorithms, both discrete and continuous, and these simulation
algorithms can be used in any combinations. This section exlains how you
can find appropriate set of object classes for your modeling and
simulation projects. This section does not give a complete list of
available object classes nor detailed usage of those classes. Read the
chapter "Standard Dynamic Module Library" for more info.

Discrete Or Continuous ?
------------------------

ECELL can model both discrete and continuous processes, and these can be
mixed in simulation. The system models discrete and continuous systems
by discriminating two different types of PROCESS and STEPPER objects:
discrete PROCESS / STEPPER and continuous PROCESS / STEPPER.

    **Note**

    VARIABLE and SYSTEM do not have special discrete and continuous
    classes. The base VARIABLE class supports both discrete and
    continous operations, because it can be connected to any types of
    PROCESS and STEPPER objects. SYSTEM objects do not do any
    computation that needs to discriminate discrete and continuos.

Discrete classes
~~~~~~~~~~~~~~~~

A PROCESS object that models discrete changes of one or more VARIABLE
objects is called a *discrete PROCESS*, and it must be used in
conjunction with a *discrete STEPPER*. A discrete PROCESS directly
changes the *values* of related VARIABLE objects when its STEPPER
requests to do so.

There are two types of discrete PROCESS / STEPPER classes: discrete and
discrete event.

-  Discrete

   A discrete PROCESS changes values of connected VARIABLE objects (i.e.
   appear in its VariableReferenceList property) discretely. In the
   current version, there is no special class named DiscreteProcess,
   because the base PROCESS class is already a discrete PROCESS by
   default. The manner of the change of VARIABLE values is determined
   from values of its accessor VARIABLEREFERENCEs, its property values,
   and sometimes the current time of the STEPPER. Unlike discrete event
   PROCESS, which is explained in the next item, it does not necessary
   specify when the discrete changes of VARIABLE values occur. Instead,
   it is unilaterally determined and fired by a discrete STEPPER.

   A STEPPER that requires all PROCESS objects connected is discrete
   PROCESS objects is call a discrete STEPPER. The current version has
   no special class DiscreteStepper, because the base STEPPER class is
   already discrete.

-  Discrete event

   Discrete event is a special case of discreteness. The system provides
   DiscreteEventStepper and DiscreteEventProcess classes for
   discrete-event modeling. In addition to the ordinary firing method
   (fire() method) of the base PROCESS class, the DiscreteEventProcess
   defines a method to calculate *when* is the next occurrence of the
   event (the discrete change of VARIABLE values that this discrete
   event PROCESS models) from values of its accessor VARIABLEREFERENCEs,
   its property values, and the current time of the STEPPER.
   DiscreteEventStepper uses information given by this method to
   determine when each of discrete event PROCESS should be fired.
   DiscreteEventStepper is instantiatable. See the chapter Standard
   Dynamic Module Library for more detailed description of how
   DiscreteEventStepper works.

Continuous classes
~~~~~~~~~~~~~~~~~~

On the other hand, a PROCESS that calculates continuous changes of
VARIABLE objects is called a *continuous PROCESS*, and is used in
combination with a *continuous STEPPER*. Continuous PROCESS objects
simulate the phenomena that represents by setting *velocities* of
connected VARIABLE objects, rather than directly changing their values
in the case of discrete PROCESS objects. A continuous STEPPER integrates
the values of VARIABLE objects from the velocities given by the
continuous PROCESS objects, and determines when the velocities should be
recalculated by the PROCESS objects. A typical application of continuous
PROCESS and STEPPER objects is to implement differential equations and
differential equation solvers, respectively, to form a simulation system
of the system of differential equations.

Some Available Discrete Classes
-------------------------------

Followings are some available discrete classes.

NRStepper and GillespieProcess (Gillespie-Gibson pair)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example of discrete-event simulation method provided by ECELL is a
variant of Gillespie's stochastic algorithm, the Next Reaction Method,
or Gillespie-Gibson algorithm. NRStepper class implements this
algorithm. When this STEPPER is used in conjunction with
GillespieProcess objects, which is a subclass of DiscreteEventProcess
and calculates a time of the next occurence of the reaction using
Gillespie's reaction probability equation and a random number, ECELL
conducts a Gillespie-Gibson stochastic simulation of elementary chemical
reactions. In fact, the Next Reaction Method is nothing but a standard
discrete event simulation algorithm, and NRStepper is just an alias of
the DiscreteEventStepper class.

Usage of this pair of classes of objects is simple: just set the
StepperID, VariableReferenceList and the rate constant property k of
those GillespieProcess objects.

DiscreteTimeStepper
~~~~~~~~~~~~~~~~~~~

A type of discrete STEPPER that is provided by the system is
*DiscreteTimeStepper*. This class of STEPPER, when instantiated, calls
all discrete PROCESS objects with a fixed user-specified time-interval.
For example, if the model has a DiscreteTimeStepper with 0.001 (second)
of StepInterval property, it fires all of its PROCESS objects every
milli-second. DiscreteTimeStepper is discrete time because it does not
have time between steps; it ignores a signal from other STEPPER objects
(*STEPPER interruption*) that notifies a change of system state (values
of VARIABLE objects) that may affect its PROCESS objects. Such a change
is reflected in the next step.

PassiveStepper
~~~~~~~~~~~~~~

Another class of discrete STEPPER is PassiveStepper. This can partially
be seen as a DiscreteTimeStepper with an infinite StepInterval, but
there is a difference. Unlike DiscreteTimeStepper, this does *not*
ignore STEPPER interruptions, which notify change in the system state
that may affect this STEPPER's PROCESS objects.

This STEPPER is used when some special procedures (coded in discrete
PROCESS objects) must be invoked when other STEPPER object may have
changed a value or a velocity of at least one VARIABLE that this
STEPPER's PROCESS objects accesses.

PythonProcess
~~~~~~~~~~~~~

PythonProcess allows users to script a PROCESS object in full PYTHON
syntax.

initialize() and fire() methods can be scripted with InitializeMethod
and FireMethod properties, respectively.

PythonProcess can be either discrete or continuous. This 'operation
mode' can be specified by setting IsContinuous property. The default is
false (0), or discrete. To switch to the continuous mode, set 1 to the
property:

::

    Process PythonProcess( PY1 )
    {
        IsContinuous 1;
    }

In addition to regular PYTHON constructs, the following objects,
methods, and attributes are available in both of the method properties
(InitializeMethod and FireMethod):

-  Properties

   PythonProcess accepts arbitrary names of properties. For example, the
   following code creates two new properties.

   ::

       Process PythonProcess( PY1 )
       {
           NewProperty "new property";
           KK          3.0;
       }

   These properties can be use in PYTHON methods:

   ::

       Process PythonProcess( PY1 )
       {
           # ... NewProperty and KK are set

           InitializeMethod "print NewProperty";

           FireMethod '''
       KK += 1.0
       print KK 
       ''';
       }

   A new property can also be created within PYTHON methods.

   ::

           InitializeMethod "A = 3.0"; # A is created
           FireMethod "print A * 2";   # A can be used here

   These properties are treated as a global variable.

-  Objects

   -  ``self``

      This is the PROCESS object itself. This has the following
      attributes:

      -  Activity

         The current value of Activity property of this PROCESS.

      -  addValue( ``value`` )

         Add each VARIABLEREFERENCE the ``value`` multiplied by the
         coefficient of the VARIABLEREFERENCE.

         Using this method implies that this PROCESS is discrete. Check
         that IsContinuous property is false.

      -  getSuperSystem()

         This method gets the super system of this PROCESS. See below
         for the attributes of SYSTEM objects.

      -  Priority

         The Priority property of this PROCESS.

      -  setFlux( ``value`` )

         Add each VARIABLEREFERENCE's velocity the ``value`` multiplied
         by the coefficient of the VARIABLEREFERENCE.

         Using this method implies that this PROCESS is continuous.
         Check that IsContinuous property is true.

      -  StepperID

         StepperID of this PROCESS.

   -  VARIABLEREFERENCE

      VARIABLEREFERENCE instances given in the VariableReferenceList
      property of this PROCESS can be used in the PYTHON methods. Each
      instance has the following attributes:

      -  addFlux( ``value`` )

         Multiply the ``value`` by the Coefficient of this
         VARIABLEREFERENCE, and add that to the VARIABLE's velocity.

      -  addValue( ``value`` )

         Add the ``value`` to the Value property of the VARIABLE.

      -  addVelocity( ``value`` )

         Add the ``value`` to the Velocity property of the VARIABLE.

      -  Coefficient

         The coefficient of the VARIABLEREFERENCE

      -  getSuperSystem()

         Get the super system of the VARIABLE. A SYSTEM object is
         returned.

      -  MolarConc

         The concentration of the VARIABLE in Molar [M].

      -  Name

         The name of the VARIABLEREFERENCE.

      -  NumberConc

         The concentration in number [ num / size of the VARIABLE's
         super system. ]

      -  IsFixed

         Zero if the Fixed property of the VARIABLE is false. Otherwise
         a non-zero integer.

      -  IsAccessor

         Zero if the IsAccessor flag of the VARIABLEREFERENCE is false.
         Otherwise a non-zero integer.

      -  TotalVelocity

         The total current velocity. Usually of no use.

      -  Value

         The value of the VARIABLE

      -  Velocity

         The provisional velocity given by the currently stepping
         STEPPER. Usually of no use.

   -  SYSTEM

      A SYSTEM object has the following attributes.

      -  getSuperSystem()

         Get the super system of the SYSTEM. A SYSTEM object is
         returned.

      -  Size

         The size of the SYSTEM.

      -  SizeN\_A

         Equivalent to ``Size *
                     N_A``, where N\_A is a Avogadro's number.

      -  StepperID

         The StepperID of the SYSTEM.

Here is an example uses of PythonProcess.

::

    Process PythonProcess( PY1 )
    {
        # IsContinuous 0; -- default
        FireMethod "S1.Value = S2.Value + S3.Value";
        VariableReferenceList [(S1)] [(S2)] [(S3)];
    }

PythonEventProcess
~~~~~~~~~~~~~~~~~~

This class enables users PYTHON scripting of time-events. In addition to
initialize() and fire(), updateStepInterval() method can be scripted
with this class. Use UpdateStepIntervalMethod property to set this.

In addition to those of PythonProcess, the ``self`` object of
PythonEventProcess has some more attributes:

-  StepInterval

   The most recent StepInterval calculated by the updateStepInterval()
   method.

-  DependentProcessList

   This attribute holds a tuple of IDs of dependent PROCESSes of this
   PROCESS.

This class of objects must be used with a DiscreteEventStepper.

This class is under development.

Other discrete classes
~~~~~~~~~~~~~~~~~~~~~~

STEPPER classes for explicit and implicit tau leaping algorithms are
under development.

A flux-distribution method for hybrid dynamic/static simulation of
biochemical pathways is available with the following classes:
FluxDistributionStepper, FluxDistributionProcess,
QuasiDynamicFluxProcess. Usage of this scheme is to be described.

Some Available Continuous Classes
---------------------------------

ECELL supports both Ordinary Differential Equation (ODE) and
Differential-Algebraic Equation (DAE) models, and has STEPPER classes
for each type of formalisms.

Also, the system is shipped with some continuous PROCESS classes. For
example, MassActionFluxProcess calculates a reaction rate according to
the law of mass action. ExpressionFluxProcess allows users to describe
arbitrary rate equations in model files. PythonProcess and
PythonFluxProcess are used to script PROCESS objects in PYTHON. Some
enzyme kinetics rate laws are also available.

Generic ordinary differential Steppers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your model is a system of ODEs, then in this version of the software
(version APPVERSION) the recommended choice is ODEStepper. This STEPPER
is a high-performance replacement of ODE45Stepper, which was the choice
for the previous versions.

ODEStepper is implemented so that it can adaptively switch the solving
method between the implicit one (Radau IIA) and the explicit one
(Dormand-Prince), according to the current stiffness of the input.

Some other available ODE STEPPER classes are ODE23Stepper, which
employes a lower (the second) order integration algorithm, and
FixedODE1Stepper that implements the simplest Euler algorithm without an
adaptive step sizing mechanism.

These ODE STEPPER classes except for the FixedODE1Stepper have some
common property slots for user-specifiable parameters. Here is a partial
list:

-  Tolerance

   An error tolerance in local truncation error. Giving this smaller
   numbers forces the STEPPER to take smaller step sizes, and slows down
   the simulation. Greater numbers results in faster run with sacrifice
   of accuracy. A typical number is 1e-6.

-  MinStepInterval

   Species the minimum value of step width. This limit precedes the
   Tolerance property above.

   These properties can also be useful to completely disable the
   adaptive step size control mechanism: set the same number to both of
   the property slots.

-  MaxStepInterval

   This property is no longer supported and has no specific effect if it
   is set

MassActionFluxProcess
~~~~~~~~~~~~~~~~~~~~~

MassActionFluxProcess is a class of PROCESS for simple mass-actions.
This class calculates a flux rate according to the irreversible
mass-action. Use a property k to specify a rate constant.

ExpressionFluxProcess
~~~~~~~~~~~~~~~~~~~~~

ExpressionFluxProcess is designed for easy and efficient representations
of continuous flux rate equations.

Expression property of this class accepts a plain text rate expression.
The expression must be evaluated to give a flux rate in [ number /
second ]. (Note that this is a number per second, not concentration per
second.) Here is an example use of ExpressionFluxProcess:

::

    Process ExpressionFluxProcess( P1 )
    {
        k 0.1;
        Expression "k * S.Value";

        VariableReferenceList [ S :.:S -1 ] [ P :.:P 1 ];
    }

Compared to PythonProcess or PythonFluxProcess below, it runs
significantly faster with sacrifice of some flexibility in scripting.

The following shows elements those can be used in the Expression
property. The set of available arithmetic operators and mathematical
functions are meant to be equivalent to SBML level 2, except control
structures.

-  Constants

   Numbers (e.g. 10, 10.33, 1.33e-5), ``true``, ``false`` (equivalent to
   zero), ``pi`` (Pi), ``NaN`` (Not-a-Number), ``INF`` (Infinity),
   ``N_A`` (Avogadro's number), ``exp`` (the base of natural
   logarithms).

-  Arithmetic operators

   ``+``, ``-``, ``*``, ``/``, ``^`` (power; this can equivalently be
   written as ``pow( x, y )``).

-  Built-in functions

   ``abs``, ``ceil``, ``exp``, \*\ ``fact``, ``floor``, ``log``,
   ``log10``, ``pow`` ``sqrt``, \*\ ``sec``, ``sin``, ``cos``, ``tan``,
   ``sinh``, ``cosh``, ``tanh``, ``coth``, \*\ ``csch``, \*\ ``sech``,
   \*\ ``asin``, \*\ ``acos``, \*\ ``atan``, \*\ ``asec``, \*\ ``acsc``,
   \*\ ``acot``, \*\ ``asinh``, \*\ ``acosh``, \*\ ``atanh``,
   \*\ ``asech``, \*\ ``acsch``, \*\ ``acoth``. (Functions with astarisk
   '\*' are currently not available on the Windows version.)

   All functions but ``pow`` are unary functions. ``pow`` is a binary
   function.

-  Properties

   Similar to PythonProcess, ExpressionFluxProcess accepts arbitrary
   name properties in the model. Unlike PythonProcess, however, these
   properties of this class can hold only REAL values.

-  Objects

   -  ``self``

      This PROCESS object itself. This has the following attribute which
      is a sub set of that of PythonProcess:

      -  getSuperSystem()

   -  VARIABLEREFERENCE

      VARIABLEREFERENCE instances given in the VariableReferenceList
      property of this PROCESS can be used in the expression. Each
      instance has the following set of attributes, which is a sub set
      of that of PythonProcess:

      -  Value

      -  MolarConc

      -  NumberConc

      -  TotalVelocity

      -  Velocity

   -  SYSTEM

      A SYSTEM object has the following two attributes.

      -  Size

      -  SizeN\_A

Below is an example of the basic Michaelis-Menten reaction programmed
with the ExpressionFluxProcess.

::

    Process ExpressionFluxProcess( P )
    {
        Km    1.0;
        Kcat  10;

        Expression "E.Value * Kcat * S.MolarConc / ( S.MolarConc + Km )";

        VariableReferenceList [ S :.:S -1 ] [ P :.:P 1 ] [ E :.:E 0 ];
    }

Some pre-defined reaction rate classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the standard dynamic module library reference for availability of
some enzyme kinetics PROCESS classes.

PythonFluxProcess
~~~~~~~~~~~~~~~~~

PythonFluxProcess is almost the same as PythonProcess, except that (1)
it takes just a PYTHON expression (instead of statements) to its
Expression property, and (2) similar to ExpressionFluxProcess, the
evaluated value of the expression is implicitly passed to the setFlux()
method.

Generic differential-algebraic Steppers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For DAE models, use DAEStepper. The model must form a valid index-1 DAE
system. When a DAE STEPPER detects one or more discrete PROCESS objects,
it assumes that these are *algebraic PROCESS* objects. Thus, all
discrete PROCESS objects in a DAE STEPPER must be algebraic. See below
for what is algebraic PROCESS.

    **Note**

    Because it can be viewed that ODE is a special case of DAE problems
    which does not have a algebraic equations, but only differential
    equations, a DAE STEPPER can be used to run an ODE model. However,
    ODE Steppers are specialized for ODE problems, in terms of both the
    selection of integration algorithms and implementation issues, and
    generally use of an ODE STEPPER benefits in performance when the
    model is a system of ODEs.

Those properties of ODE STEPPER classes described above (such as the
Tolerance property) are also available for DAE STEPPER classes.

Algebraic Processes
~~~~~~~~~~~~~~~~~~~

This is a type of discrete PROCESS, but placed here because it is used
with a DAE STEPPER, which is continuous.

In principle, continuous PROCESS objects must be connected with
continuous STEPPER instances, and a discrete STEPPER is assumed to take
only discrete PROCESS objects. However, there are some exceptions. One
of such is the *algebraic processes*. Strangely enough, in DAE
simulations, seemingly discrete algebraic equations are solved
continuously in conjunction with other differential equations.

Algebraic equations in ECELL has the following form:

::

    0 = g( t, x )

where t is the time and x is a vector of variable references.

The DAE solver system of ECELL uses Activity property of PROCESS objects
to represent the value of the algebraic function ``g( t, x )``. An algebraic PROCESS must *not* change values of
VARIABLE objects explicitly. The DAE STEPPER does this job of finding a
point where the equation ``g()`` holds.

When modeling, be careful about coefficients of VARIABLEREFERENCEs of an
algebraic PROCESS. In most cases, simply set unities. The solver
respects these numbers when solving the system. For example, if the
coefficient of ``A`` is zero, it does not change the variable when
trying to find the solution, while it is used in the calculation of the
equation.

As a means of describing algebraic equations, ExpressionAlgebraicProcess
is available. The usage is the same as ExpressionFluxProcess, except
that the evaluation of its expression is interpreted as the value of the
algebraic function ``g()``.

The following examble describes an equation

::

    a * A + B = 10,  a = 1.5

::

    Stepper DAEStepper( DAE1 ) {}

    Process ExpressionAlgebraicProcess( P )
    {
        StepperID DAE1;

        a    1.5;

        Expression "( a * A + B ) - 10";

        VariableReferenceList [ A :.:A 1 ] [ B :.:B 1 ];
    }

To use C++ or PythonProcess for algebraic equations, call setActivity()
method to set the value of the equation. The following is an example
with a PythonProcess:

::

    Process PythonProcess( PY )
    {
        a    1.5;

        FireMethod "self.setActivity( ( a * A + B ) - 10 )";

        VariableReferenceList [ A :.:A 1 ] [ B :.:B 1 ];
    }

Power-law canonical DEs (S-System and GMA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESSYNSStepper supports S-System and GMA simulations by using the ESSYNS
algorithm. A ESSYNSStepper must be connected with either a
SSystemProcess or a GMAProcess as its sole VARIABLEREFERENCE. Use
SSystemMatrix or GMAMatrix property to set the system parameters.

A sample model under the directory ``doc/sample/ssystem/`` gives an
example usage.

These modules are still under development. More descriptions to come...

Modeling Convensions
====================

Units
-----

In APP, the following units are used. This standard is meant only for
the simulator's internal representation, and any units can be used in
the process of modeling. However, it must be converted to these standard
units before loaded by the simulator.

-  Time

   s (second)

-  Volume

   L (liter)

-  Concentration

   Molar concentration (M, or molar per L (liter), used for example in
   MolarConc property of a VARIABLE object) or,

   Number concentration (number per L (liter), NumberConc property of
   VARIABLE has this unit).


