==============================
Scripting A Simulation Session
==============================

By reading this chapter, you can get information about the following
items: What is ECELL Session Script (ESS)., How to run ESS in scripting
mode., How to use ESS in GUI mode., How to automate a simulation run by
writing an ESS file., How to write frontend software components for
ECELL in PYTHON.

Session Scripting
=================

An ECELL Session Script (ESS) is a PYTHON script which is loaded by a
ECELL SESSION object. A SESSION instance represents a single run of a
simulation.

An ESS is used to automate a single run of a simulation session. A
simple simulation run typically involves the following five stages:

1. Loading a model file.

   Usually an EML file is loaded.

2. Pre-simulation setup of the simulator.

   Simulator and model parameters, such as initial values of VARIABLE
   objects and property values of PROCESS objects, are set and/or
   altered. Also, data LOGGERs may be created in this phase.

3. Running the simulation.

   The simulation is run for a certain length of time.

4. Post-simulation data processing.

   In this phase, the resulting state of the model after the simulation
   and the data logged by the LOGGER objects are examined. The
   simulation result may be numerically processed. If necessary, go back
   to the previous step and run the simulation for more seconds.

5. Data saving.

   Finally, the processed and raw simulation result data are saved to
   files.

An ESS file usually has an extension '``.py``\ '.

Running ECELL Session Script
============================

There are three ways to execute ESS;

-  Execute the script from the operating system's command line (the
   shell prompt).

-  Load the script from frontend software such as OSOGO.

-  Use SESSIONMANAGER to automate the invokation of the simulation
   sessions itself. This is usually used to write mathematical analysis
   scripts, such as parameter tuning, which involves multiple runs of
   the simulator.

Running ESS in command line mode
--------------------------------

An ESS can be run by using ECELL3-SESSION command either in *batch mode*
or in *interactive mode*.

Batch mode
~~~~~~~~~~

To execute an ESS file without user interaction, type the following
command at the shell prompt:

::

               
            

ECELL3-SESSION command creates a simulation SESSION object and executes
the ESS file ``ess.py`` on it. The option [-e] can be omitted.
Optionally, if [-f model.eml] is given, the EML file ``model.eml`` is
loaded immediately before executing the ESS.

Interactive mode
~~~~~~~~~~~~~~~~

To run the ECELL3-SESSION in interactive mode, invoke the command
without an ESS file.

::

     
    ecell3-session [ for E-Cell SE Version 3, on Python Version 2.2.1 ]
    Copyright (C) 1996-2012 Keio University.
    Send feedback to Koichi Takahashi 

            

The banner and the prompt shown here may vary according to the version
you are using. If the option [-f model.eml] is given, the EML file
``model.eml`` is loaded immediately before prompting.

Giving parameters to the script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optionally *session parameters* can be given to the script. Given
session parameters can be accessible from the ESS script as global
variables (see the following section).

To give the ESS parameters from the ECELL3-SESSION command, use either
``-D`` or ``--parameters=`` option.

::

               
               
            

Both ways, ``-D`` and ``--parameters``, can be mixed.

Loading ESS from OSOGO
----------------------

To manually load an ESS file from the GUI, use File->loadScript menu
button.

GECELL command accepts ``-e`` and ``-f`` options in the same way as the
ECELL3-SESSION command.

Using SessionManager
--------------------

(a separate chapter?)

Writing ECELL Session Script
============================

The syntax of ESS is a full set of PYTHON language with some convenient
features.

Using Session methods
---------------------

General rules
~~~~~~~~~~~~~

In ESS, an instance of SESSION is given, and any methods defined in the
class can be used as if it is defined in the global namespace.

For example, to run the simulation for 10 seconds, use run method of the
SESSION object. self.run( 10 ) where self. points to the current SESSION
object given by the system. Alternatively, you can use ``theSession`` in
place of the ``self``. theSession.run( 10 )

Unlike usual PYTHON script, you can omit the object on which the method
is called if the method is for the current SESSION. run( 10 )

Loading a model
~~~~~~~~~~~~~~~

Let's try this in the interactive mode of the ECELL3-SESSION command. On
the prompt of the command, load an EML file by using loadModel() method.
``ecell3-session>>> ``\ ``loadModel( 'simple.eml' )`` Then the prompt
changes from ``ecell3-session>>> `` to ``, t=>>> `` ``simple.eml, t=0>>> ``

Running the simulation
~~~~~~~~~~~~~~~~~~~~~~

To proceed the time by executing the simulation, step() and run()
methods are used.

::



step( ``n`` ) conducts ``n`` steps of the simulation. The default value
of ``n`` is 1.

    **Note**

    In above example you may notice that the first call of step() does
    not cause the time to change. The simulator updates the time at the
    beginning of the step, and calculates a tentative step size after
    that. The initial value of the step size is zero. Thus it needs to
    call step() twice to actually proceed the time. See chapter 6 for
    details of the simulation mechanism.

To execute the simulation for some seconds, call run method with a
duration in seconds. (e.g. run( ``10`` ) for 10 seconds.) run method
steps the simulation repeatedly, and stops when the time is proceeded
for the given seconds. In other words, the meaning of run( ``10`` ) is
to run the simulation *at least* 10 seconds. It always overrun the
specified length of time to a greater or less.

The system supports run without an argument to run forever, if and only
if both *event checker* and *event handler* are set. If not, it raises
an exception. See setEventChecker() in the method list of Session class.

Getting current time
~~~~~~~~~~~~~~~~~~~~

To get the current time of the simulator, getCurrentTime() method can be
used.

::


Printing messages
~~~~~~~~~~~~~~~~~

You may want to print some messages in your ESS. Use message(
``message`` ) method, where ``message`` argument is a string to be
outputed.

By default the message is handled in a way the same as the Python's
print statement; it is printed out to the standard out with a trailing
new line. This behavior can be changed by using setMessageMethod()
method.

An example of using SESSION methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a tiny example of using SESSION methods which loads a model, run
a hundred seconds, and print a short message.

::

    loadModel( 'simple.eml' )
    run( 100 )
    message( 'stopped at %f seconds.' % getCurrentTime() )

Getting Session Parameters.
---------------------------

Session parameters are given to an ESS as global variables. Therefore
usage of the session parameters is very simple. For example, if you can
assume a session parameter ``MODELFILE`` is given, just use it as a
variable: loadModel( MODELFILE )

To check what parameters are given to ESS, use dir() or globals()
built-in functions. Session parameters are listed as well as other
available methods and variables. To check if a certain ESS parameter or
a global variable is given, write an if statement like this: if
'MODELFILE' in globals(): # MODELFILE is given else: # not given

    **Note**

    Currently there is no way to distinguish the Session parameters from
    other global variables from ESS.

Observing and Manipulating the Model with OBJECTSTUBs
-----------------------------------------------------

What is OBJECTSTUB?
~~~~~~~~~~~~~~~~~~~

OBJECTSTUB is a proxy object in the frontend side of the system which
corresponds to an internal object in the simulator. Any operations on
the simulator's internal objects should be done via the OBJECTSTUB.

There are three types of OBJECTSTUB:

-  ENTITYSTUB

-  STEPPERSTUB

-  LOGGERSTUB

each correspond to ENTITY, STEPPER, and LOGGER classes in the simulator,
respectively.

Why OBJECTSTUB is needed
~~~~~~~~~~~~~~~~~~~~~~~~

OBJECTSTUB classes are actually thin wrappers over the
ecell.ecs.Simulator class of the E-Cell Python Library, which provides
object-oriented appearance to the flat procedure-oriented API of the
class. Although SIMULATOR object can be accessed directly via
``theSimulator`` property of SESSION class, use of OBJECTSTUB is
encouraged.

This backend / frontend isolation is needed because lifetimes of backend
objects are not the same as that of frontend objects, nor are their
state transitions necessarily synchronous. If the frontend directly
manipulates the internal objects of the simulator, consistency of the
lifetime and the state of the objects can easily be violated, which must
not happen, without some complicated and tricky software mechanism.

Creating an OBJECTSTUB by ID
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get an OBJECTSTUB object, createEntityStub(), createStepperStub(),
and createLoggerStub() methods of SESSION class are used.

For example, to get an ENTITYSTUB, call the createEntityStub() method
with a *FullID* string:

::

     = createEntityStub(  )

Similarly, a STEPPERSTUB object and a LOGGERSTUB object can be retrieved
with a *StepperID* and a *FullPN*, respectively.

::

     = createStepperStub(  )

::

     = createLoggerStub(  )

Creating and checking existence of the backend object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating an OBJECTSTUB does not necessarily mean a corresponding object
in the simulator backend exists, or is created. In other words, creation
of the OBJECTSTUB is purely a frontend operation. After creating an
OBJECTSTUB, you may want to check if the corresponding backend object
exists, and/or to command the backend to create the backend object.

To check if a corresponding object to an OBJECTSTUB exists in the
simulator, use exists() method. For example, the following if statement
checks if a Stepper whose ID is ``STEPPER_01`` exists: aStepperStub =
createStepperStub( 'STEPPER\_01' ) if aStepperStub.exists(): # it
already exists else: # it is not created yet

To create the backend object, just call create() method.
aStepperStub.create()# Stepper 'STEPPER\_01' is created here

Getting the name and a class name from an OBJECTSTUB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get the name (or an ID) of an OBJECTSTUB, use getName() method.

To get the class name of an ENTITYSTUB or a STEPPERSTUB, call
getClassName() method. This operation is not applicable to LOGGERSTUB.

Setting and getting properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As described in the previous chapters, ENTITY and STEPPER objects has
*properties*. This section describes how to access the object properties
via OBJECTSTUBs. This section is not applicable to LOGGERSTUBs.

To get a property value from a backend object by using an ENTITYSTUB or
a STEPPERSTUB, invoke getProperty() method or access an object attribute
with a property name: aValue = aStub.getProperty( 'Activity' ) or
equivalently, aValue = aStub[ 'Activity' ]

To set a new property value to an ENTITY or a STEPPER, call
setProperty() method or mutate an object attribute with a property name
and the new value: aStub.getProperty( 'Activity', aNewValue ) or
equivalently, aStub[ 'Activity' ] = aNewValue

List of all the properties can be gotten by using getPropertyList
method, which returns a list of property names as a Python TUPLE
containing string objects. aStub.getPropertyList()

To know if a property is *getable* (accessible) or *setable* (mutable),
call getPropertyAttribute() with the name of the property. The method
returns a Python TUPLE whose first element is true if the property is
setable, and the second element is true if it is getable. Attempts to
get a value from an inaccessible property and to set a value to a
immutable property result in exceptions. aStub.getPropertyAttribute(
'Activity' )[0] # ``true`` if setable aStub.getPropertyAttribute(
'Activity' )[1] # ``true`` if getable

Getting LOGGER data
~~~~~~~~~~~~~~~~~~~

To get logged data from a LOGGERSTUB, use getData() method.

getData() method has three forms according to requested range and time
resolution of the data:

-  getData()

   Get the whole data.

-  getData( ``starttime`` [, ``endtime``] )

   Get a slice of the data from ``starttime`` to ``endtime``. If
   ``endtime`` is omitted, the slice includes the tail of the data.

-  getData( ``starttime``, ``endtime``, ``interval`` )

   Get a slice of the data from ``starttime`` to ``endtime``. This omits
   data points if a time interval between two datapoints is smaller than
   ``interval``. This is not suitable for scientific data analysis, but
   optimized for speed.

getData() method returns a rank-2 (matrix) ARRAY object of NUMERICPYTHON
module. The ARRAY has either one of the following forms: [ [ time value
average min max ] [ time value average min max ] ... ] or [ [ time value
] [ time value ] ... ] The first five-tuple data format has five values
in a single datapoint:

-  time

   The time of the data point.

-  value

   The value at the time point.

-  average

   The time-weighted average of the value after the last data point to
   the time of this data point.

-  min

   The minimum value after the last data point to the time of this data
   point.

-  max

   The maximum value after the last data point to the time of this data
   point.

The two-tuple data format has only time and value.

To know the start time, the end time, and the size of the logged data
before getting data, use getStartTime(), getEndTime(), and getSize()
methods of LOGGERSTUB. getSize() returns the number of data points
stored in the LOGGER.

Getting and changing logging interval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Logging interval of a LOGGER can be checked and changed by using
getMinimumInterval() and setMinimumInterval( ``interval`` ) methods of
LOGGERSTUB. ``interval`` must be a zero or positive number in second. If
``interval`` is a non-zero positive number, the LOGGER skips logging if
a simulation step occurs before ``interval`` second past the last
logging time point. If ``interval`` is zero, the LOGGER logs at every
simulation step.

An example usage of an ENTITYSTUB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following example loads an EML file, and prints the value of ATP
VARIABLE in SYSTEM ``/CELL`` every 10 seconds. If the value is below
1000, it stops the simulation.

::

    loadModel( 'simple.eml' )

    ATP = createEntityStub( 'Variable:/CELL:ATP' )

    while 1:

        ATPValue = ATP[ 'Value' ]

        message( 'ATP value = %s' % ATPValue )

        if ATPValue <= 1000:
            break

        run( 10 )

    message( 'Stopped at %s.' % getCurrentTime() )

Handling Data Files
===================

About ECD file
--------------

ECELL SE uses ECD (E-Cell Data) file format to store simulation results.
ECD is a plain text file, and easily handled by user-written and
third-party data processing and plotting software such as gnuplot.

An ECD file can store a matrix of floating-point numbers.

ecell.ECDDataFile class can be used to save and load ECD files. A
ECDDataFile object takes and returns a rank-2 ARRAY of NUMERICPYTHON. A
'rank-2' ARRAY is a matrix, which means that Numeric.rank( ARRAY ) and
len( Numeric.shape( ARRAY ) ) returns '``2``\ '.

Importing ECDDataFile class
---------------------------

To import the ECDDataFile class, import the whole ecell module,

::

    import ecell

or import ecell.ECDDataFile module selectively.

::

    import ecell.ECDDataFile

Saving and loading data
-----------------------

To save data to an ECD file, say, ``datafile.ecd``, instantiate an
ECDDataFile object and use save() method. import ecell aDataFile =
ecell.ECDDataFile( DATA ) aDataFile.save( 'datafile.ecd' ) here ``DATA``
is a rank-2 ARRAY of NUMERICPYTHON or an equivalent object. The data can
also be set by using setData() method after the instantiation. If the
data is already set, it is replaced. aDataFile.setData( DATA )

Loading the ECD file is also straightforward. aDataFile =
ecell.ECDDataFile() aDataFile.load( 'datafile.ecd' ) DATA =
aDataFile.getData() The getData() method extracts the data from the
ECDDataFile object as an ARRAY.

ECD header information
----------------------

In addition to the data itself, an ECD file can hold some information in
its header.

-  Data name

   The name of data. Setting a *FullPN* may be a good idea. Use
   setDataName( ``name`` ) and getDataName() methods to set and get this
   field.

-  Label

   This field is used to name axes of the data. Use setLabel( ``labels``
   ) and getLabel() methods. These methods takes and returns a PYTHON
   TUPLE, and stored in the file as a space-separated list. The default
   value of this field is: ``( 't', 'value', 'avg', 'min', 'max' )``.

-  Note

   This is a free-format field. This can be a multi-line or a
   single-line string. Use setNote( ``note`` ) and getNote().

The header information is stored in the file like this.

::

    #DATA:
    #SIZE: 5 1010
    #LABEL: t       value   avg     min     max
    #NOTE:
    #
    #----------------------
    0 0.1 0.1 0.1 0.1
    ...

Each line of the header is headed by a sharp (#) character. The
``'#SIZE:'`` line is automatically set when saved to show size of the
data. This field is ignored in loading. The header ends with
``'#----...'``.

Using ECD outside ECELL SE
--------------------------

For most cases NUMERICPYTHON will offer any necessary functionality for
scientific data processing. However, using some external software can
enhance the usability.

ECD files can be used as input to any software which supports white
space-separated text format, and treats lines with heading sharps (#) as
comments.

GNU gnuplot is a scientific presentation-quality plotting software with
a sophisticated interactive command system. To plot an ECD file from
gnuplot, just use ``plot`` command. For example, this draws a time-value
2D-graph: ``gnuplot> ``\ ``plot 'datafile.ecd' with lines`` Use
``using`` modifier to specify which column to use for the plotting. The
following example makes a time-average 2D-plot.
``gnuplot> ``\ ``plot 'datafile.ecd' using 1:3 with lines``

Another OpenSource software useful for data processing is GNU Octave.
Loading an ECD from Octave is also simplest. ``octave:1>``
``load datafile.ecd`` Now the data is stored in a matrix variable with
the same name as the file without the extension (``datafile``).
``octave:2> ``\ ``mean(datafile)`` ``ans =
 
   5.0663  51.7158  51.7158  51.2396  52.2386``

Binary format
-------------

Currently loading and saving of the binary file format is not supported.
However, Numeric Python has an efficient, platform-dependent way of
exporting and importing ARRAY data. See the Numeric Python manual.

Manipulating Model Files
========================

This section describes how to create, modify, and read EML files with
the EML module of the ECELL PYTHON library.

Importing EML module
--------------------

To import the EML module, just import ecell module.

::

    import ecell

And ecell.Eml class is made available.

Other Methods
=============

Getting version numbers
-----------------------

getLibECSVersion() method of ecell.ecs module gives the version of the
C++ backend library (libecs) as a string. getLibECSVersionInfo() method
of the module gives the version as a PYTHON TUPLE. The TUPLE contains
three numbers in this order: ( ``MAJOR_VERSION``, ``MINOR_VERSION``,
``MICRO_VERSION`` )

::




DM loading-related methods
--------------------------

The search path of DM files can be specified and retrieved by using
setDMSearchPath() and getDMSearchPath() methods. These methods gets and
returns a colon (:) separated list of directory names. The search path
can also be specified by using ECELL3\_DM\_PATH environment variable.
See the previous section for more about DMsearch path.

::



A list of built-in and already loaded DM classes can be gotten with
getDMInfo() method of ecell.ecs.Simulator class. The SIMULATOR instance
is available to SESSION as ``theSimulator`` variable. The method returns
a nested PYTHON TUPLE in the form of ( ( TYPE1, CLASSNAME1, PATH1 ), (
TYPE2, CLASSNAME2, PATH2 ), ... ). TYPE is one of ``'Process'``,
``'Variable'``, ``'System'``, or ``'Stepper'``. CLASSNAME is the class
name of the DM. PATH is the directory from which the DM is loaded. PATH
is an empty string (``''``) if it is a built-in class.

::


Advanced Topics
===============

How ECELL3-SESSION runs
-----------------------

ECELL3-SESSION command runs on ECELL3-PYTHON interpreter command.
ECELL3-PYTHON command is a thin wrapper to the PYTHON interpreter.
ECELL3-PYTHON command simply invokes a PYTHON interpreter command
specified at compile time. Before executing PYTHON, ECELL3-PYTHON sets
some environment variables to ensure that it can find necessary ECELL
PYTHON extension modules and the Standard DM Library. After processing
the commandline options, ECELL3-SESSION command creates an
ecell.ecs.Simulator object, and then instantiate a ecell.Session object
for the simulator object.

Thus basically ECELL3-PYTHON is just a PYTHON interpreter, and frontend
components of ECELL SE run on this command. To use the ECELL Python
Library from ECELL3-PYTHON command, use

::

    import ecell

statement from the prompt: ``$ ``\ ``ecell3-python``
``Python 2.2.2 (#1, Feb 24 2003, 19:13:11)
[GCC 3.2.2 20030222 (Red Hat Linux 3.2.2-4)] on linux2
Type "help", "copyright", "credits" or "license" for more information.``
``>>> ``\ ``import ecell`` ``>>> `` or, (on UNIX-like systems) execute a
file starting with:

::

    #!/usr/bin/env ecell3-python
    import ecell
    [...]

Getting information about execution environment
-----------------------------------------------

To get the current configuration of ECELL3-PYTHON command, invoke
ECELL3-PYTHON command with a ``-h`` option. This will print values of
some variables as well as usage of the command.
``$ ``\ ``ecell3-python -h`` ``[...]

Configurations:
 
        PACKAGE         = ecell
        VERSION         = 3.2.0
        PYTHON          = /usr/bin/python
        PYTHONPATH      = /usr/lib/python2.2/site-packages:
        DEBUGGER        = gdb
        LD_LIBRARY_PATH = /usr/lib:
        prefix          = /usr
        pythondir       = /usr/lib/python2.2/site-packages
        ECELL3_DM_PATH  =

[...]
`` The '``PYTHON =``\ ' line gives the path of the PYTHON interpreter to
be used.

Debugging
---------

To invoke ECELL3-PYTHON command in debugging mode, set ECELL\_DEBUG
environment variable. This runs the command on a debugger software. If
found, GNU gdb is used as the debugger. ECELL\_DEBUG can be used for any
commands which run on ECELL3-PYTHON, including ECELL3-SESSION and
GECELL. For example, to run ECELL3-SESSION in debug mode on the shell
prompt: ``$ ``\ ``ECELL_DEBUG=1 ecell3-session -f foo.eml``
``gdb --command=/tmp/ecell3.0mlQyE /usr/bin/python
GNU gdb Red Hat Linux (5.3post-0.20021129.18rh)
Copyright 2003 Free Software Foundation, Inc.
GDB is free software, covered by the GNU General Public License, and you are
welcome to change it and/or distribute copies of it under certain conditions.
Type "show copying" to see the conditions.
There is absolutely no warranty for GDB.  Type "show warranty" for details.
This GDB was configured as "i386-redhat-linux-gnu"...
[New Thread 1074178112 (LWP 7327)]
ecell3-session [ E-Cell SE Version 3.2.0, on Python Version 2.2.2 ]
Copyright (C) 1996-2003 Keio University.
Send feedback to Koichi Takahashi <shafi@e-cell.org>``
``<foo.eml, t=0>>> ``\ ```` ``Program received signal SIGINT, Interrupt.
[Switching to Thread 1074178112 (LWP 7327)]
0xffffe002 in ?? ()`` ``(gdb)`` It automatically runs the program with
the commandline options with '``--command=``\ ' option of gdb. The gdb
prompt appears when the program crashes or interrupted by the user by
pressing Ctrl C.

ECELL\_DEBUG runs gdb, which is operates at the level of C++ code. For
debugging of PYTHON layer scripts, see PYTHON Library Reference Manual
for Python Debugger.

Profiling
---------

It is possible to run ECELL3-PYTHON command in profiling mode, if the
operating system has GNU sprof command, and its C library supports
LD\_PROFILE environmental variable. Currently it only supports
per-shared object profiling. (See GNU C Library Reference Manual)

To run ECELL3-PYTHON in profiling mode, set ECELL\_PROFILE environment
variable to *SONAME* of the shared object. SONAME of a shared object
file can be found by using objdump command, with, for example, ``-p``
option.

For example, the following commandline takes a performance profile of
Libecs: ``$ ``\ ``ECELL_PROFILE=libecs.so.2 ecell3-session [...]`` After
running, it creates a profiling data file with a filename
``SONAME.profile`` in the *current directory*. In this case, it is
``libecs.so.2.profile``. The binary profiling data can be converted to a
text format by using ``sprof`` command. For example:
``$ ``\ ``sprof -p libecs.so.2 libecs.so.2.profile``

ECELL Python Library API
========================

This section provides a list of some commonly used classes in ECELL
Python library and their APIs.

SESSION Class API
-----------------

Methods of SESSION class has the following five groups.

-  Session methods

-  Simulation methods

-  Stepper methods

-  Entity methods

-  Logger methods

SESSION-CLASS-API
OBJECTSTUB Classes API
----------------------

There are three subclasses of OBJECTSTUB

-  ENTITYSTUB

-  STEPPERSTUB

-  LOGGERSTUB

Some methods are common to these subclasses.

OBJECTSTUBS-API
ECDDataFile Class API
---------------------

ECDDataFile class has the following set of methods.

ECDDATAFILE-API
