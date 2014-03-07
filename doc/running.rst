Getting Started
===============

By reading this chapter, you can get information about the following
items.

What types of files are needed to run the simulator.
How to prepare the files needed to run the simulator.
How to run the simulation with APP.

Preparing Simulation
---------------------

To start the simulation, you need to have these types of files:

-  A model file in EML format.

-  (optionally) shared object files (.so in Linux operating system), if
   you are using special classes of object in the model file which is
   not provided by the system by default.

-  (optionally) a script file (ECELL Session Script, or ESS) to automate
   the simulation session.

Converting EM to EML
~~~~~~~~~~~~~~~~~~~~~~

Simulation models for ECELL is often written in EM format. To convert EM
(.em) files to EML (.eml) files, type the following command.

``ecell3-em2eml`` filename.em

You can obtain the full description of the command line options giving
-h option to ``ecell3-em2eml``.

::

    ecell3-eml2em -- convert eml to em
             
    Usage:
            ecell3-eml2em [-h] [-f] [-o outfile] infile
     
             
    Options:
            -h or --help    :  Print this message.
            -f or --force   :  Force overwrite even if outfile already exists.
            -o or --outfile=:  Specify output file name.  '-' means stdout.

Compiling C++ Dynamic Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You might have some Dynamic Modules (or DM in short) specifically made
for the simulation model in the form of C++ source code. If that is the
case, those files have to be compiled and linked to form shared module
files (usually suffixed by ``.so`` on Unix-like platforms, ``.dylib`` on
Mac OS X or ``.dll`` on Windows) before running the simulation. You will
also need to set ``ECELL3_DM_PATH`` environment variable to the
appropriate value to use the DMs (discussed below).

To compile and like DMs, ``ecell3-dmc`` command is provided for
convenience.

::

     
     

The arguments given before the file name (``[command
      options]`` are interpreted as options to the ``ecell3-dmc``
command itself.

The arguments after the file name are passed to a backend compiler (such
as g++) as-is. The backend compiler used is the same as that used to
build the system itself.

To inspect what the command actually does inside, enable verbose mode by
specifying ``-v`` option.

To get a full list of available ``ecell3-dmc`` options, invoke the
command with ``-h`` option, and without the input file. Here is the help
message shown by issuing ``ecedll3-dmc`` ``-h``. Compile dynamic modules
for E-Cell Simulation Environment Versin 3. Usage: ecell3-dmc [
ecell3-dmc options ] sourcefile [ compiler options ] ecell3-dmc
-h\|--help ecell3-dmc options: --no-stdinclude Don't set standard
include file path. --no-stdlibdir Don't set standard include file path.
--ldflags=[ldflags] Specify options to the linker. --cxxflags=[cxxflags]
Override the default compiler options. --dmcompile=[path] Specify
dmcompile path. -v or --verbose Be verbose. -h or --help Print this
message.

Starting APP
-------------

You can start APP either in scripting mode and GUI mode.

GUI mode
~~~~~~~~~~

To start APP in GUI mode, type the following command.

::

     &

This will invoke an instance of the simulator with Osogo Session Manager
attached as a GUI frontend.

Scripting mode
~~~~~~~~~~~~~~~~

To start APP in scripting mode, type the following command:

::

     

where filename.ess is the name of the Python script file you want to
execute.

If filename.ess is omitted, the interpreter starts up in interactive
mode.

See chapter 5 for the scripting feature.

DM search path and ``ECELL3_DM_PATH`` environment variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your model makes use of non-standard DMs that you had to build using
``ecell3-dmc``, then you need to specify the directory where the DMs are
placed in ``ECELL3_DM_PATH`` environment variable. ``ECELL3_DM_PATH``
can have multiple directory names separated by either ``:`` (colon) on
Unix-like platform or ``;`` (semicolon) on Windows.

The following is an example of setting ``ECELL3_DM_PATH`` before
launching ``ecell3-session-monitor``:

::

     
     
     
              

Note that up to E-Cell SE 3.1.105, the current working directory was
implicitly treated as if it was included in ``ECELL3_DM_PATH``. This
quirk is removed since 3.1.106.
