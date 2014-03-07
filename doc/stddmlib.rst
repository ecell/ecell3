===============================
Standard Dynamic Module Library
===============================

This chapter overviews:

An incomplete list of classes available as the Standard Dynamic Module
Library, and,
Some usage the classes in the Standard Dynamic Module Library.
This chapter briefly describes the Standard Dynamic Module Library
distributed with APP. If the system is installed correctly, the classes
provided by the library can be used without any special procedure.

This chapter is not meant to be a complete reference. To know more about
the classes defined in the library, see the E-Cell3 Standard Dynamic
Module Library Reference Manual (under preparation).

Steppers
========

There are three direct sub-classes of STEPPER: DifferentialStepper,
DiscreteEventStepper, DiscreteTimeStepper

DifferentialSteppers
--------------------

General-purpose DifferentialStepper classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following STEPPER classes implement general-purpose ordinary
differential equation solvers. Basically these classes must work well
with any simple continuous PROCESS classes.

-  ODE45Stepper

   This STEPPER implements Dormand-Prince 5(4)7M algorithm for ODE
   systems.

   In most cases this STEPPER is the best general purpose solver for ODE
   models.

-  ODE23Stepper

   This STEPPER implements Fehlberg 2(3) algorithm for ODE systems.

   Try this STEPPER if other part of the model has smaller timescales.
   This STEPPER can be used for a moderately stiff systems of
   differential equations.

-  FixedODE1Stepper

   A DifferentialStepper without adaptive stepsizing mechanism. The
   solution of this STEPPER is first order.

   This stepper calls process() method of each PROCESS just once in a
   single step.

   Although this STEPPER is not suitable for high-accuracy solution of
   smooth continuous systems of differential equations, its simplicity
   of the algorithm is sometimes useful.

S-System and GMA Steppers
~~~~~~~~~~~~~~~~~~~~~~~~~

FIXME: need description here.

DiscreteEventSteppers
---------------------

-  DiscreteEventStepper

   This STEPPER is used to conduct discrete event simulations. This
   STEPPER should be used in combination with subclasses of
   DiscreteEventProcess.

   This STEPPER uses its PROCESS objects as event generators. The
   procedure of this STEPPER for initialize() method is like this:

   1. updateStepInterval() method of its all DiscreteEventProcess
      objects.

   2. Find a PROCESS with the least *scheduled time* (top process). The
      scheduled time is calculated as: ( current time ) + ( StepInterval
      of the process ).

   3. Reschedule itself to the scheduled time of the top process.

   step() method of this STEPPER is as follows:

   1. Call process() method of the current top process.

   2. Calls updateStepInterval() method of the top process and all
      *dependent processes* of the top process, and update scheduled
      times for those processes to find the new top process.

   3. Lastly the STEPPER reschedule itself to the scheduled time of the
      new top process.

   The procedure for interrupt() method of this class is the same as
   that for initialize(). FIXME: need to explain about TimeScale
   property.

-  NRStepper

   This is an alias to the DiscreteEventStepper. This class can be used
   as an implementation of Gillespie-Gibson algorithm.

   To conduct the Gillespie-Gibson simulation, use this class of STEPPER
   in combination with GillespieProcess class. GillespieProcess is a
   subclass of DiscreteEventProcess.

DiscreteTimeStepper
-------------------

-  DiscreteTimeStepper

   This STEPPER steps with a fixed interval. For example, StepInterval
   property of this STEPPER is set to ``0.1``, this STEPPER steps every
   0.1 seconds.

   When this STEPPER steps, it calls process() of all of its PROCESS
   instances. To change this behavior, create a subclass.

   This STEPPER ignores incoming interruptions from other STEPPERs.

PassiveStepper
--------------

-  PassiveStepper

   This STEPPER never steps spontaneously (step interval = infinity).
   Instead, this STEPPER steps upon interruption. In other words, this
   STEPPER steps everytime immediately after a dependent STEPPER steps.

   When this STEPPER steps, it calls process() of all of its PROCESS
   instances. To change this behavior, create a subclass.

Process classes
===============

Continuous Process classes
--------------------------

Differential equation-based Process classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following PROCESS classes are straightforward implementations of
differential equations, and can be used with the general-purpose
DifferentialSteppers such as ODE45Stepper, ODE23Stepper, and
FixedODE1Stepper.

In the current version, most of the classes represent certain reaction
rate equations. Of course it is not limited to chemical and biochemical
simulations.

-  CatalyzedMassActionFluxProcess

-  DecayFluxProcess

-  IsoUniUniFluxProcess

-  MassActionProcess

-  MichaelisUniUniProcess

-  MichaelisUniUniReversibleProcess

-  OrderedBiBiFluxProcess

-  OrderedBiUniFluxProcess

-  OrderedUniBiFluxProcess

-  PingPongBiBiFluxProcess

-  RandomBiBiFluxProcess

-  RandomBiUniFluxProcess

-  RandomUniBiFluxProcess

Other continuous Process classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  PythonFluxProcess

-  SSystemProcess

Discrete Process classes
------------------------

-  GammaProcess

   Under development.

-  GillespieProcess

   This PROCESS must be used with a Gillespie-type STEPPER, such as
   NRStepper.

-  RapidEquilibriumProcess

Other Process classes
---------------------

-  PythonProcess

Variable classes
================

-  Variable

   A standard class to represent a state variable.


