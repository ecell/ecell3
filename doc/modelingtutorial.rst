=================
Modeling Tutorial
=================

This chapter is a simple modeling tutorial using ECELL.

Running the model
=================

All the examples in this section can be run by the following procedure.

1. Create and save the model file (for example, ``simple-gillespie.em``)
   with a text editor.

2. Convert the EM file to an EML file by ``ecell3-em2eml`` command.

   ::

        

3. Run it in GUI mode with ``gecell`` command.

   ::

        

   or, in the script mode with ``ecell3-session`` command (see the
   following chapter):

   ::

        

Using Gillespie algorithm
=========================

APP comes with a set of classes for simulations using Gillespie's
stochastic algorithm.

A Trivial Reaction System
-------------------------

At the very first, let us start with the simplest possible stable system
of elementary reactions, which has two variables (in this case the
numbers of molecules of molecular species) and a couple of elementary
reaction processes. Because elementary reactions are irreversible, at
least two instances of the reactions are needed for the system to be
stable. The reaction system looks like this: -- P1 --> S1 S2 <-- P2 --
``S1`` and ``S2`` are molecular species, and ``P1`` and ``P2`` are
reaction processes. Rate constants of both reactions are the same: 1.0
[1/s]. Initial numbers of molecules of ``S1`` and ``S2`` are 1000 and 0,
respectively. Because rate constants are the same, the system has a
steady state at ``S1 == S2 == 500``.

Specifying the Next Reaction method
-----------------------------------

NRStepper class implements Gibson's efficient variation of the Gillespie
algorithm, or the Next Reaction (NR) method.

To use the NRStepper in your simulation model, write like this in your
EM file:

::

    Stepper NRStepper( NR1 )
    {
        # no property
    }

In this example, the NRStepper has the StepperID '``NR1``\ '. For now,
no property needs to be specified for this object.

Defining the compartment
------------------------

Next, define a compartment, and connect it to the STEPPER ``NR1``.
Because this model has only a single compartment, we use the root sytem
(``/``). Although this model does not depend on size of the compartment
because all reactions are first order, it is a good idea to always
define the size explicitly rather than letting it defaults to ``1.0``.
Here we set it to ``1e-15`` [L].

::

    System System( / )
    {
        StepperID       NR1;

        Variable Variable( SIZE ) { Value 1e-15; }

        # ...
    }

Defining the variables
----------------------

Now define the main variables ``S1`` and ``S2``. Use 'Value' properties
of the objects to set the initial values.

::

    System System( / )
    {
        # ...

        Variable Variable( S1 )
        {
            Value   1000;
        }
            
        Variable Variable( S2 )
        {
            Value   0;
        }
            
        # ...
    }

Defining reaction processes
---------------------------

Lastly, create reaction process instances ``P1`` and ``P2``.
GillespieProcess class works together with the NRStepper to simulate
elementary reactions.

Two different types of properties, k and VariableReferenceList, must be
set for each of the GillespieProcess object. k is the rate constant
parameter in [1/sec] if the reaction is first-order, or [1/sec/M] if it
is a second-order reaction. (Don't forget to define ``SIZE`` VARIABLE if
there is a second-order reaction.) Set VariableReferenceList properties
so that ``P1`` consumes ``S1`` and produce ``S2``, and ``P2`` uses
``S2`` to create ``S1``.

::

    System System( / )
    {
        # ...

        Process GillespieProcess( P1 )              # the reaction S1 --> S2
        {
            VariableReferenceList   [ S :.:S1 -1 ]
                                    [ P :.:S2 1 ];
            k       1.0;                            # the rate constant
        }

        Process GillespieProcess( P2 )              # the reaction S2 --> S1
        {
            VariableReferenceList   [ S :.:S2 -1 ]
                                    [ P :.:S1 1 ];
            k       1.0;
        }
    }

Putting them together
---------------------

Here is the complete EM of the model that really works. Run this model
with ``gecell`` and open a TracerWindow to plot trajectories of ``S1``
and ``S2``. You will see those two VARIABLEs immediately reaching the
steady state around 500.0. If you zoom around the trajectories, you will
be able to see stochastic fluctuations.

::

    Stepper NRStepper( NR1 )
    {
        # no property
    }

    System System( / )
    {
        StepperID       NR1;

        Variable Variable( SIZE ) { Value 1e-15; }

        Variable Variable( S1 )
        {
            Value   1000;
        }
            
        Variable Variable( S2 )
        {
            Value   0;
        }
            
        Process GillespieProcess( P1 )              # the reaction S1 --> S2
        {
            VariableReferenceList   [ S :.:S1 -1 ]
                                    [ P :.:S2 1 ];
            k       1.0;                            # the rate constant
        }

        Process GillespieProcess( P2 )              # the reaction S2 --> S1
        {
            VariableReferenceList   [ S :.:S2 -1 ]
                                    [ P :.:S1 1 ];
            k       1.0;
        }
    }

Using Deterministic Differential Equations
==========================================

The previous section described how to create a model that runs with the
stochastic Gillespie's algorithm. ECELL is a multi-algorithm simulator,
and different algorithms can be used to run the model. This section
explains a way to use a deterministic differential equation solver to
run the system of simple mass-action reactions.

Choosing Stepper and Process classes
------------------------------------

In the current version, we recommend ODE45Stepper class as a
general-purpose STEPPER for differential equation systems. This STEPPER
implements an explicit fourth order numerical integration algorithm with
a fifth-order error control.

MassActionFluxProcess is the continuous differential equation conterpart
of the discrete-event GillespieProcess. Unlike GillespieProcess,
MassActionFluxProcess does not have limitation in the order of the
reaction mechanism. For example, it can handle the reaction like this:
``S0 + S1 + 2 S2 --> P0 + P1``.

Converting the model
--------------------

Converting the trivial reaction system model for Gillespie to use
differential equations is very easy; just replace NRStepper with
ODE45Stepper, and change the classname of GillespieProcess to
MassActionFluxProcess.

The following is the model of the trivial model that runs on the
differential ODE45Stepper. You will get similar simulation result than
the stochastic model in the previous section. However, if you zoom, you
would notice that the stochastic fluctuation can no longer be observed
because the model is turned from stochastic to deterministic.

::

    Stepper ODE45Stepper( ODE1 )
    {
        # no property
    }

    System System( / )
    {
        StepperID       ODE1;

        Variable Variable( SIZE ) { Value 1e-15; }

        Variable Variable( S1 )
        {
            Value   1000;
        }
            
        Variable Variable( S2 )
        {
            Value   0;
        }
            
        Process MassActionFluxProcess( P1 )
        {
            VariableReferenceList   [ S0 :.:S1 -1 ]
                                    [ P0 :.:S2 1 ];
            k       1.0;
        }

        Process MassActionFluxProcess( P2 )
        {
            VariableReferenceList   [ S0 :.:S2 -1 ]
                                    [ P0 :.:S1 1 ];
            k       1.0;
        }
    }

Making the Model Switchable Between Algorithms
==============================================

Fortunately, at least as far as the model has only elementary reactions,
switching between these stochastic and deterministic algorithms is just
to switch between NRStepper/GillespieProcess pair and
ODE45Stepper/MassActionFluxProcess pair of classnames. Both PROCESS
classes takes the same property 'k' with the same unit.

Some use of EMPY macros makes the model generic. In the following
example, setting the PYTHON variable ``TYPE`` to ``ODE`` makes it run in
deterministic differential equation mode, and setting ``TYPE`` to ``NR``
turns it stochastic.

::

    @{ALGORITHM='ODE'}

    @{
    if ALGORITHM == 'ODE':
        STEPPER='ODE45Stepper'
        PROCESS='MassActionFluxProcess'
    elif ALGORITHM == 'NR':
        STEPPER='NRStepper'
        PROCESS='GillespieProcess'
    else:
        raise 'unknown algorithm: %s' % ALGORITHM
    }

    Stepper @(STEPPER)( STEPPER1 )
    {
        # no property
    }

    System System( / )
    {
        StepperID       STEPPER1;

        Variable Variable( SIZE ) { Value 1e-15; }

        Variable Variable( S1 )
        {
            Value   1000;
        }
            
        Variable Variable( S2 )
        {
            Value   0;
        }
            
        Process @(PROCESS)( P1 )
        {
            VariableReferenceList   [ S0 :.:S1 -1 ]
                                    [ P0 :.:S2 1 ];
            k       1.0;
        }

        Process @(PROCESS)( P2 )
        {
            VariableReferenceList   [ S0 :.:S2 -1 ]
                                    [ P0 :.:S1 1 ];
            k       1.0;
        }
    }

A Simple Deterministic / Stochastic Composite Simulation
========================================================

ECELL can drive a model with multiple STEPPER objects. Each STEPPER can
implement different simulation algorithms, and have different time
scales. This framework of multi-algorithm, multi-timescale simulation is
quite generic, and virtually any combination of any number of different
types of sub-models is possible. This section exemplifies a tiny model
of coupled ODE and Gillespie reactions.

A tiny multi-timescale reactions model
--------------------------------------

Consider this tiny model of four VARIABLEs and six reaction PROCESSes:
-- P1 --> -- P4 --> S1 S2 -- P3 --> S3 S4 ^ <-- P2 -- <-- P5 -- \| \| \|
\\ \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ P6
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_/ Although it may look
complicated at first glance, this system consists of two instances of
the 'trivial' system we have modeled in the previous sections coupled
together: Sub-model 1: -- P1 --> S1 S2 <-- P2 -- and Sub-model 2: -- P4
--> S3 S4 <-- P5 -- These two sub-models are in turn coupled by reaction
processes ``P3`` and ``P6``. Because time scales of ``P3`` and ``P6``
are determined by ``S2`` and ``S4``, respectively, ``P3`` belongs to the
sub-model 1, and ``P6`` is a part of the sub-model 2. Sub-model 1: S2 --
P3 --> S3 S1 <-- P6 --> S4 :Sub-model 2 Rate constants of the main
reactions, ``P1``, ``P2``, ``P4``, and ``P5`` are the same as the
previous model: ``1.0`` [1/sec]. But the 'bridging' reactions are slower
than the main reactions: ``1e-1`` for ``P3`` and ``1e-3`` for ``P6``.
Consequently, sub-models 1 and 2 would have approximately
``1e-1 / 1e-3 == 1e-2`` times different steady-state levels. Because the
rate constants of the main reactions are the same, this implies time
scale of both sub-models are different.

Writing model file
------------------

The following code implements the multi-time scale model. The first two
lines specify algorithms to use for those two parts of the model.
``ALGORITHM1`` variable specifies the algorithm to use for the sub-model
1, and ``ALGORITHM2`` is for the sub-model 2. Values of these variables
can either be ``'NR'`` or ``'ODE'``.

For example, to try pure-stochastic simulation, set these variables like
this:

::

    @{ALGORITHM1='NR'}
    @{ALGORITHM2='NR'}

Setting ``ALGORITHM1`` to ``'NR'`` and ``ALGORITHM2`` to ``'ODE`` would
be an ideal configuration. This runs a magnitude faster than the
pure-stochastic configuration.

::

    @{ALGORITHM1='NR'}
    @{ALGORITHM2='ODE'}

Also try pure-deterministic run.

::

    @{ALGORITHM1='ODE'}
    @{ALGORITHM2='ODE'}

In this particular model, this configuration runs very fast because the
system easily reaches the steady-state and stiffness of the model is
low. However, this does not necessary mean pure-ODE is always the
fastest. Under some situations NR/ODE composite simulation exceeds both
pure-stochastic and pure-deterministic (reference?).

::

    @{ALGORITHM1= ['NR' or 'ODE']}
    @{ALGORITHM2= ['NR' or 'ODE']}


    # a function to give appropriate class names.
    @{
    def getClassNamesByAlgorithm( anAlgorithm ):
        if anAlgorithm == 'ODE':
            return 'ODE45Stepper', 'MassActionFluxProcess'
        elif anAlgorithm == 'NR':
            return 'NRStepper', 'GillespieProcess'
        else:
            raise 'unknown algorithm: %s' % ALGORITHM1
    }

    # get classnames
    @{
    STEPPER1, PROCESS1 = getClassNamesByAlgorithm( ALGORITHM1 )
    STEPPER2, PROCESS2 = getClassNamesByAlgorithm( ALGORITHM2 )
    }

    # create appropriate steppers.
    # stepper ids are the same as the ALGORITHM.
    @('Stepper %s ( %s ) {}' % ( STEPPER1, ALGORITHM1 ))

    # if we have two different algorithms, one more stepper is needed.
    @(ALGORITHM1 != ALGORITHM2 ? 'Stepper %s( %s ) {}' % ( STEPPER2, ALGORITHM2 ))



    System CompartmentSystem( / )
    {
        StepperID   @(ALGORITHM1);
        
        Variable Variable( SIZE ) { Value 1e-15; }

        Variable Variable( S1 )
        {
            Value   1000;
        }
        
        Variable Variable( S2 )
        {
            Value   0;
        }
        
        Variable Variable( S3 )
        {
            Value   1000000;
        }
        
        Variable Variable( S4 )
        {
            Value   0;
        }


        Process @(PROCESS1)( P1 )
        {
            VariableReferenceList   [ S0 :.:S1 -1 ] [ P0 :.:S2 1 ];
            k       1.0;
        }

        Process @(PROCESS1)( P2 )
        {
            VariableReferenceList   [ S0 :.:S2 -1 ] [ P0 :.:S1 1 ];
            k       1.0;
        }

        Process @(PROCESS1)( P3 )
        {
            VariableReferenceList   [ S0 :.:S2 -1 ] [ P0 :.:S3 1 ];
            k       1e-1;
        }

        Process @(PROCESS2)( P4 )
        {
            StepperID @(ALGORITHM2);

            VariableReferenceList   [ S0 :.:S3 -1 ] [ P0 :.:S4 1 ];
            k       1.0;
        }

        Process @(PROCESS2)( P5 )
        {
            StepperID @(ALGORITHM2);

            VariableReferenceList   [ S0 :.:S4 -1 ] [ P0 :.:S3 1 ];
            k       1.0;
        }

        Process @(PROCESS2)( P6 )
        {
            StepperID @(ALGORITHM2);

            VariableReferenceList   [ S0 :.:S4 -1 ] [ P0 :.:S1 1 ];
            k       1e-4;
        }
        
    }

Custom equations
================

Complex flux rate equations
---------------------------

The simplest way to script custom rate equations is to use
ExpressionFluxProcess. Here is an example taken from the Drosophila
sample model which you can find under
``${datadir}/doc/ecell/samples/Drosophila``  [1]_ In this expression,
Size \* N\_A of the supersystem of the PROCESS is used to keep the unit
of the expression [ num / second ].

::

    Process ExpressionFluxProcess( R_toy1 )
    {
        vs      0.76;
        KI      1;
        Expression "(vs*KI) / (KI + C0.MolarConc ^ 3) 
                                           * self.getSuperSystem().SizeN_A";

        VariableReferenceList   [ P0 :.:M 1 ] [ C0 :.:Pn 0 ];
    }

FIXME: some more examples

Algebraic equations
-------------------

Use of ExpressionAlgebraicProcess is the easiest method to describe
algebraic equations.

Be careful about the coefficients of the VARIABLEREFERENCEs. (Usually
just set unities.)

FIXME: some more examples here

Other Modeling Schemes
======================

Discrete events
---------------

.. [1]
   ``${datadir}`` refers to the directory either given to ``--datadir``
   option of ``configure`` script or ``${prefix}/share``. On Windows,
   ``${prefix}`` would be the directory to which the application is
   installted.
