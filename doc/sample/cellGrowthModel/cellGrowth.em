Stepper ODEStepper( ODE_S1 )
{
        # no property
}

Stepper DiscreteEventStepper( DEvent_S1 )
{
        # no property
}

Stepper DiscreteTimeStepper( DTime_1 )
{
        StepInterval .1;
}

System System( / )
{
        Variable Variable( SIZE )
        {
                Value 1e-5; # 10 micrometeters
        }
        
        # Some external media.

        Variable Variable( A )
        {
                MolarConc 5e-7; # .5 micromolar.
        }

        VariableVariable( B )
        {

                MolarConc 5e-7;
        }


        System System( Cell_0 )
        {
                Variable Variable( SIZE )
                {
                        Value 1e-9;
                }      

               # Something in here representing something....
               
               A->B ODE 
               B->C Gillespie
               C makes grow

               Process CellDivisionProcess
               {
                DivideWhenConditionMet "DeathMol.MolarConc() > 1.0"
                VariableReferenceList [DeathMol :.:C 1]
                                      [Dum1 :.:A -1]               }
               
               Variable Variable( A )
               {
                MolarConc 1.0;
               }
               
        }

        
