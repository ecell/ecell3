Stepper DiscreteTimeStepper( DTS01 )
{
    StepInterval 1.0;
}

System System( / )
{
    StepperID DTS01;

    Variable Variable( X ) { Value 0.5; }
#     Variable Variable( X ) { Value 0.0; }

    Process ExpressionAssignmentProcess( R )
    {
        a 0.0;
        Expression "a * X.Value * (1.0 - X.Value)";
        VariableReferenceList [X :.:X +1];
    }
}
