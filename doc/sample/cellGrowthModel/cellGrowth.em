Stepper ODEStepper( ODE_S1 )
{
        # no property
}

Stepper DiscreteEventStepper( DEvent_S1 )
{
        # no property
}

Stepper DiscreteTimeStepper( DTime_S1 )
{
        StepInterval .1;
}        

# This is the external media.
System System( / )
{
        StepperID ODE_S1;
}

System System( /MEDIA )
{
        StepperID ODE_S1;

        Process GillespieProcess( BoundReceptorInternalization )
        {
                StepperID DEvent_S1;

                k .00000001;  
                VariableReferenceList [BoundReceptor :.:DUMMY -1];
        }

        Variable Variable( DUMMY )
        {
                Value 0;
        }

        Variable Variable( SIZE )
        {
                Value 1.0; # 10 micrometeters
        }
        
        Variable Variable( Cell_Growth_Factor )
        {
                MolarConc 5e-7; # .5 micromolar.
        }
        
        Variable Variable( Waste )
        {
                MolarConc 0.0;
        }

        Process PythonProcess( Growth_Factor_Synthesis )
        {
                # Improve derivative.
                FireMethod "self.setFlux( 10.0 )";
                VariableReferenceList [A :.:Cell_Growth_Factor 1];
        }

}


System System( /MEDIA/CELL0 )
{
        StepperID ODE_S1;

        Variable Variable( SIZE )
        {
                Value 1e-10;
        }

        Variable Variable( Cell_Growth_Factor )
        {
                Value 0;
        }

        Variable Variable( Cell_Waste )
        {
                Value 0;
        }

        Variable Variable( Receptor )
        {
                Value 10000;
        }

        Variable Variable( Bound_Receptor )        
        {
                Value 0;
        }

        
        Variable Variable( C )
        {
                Value 100000;
        }

        Variable Variable( D )
        {
                Value 0;
        }
        
        Process PythonProcess( ImportFood )
        {

                # The cell ingests food...

                VariableReferenceList [ExternalFood :/MEDIA:Cell_Growth_Factor -1]
                                      [InternalFood :.:Cell_Growth_Factor 1];

                FireMethod "self.setFlux(0)";
        }
        
        Process PythonProcess( ExcrementProcess )
        {
                # It excretes wastes....
                VariableReferenceList [InternalWaste :.:Cell_Waste -1]
                                      [ExternalWaste :/MEDIA:Waste 1];
                
                FireMethod "0";
        }
                              

        Process PythonProcess( GrowProcess )
        {

                # It grows by eating food, which increases the volume, sensitivity
                # to waste, and internal wastes...

                VariableReferenceList [Food :.:Cell_Growth_Factor -1]
                                      [Volume :.:SIZE 1]      
                                      [Waste :.:Cell_Waste 1]
                                      [ApoptosisReceptors :.:Receptor 1];

                FireMethod "1.0";                                      
        }


        Process MassActionFluxProcess( ReceptorToExternalBBinding )
        {
                k .1;
                VariableReferenceList [A :.:Receptor -1]
                                      [B :/MEDIA:Waste -1]
                                      [BoundReceptor :.:Bound_Receptor 1];
        }

        Process GillespieProcess( BoundReceptorInternalization )
        {
                StepperID DEvent_S1;

                k .01;  
                VariableReferenceList [BoundReceptor :.:Bound_Receptor -1];
        }

        Process MichaelisUniUniFluxProcess( CToDForward )
        {
                VariableReferenceList [ S0 :.:C -1]
                                      [ P0 :.:D 1]
                                      [ C0 :.:Bound_Receptor 0];

                KmS   1;
                KcF   10;
        }
 

        Process DivisionProcess( CellDivision )
        {
                StepperID DTime_S1;

                InitializeMethod "CellDivisionThreshold = 1e-8";

                Expression "volume.Value > 0.0";

                VariableReferenceList [volume :.:SIZE 1]
                                      [V1 :.:Cell_Growth_Factor 1]
                                      [V2 :.:Cell_Waste 1]
                                      [V3 :.:C 1]
                                      [V4 :.:D 1]
                                      [V5 :.:Receptor 1]
                                      [V6 :.:Bound_Receptor 1];
        }

        Process ApoptosisProcess( ApoptosisProcess )
        {

                StepperID DTime_S1;
                
                InitializeMethod "DeathThreshold = 5000";

                #                Expression "V4.Value > DeathThreshold";
                Expression "0 == 0"

                VariableReferenceList [V0 :.:SIZE 1]
                                      [V1 :.:Cell_Growth_Factor 1]
                                      [V2 :.:Cell_Waste 1]
                                      [V3 :.:C 1]
                                      [V4 :.:D 1]
                                      [V5 :.:Receptor 1]
                                      [V6 :.:Bound_Receptor 1];
        }

}