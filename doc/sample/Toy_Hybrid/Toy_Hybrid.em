#
# * README
#
# When you use Hybrid Dynamic/Static Simulation Algorithm in your model,
# you need FluxDistributionStepper, FluxDistributionProcess, 
# QuasiDynamicFluxProcess correctly.  
#
# 
# * FluxDistributionStepper
#
# When you use FluxDistributionStepper, 
# FluxDistributionStepper does not need to set Properties.
# 
# When Influx Process or efflux Process in Dynamic part write 
# Variables in Static part, FluxDistributionStepper 
# call Process in Static part.  
# FluxDistributionStepper do not have StepInterval, 
# call step() at interupt by Other Stepper, do not call step() by self.
# 
#
# * FluxDistributionProcess
#
# When you use FluxDistributionProcess, 
# FluxDIstributionProcess need to set follwing Properties.  
#
# -StepperID
# set FluxDistributuinStepper's ID.
#
# -UnknownProcessList
# set a list of Process FullID that QuasiDynamicFluxProcess in Static Part. 
#
# -KnownProcessList
# set a list of Process FullID that Influx Process 
# or efflux Process in Dynamic part. 
#
#
# * QuasiDynamicFluxProcess
#
# When you use QuasiDynamicFluxProcess, 
# QuasiDynamicFluxProcess need to set follwing Properties.  
#
# -StepperID
# set FluxDistributuinStepper's ID.
#
# -VariableReferenceList
# set VariableReferenceList that show Flux Information in Static part.
#
# -Irreversible
# set 1, if this Process is Ireeversible. default 0.
#

Stepper FluxDistributionStepper( FD )
{
	# nothing
}

Stepper FixedODE1Stepper( DE1 )
{
	#StepInterval 	0.001;
}

System System( / )
{
	StepperID	DE1;
	Name	"The culture medium";
}

System System( /CELL )
{
	StepperID	DE1;
	Name	"The cell";
}

System System( /CELL/CYTOPLASM )
{
	StepperID	DE1;
	Name	"The cytoplasm";

	Variable Variable( SIZE )
	{
		Value	1E-018;
	}

	Variable Variable( A )
	{
		Value	10000;
		Fixed	1;
		Name	A;
	}

	Variable Variable( B )
	{
		Value	10000;
		Name	B;
	}

	Variable Variable( C )
	{
		Value	10000;
		Name	C;
	}

	Variable Variable( D )
	{
		Value	5000;
		Name	D;
	}

	Variable Variable( E )
	{
		Value	5000;
		Name	E;
	}

	Variable Variable( F )
	{
		Value	5000;
		Name	F;
	}

	Variable Variable( G )
	{
		Value	5000;
		Name	G;
	}

	Variable Variable( H )
	{
		Value	0;
		Name	H;
	}


	Variable Variable( I )
	{
		Value	0;
		Name	I;
	}

#Dynamic : influx & efflux

        Process MassActionFluxProcess( E_AB )
        {
                VariableReferenceList   [ S0 :.:A -1 ]
                                        [ P0 :.:B 1 ];
                k     0.01;
        }

        Process MassActionFluxProcess( E_BC )
        {
                VariableReferenceList   [ S0 :.:B -1 ]
                                        [ P0 :.:C 1 ];
                k     0.01;
        }

        Process MassActionFluxProcess( E_FH )
        {
                VariableReferenceList   [ S0 :.:F -1 ]
                                        [ P0 :.:H 1 ];
                k     0.01;
        }

        Process MassActionFluxProcess( E_GI )
        {
                VariableReferenceList   [ S0 :.:G -1 ]
                                        [ P0 :.:I 1 ];
                k     0.01;
        }

#Static : Super & Slave 

        Process FluxDistributionProcess( Super )
        {
		StepperID	FD;

                KnownProcessList	
		:/CELL/CYTOPLASM:E_BC 
		:/CELL/CYTOPLASM:E_FH 
		:/CELL/CYTOPLASM:E_GI;

                UnknownProcessList     	
		:/CELL/CYTOPLASM:E_CD 
		:/CELL/CYTOPLASM:E_CE 
		:/CELL/CYTOPLASM:E_DF
		:/CELL/CYTOPLASM:E_EG;

        }

        Process QuasiDynamicFluxProcess( E_CD )
        {
		StepperID	FD;
                VariableReferenceList   [ S0 :.:C -1 ]
                                        [ P0 :.:D 1 ];
	}


        Process QuasiDynamicFluxProcess( E_CE )
        {
		StepperID	FD;
                VariableReferenceList   [ S0 :.:C -1 ]
                                        [ P0 :.:E 1 ];
	}

        Process QuasiDynamicFluxProcess( E_DF )
        {
		StepperID	FD;
                VariableReferenceList   [ S0 :.:D -1 ]
                                        [ P0 :.:F 1 ];
	}

        Process QuasiDynamicFluxProcess( E_EG )
        {
		StepperID	FD;
                VariableReferenceList   [ S0 :.:E -1 ]
                                        [ P0 :.:G 1 ];
	}

}
