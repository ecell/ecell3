Stepper FluxDistributionStepper( FD )
{

}

Stepper FixedODE1Stepper( DE1 )
{

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

#Static :

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
