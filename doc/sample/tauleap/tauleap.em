Stepper TauLeapStepper( TL )
{
	# Epsilon	0.01;
}

System System( / )
{
	StepperID	TL;

	Variable Variable( SIZE )
	{
		Value	1e-15;
	}

	Variable Variable( S1 )
	{
		Value	100000;
	}

	Variable Variable( S2 )
	{
		Value	0;
	}

	Variable Variable( S3 )
	{
		Value	0;
	}
	
	Process GillespieProcess( C1 )
	{
		k	1;
		VariableReferenceList	[ S0 :.:S1 -1 ];
	}

	Process GillespieProcess( C2 )
	{
		k	@(0.002 * 1e-15 * 6.02214 * 1e+23);
		VariableReferenceList	[ S0 :.:S1 -2 ]
					[ P0 :.:S2  1 ];
	}

	Process GillespieProcess( C3 )
	{
		k	0.5;
		VariableReferenceList	[ S0 :.:S2 -1 ]
					[ P0 :.:S1  2 ];
	}

	Process GillespieProcess( C4 )
	{
		k	0.04;
		VariableReferenceList	[ S0 :.:S2 -1 ]
					[ P0 :.:S3  1 ];
	}
	
}

