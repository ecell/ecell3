#
# A very simple model with one michaelis-uni-uni reaction.
#

#Stepper ODE23Stepper( DE1 )
Stepper FixedODE1Stepper( DE1 )
{
	# no property
}

System System( / )
{
	StepperID	DE1;

	Variable Variable( SIZE )
	{
		Value	1e-18;
	}

	Variable Variable( S )
	{
		Value	20000;
	}
	
	Variable Variable( P )
	{
		Value	2000;
	}
	
	Variable Variable( E )
	{
		Value	10000;
	}
	
	Process MichaelisUniUniFluxProcess( E )
	{
		VariableReferenceList	[ S0 :.:S -1 ]
 					[ P0 :.:P 1 ]
					[ C0 :.:E 0 ];
		KmS	1;
		KcF	10;
	}

	
}

