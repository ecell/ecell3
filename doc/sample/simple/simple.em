#
# A very simple model with one michaelis-uni-uni reaction.
#

#Stepper Fehlberg23Stepper( DE1 )
Stepper FixedEuler1Stepper( DE1 )
{
	# no property
}

System System( / )
{
	StepperID	DE1;
	Volume	1e-18;

	Variable Variable( S )
	{
		Value	1000000;
	}
	
	Variable Variable( P )
	{
		Value	0;
	}
	
	Variable Variable( E )
	{
		Value	1000;
	}
	
	Process MichaelisUniUniProcess( E )
	{
		VariableReferenceList	[ S0 Variable:.:S -1 ]
 					 [ P0 Variable:.:P 1 ]
					 [ C0 Variable:.:E 0 ];
		KmS	1;
		KcF	10;
	}
	
}

