
# created by eml2em program
# from file: simple.eml, date: Sun Oct 13 05:59:45 2002
#

Stepper Euler1Stepper( SRM_01 )
{
	# no property
}

System System( / )
{
	StepperID	SRM_01;
	Volume	0.000000000000001;
}

System System( / )
{
	StepperID	SRM_01;
	Volume	0.000000000000001;
}

System System( /CELL )
{
	StepperID	SRM_01;
	Volume	unknown;
}

System System( /CELL/CYTOPLASM )
{
	StepperID	SRM_01;
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
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:S -1 ] [ P0 Variable:/CELL/CYTOPLASM:P 1 ] [ C0 Variable:/CELL/CYTOPLASM:E 0 ];
		KmS	1;
		KcF	10;
	}
	
	
}

