
# created by eml2em program
# from file: simple.eml, date: Tue Sep 24 17:20:04 2002
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

System System( /CELL )
{
	StepperID	SRM_01;
	Volume	unknown;
}

System System( /CELL/CYTOPLASM )
{
	StepperID	SRM_01;
	Volume	1e-18;

	Substance Substance( S )
	{
		Quantity	1000000;
	}
	
	Substance Substance( P )
	{
		Quantity	0;
	}
	
	Substance Substance( E )
	{
		Quantity	1000;
	}
	
	Reactor MichaelisUniUniReactor( E )
	{
		ReactantList	[ S0 Substance:/CELL/CYTOPLASM:S -1 ] [ P0 Substance:/CELL/CYTOPLASM:P 1 ] [ C0 Substance:/CELL/CYTOPLASM:E 0 ];
		KmS	1;
		KcF	10;
	}
	
	
}

