 
Stepper DiscreteTimeStepper( DT_01 )
{
	StepInterval	0.001;

}

System System( / )
{
	StepperID	DT_01;
	Name	"The culture medium";
}

System System( /CELL )
{
	StepperID	DT_01;
	Name	"The cell";
}

System System( /CELL/CYTOPLASM )
{
	StepperID	DT_01;
	Name	"The cytoplasm";

	Variable Variable( SIZE )
	{
		Value	1E-018;
	}

	Variable Variable( A )
	{
		Value	10000;
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
		Value	10000;
		Name	D;
	}

	Variable Variable( E )
	{
		Value	10000;
		Name	E;
	}

	Process SSystemProcess( SSystem )
	{
		Name	"SSystemPProcess";
		Order	3;
		SSystemMatrix	
		[ 0.5 0   0   0  0   0  0.6 0 0.2 0   0   0   ]
		[ 0.7 0.1 0   0  0   0  0.2 0 0   0.2 0.2 0   ]
		[ 0.5 0   0.1 0  0.1 0  0.4 0 0   0   0   0.2 ]
		[ 0.2 0   0.1 0  0   0  0.9 0 0   0.2 0   0   ]
		[ 0.1 0   0.7 0  0   0  0.5 0 0   0   0   0   ];
		VariableReferenceList	
		[ P0 Variable:/CELL/CYTOPLASM:A  1 ]
		[ P1 Variable:/CELL/CYTOPLASM:B  1 ]
		[ P2 Variable:/CELL/CYTOPLASM:C  1 ]
		[ P3 Variable:/CELL/CYTOPLASM:D  1 ]
		[ P4 Variable:/CELL/CYTOPLASM:E  1 ];
	}
}
