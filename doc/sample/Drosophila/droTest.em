
# created by eml2em program
# from file: droTest.eml, date: Sun Oct 13 15:04:54 2002
#

#Stepper Euler1Stepper( SRM_01 )
Stepper Midpoint2Stepper( SRM_01 )
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

	Variable Variable( M )
	{
		Value	3.61328202E-01;
	}
	
	Variable Variable( Pn )
	{
		Value	6.21367E-01;
	}
	
	Variable Variable( P0 )
	{
		Value	3.01106835E-01;
	}
	
	Variable Variable( P1 )
	{
		Value	3.01106835E-01;
	}
	
	Variable Variable( P2 )
	{
		Value	3.61328202E-01;
	}
	
	Process FM1Process( R.toy1 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:M 1 ] [ C0 Variable:/CELL/CYTOPLASM:Pn 0 ];
		vs	0.76;
		KI	1;
	}
	
	Process FM2Process( R.toy2 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:M 1 ];
		vm	0.65;
		Km	0.5;
	}
	
	Process FP01Process( R.toy3 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P0 1 ] [ C0 Variable:/CELL/CYTOPLASM:M 0 ];
		Km	0.38;
	}
	
	Process FP02Process( R.toy4 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P0 1 ] [ C0 Variable:/CELL/CYTOPLASM:P0 0 ];
		V1	3.2;
		K1	2;
	}
	
	Process FP03Process( R.toy5 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P0 1 ] [ C0 Variable:/CELL/CYTOPLASM:P1 0 ];
		V2	1.58;
		K2	2;
	}
	
	Process FP11Process( R.toy6 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P1 1 ] [ C0 Variable:/CELL/CYTOPLASM:P0 0 ];
		V1	3.2;
		K1	2;
	}
	
	Process FP12Process( R.toy7 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P1 1 ] [ C0 Variable:/CELL/CYTOPLASM:P1 0 ];
		V2	1.58;
		K2	2;
	}
	
	Process FP13Process( R.toy8 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P1 1 ] [ C0 Variable:/CELL/CYTOPLASM:P1 0 ];
		V3	5;
		K3	2;
	}
	
	Process FP14Process( R.toy9 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P1 1 ] [ C0 Variable:/CELL/CYTOPLASM:P2 0 ];
		V4	2.5;
		K4	2;
	}
	
	Process FP21Process( R.toy10 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P2 1 ] [ C0 Variable:/CELL/CYTOPLASM:P1 0 ];
		V3	5;
		K3	2;
	}
	
	Process FP22Process( R.toy11 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P2 1 ] [ C0 Variable:/CELL/CYTOPLASM:P2 0 ];
		V4	2.5;
		K4	2;
	}
	
	Process FP23Process( R.toy12 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P2 1 ] [ C0 Variable:/CELL/CYTOPLASM:P2 0 ];
		k1	1.9;
	}
	
	Process FP24Process( R.toy13 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P2 1 ] [ C0 Variable:/CELL/CYTOPLASM:Pn 0 ];
		k2	1.3;
	}
	
	Process FP25Process( R.toy14 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:P2 1 ] [ C0 Variable:/CELL/CYTOPLASM:P2 0 ];
		vd	0.95;
		Kd	0.2;
	}
	
	Process FPn1Process( R.toy15 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:Pn 1 ] [ C0 Variable:/CELL/CYTOPLASM:P2 0 ];
		k1	1.9;
	}
	
	Process FPn2Process( R.toy16 )
	{
		VariableReferenceList	[ P0 Variable:/CELL/CYTOPLASM:Pn 1 ] [ C0 Variable:/CELL/CYTOPLASM:Pn 0 ];
		k2	1.3;
	}
	
	
}

