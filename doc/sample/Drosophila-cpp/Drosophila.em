
# created by eml2em program
# from file: droTest.eml, date: Sun Oct 13 15:04:54 2002
#

Stepper FixedODE1Stepper( DE )
#Stepper ODE23Stepper( DE )
#Stepper ODE45Stepper( DE )
{
	# no property
}

System System( / )
{
	StepperID	DE;
	Variable Variable( SIZE ) { Value 0.000000000000001; }
}

System System( / )
{
	StepperID	DE;

}

System System( /CELL )
{
	StepperID	DE;

}

System System( /CELL/CYTOPLASM )
{
	StepperID	DE;

	Variable Variable( SIZE ) { Value 1e-18; }

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
	
	Process FM1Process( R_toy1 )
	{
		VariableReferenceList	[ P0 Variable:.:M 1 ] [ C0 Variable:.:Pn 0 ];
		vs	0.76;
		KI	1;
	}
	
	Process FM2Process( R_toy2 )
	{
		VariableReferenceList	[ P0 Variable:.:M 1 ];
		vm	0.65;
		Km	0.5;
	}
	
	Process FP01Process( R_toy3 )
	{
		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:M 0 ];
		Km	0.38;
	}
	
	Process FP02Process( R_toy4 )
	{
		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:P0 0 ];
		V1	3.2;
		K1	2;
	}
	
	Process FP03Process( R_toy5 )
	{
		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:P1 0 ];
		V2	1.58;
		K2	2;
	}
	
	Process FP11Process( R_toy6 )
	{
		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P0 0 ];
		V1	3.2;
		K1	2;
	}
	
	Process FP12Process( R_toy7 )
	{
		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P1 0 ];
		V2	1.58;
		K2	2;
	}
	
	Process FP13Process( R_toy8 )
	{
		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P1 0 ];
		V3	5;
		K3	2;
	}
	
	Process FP14Process( R_toy9 )
	{
		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P2 0 ];
		V4	2.5;
		K4	2;
	}
	
	Process FP21Process( R_toy10 )
	{
		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P1 0 ];
		V3	5;
		K3	2;
	}
	
	Process FP22Process( R_toy11 )
	{
		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P2 0 ];
		V4	2.5;
		K4	2;
	}
	
	Process FP23Process( R_toy12 )
	{
		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P2 0 ];
		k1	1.9;
	}
	
	Process FP24Process( R_toy13 )
	{
		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:Pn 0 ];
		k2	1.3;
	}
	
	Process FP25Process( R_toy14 )
	{
		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P2 0 ];
		vd	0.95;
		Kd	0.2;
	}
	
	Process FPn1Process( R_toy15 )
	{
		VariableReferenceList	[ P0 Variable:.:Pn 1 ] [ C0 Variable:.:P2 0 ];
		k1	1.9;
	}
	
	Process FPn2Process( R_toy16 )
	{
		VariableReferenceList	[ P0 Variable:.:Pn 1 ] [ C0 Variable:.:Pn 0 ];
		k2	1.3;
	}
	
	
}

