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
	
	Process ExpressionFluxProcess( R_toy1 )
	{
		vs	0.76;
		KI	1;

		Expression "( (vs*KI) / (KI + (C0.MolarConc * C0.MolarConc * C0.MolarConc) ) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:M 1 ] [ C0 Variable:.:Pn 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy2 )
	{
		vm	0.65;
		Km	0.5;

		Expression "( (-1 * vm * P0.MolarConc) / ( Km + P0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:M 1 ];
	}
	
	Process ExpressionFluxProcess( R_toy3 )
	{
		Km	0.38;

		Expression "( (Km * C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:M 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy4 )
	{
		V1	3.2;
		K1	2;

		Expression "( (-1 * V1 * C0.MolarConc) / (K1 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:P0 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy5 )
	{
		V2	1.58;
		K2	2;

		Expression "( (V2 * C0.MolarConc) / (K2 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:P1 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy6 )
	{
		V1	3.2;
		K1	2;

		Expression "( (V1 * C0.MolarConc) / (K1 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P0 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy7 )
	{
		V2	1.58;
		K2	2;

		Expression "( (-1 * V2 * C0.MolarConc) / (K2 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P1 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy8 )
	{
		V3	5;
		K3	2;

		Expression "( (-1 * V3 * C0.MolarConc) / (K3 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P1 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy9 )
	{
		V4	2.5;
		K4	2;

		Expression "( (V4 * C0.MolarConc) / (K4 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ C0 Variable:.:P2 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy10 )
	{
		V3	5;
		K3	2;

		Expression "( (V3 * C0.MolarConc) / (K3 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P1 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy11 )
	{
		V4	2.5;
		K4	2;

		Expression "( (-1 * V4 * C0.MolarConc) / (K4 + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P2 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy12 )
	{
		k1	1.9;

		Expression "( (-1 * k1 * C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P2 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy13 )
	{
		k2	1.3;

		Expression "( (k2 * C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:Pn 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy14 )
	{
		vd	0.95;
		Kd	0.2;

		Expression "( (-1 * vd * C0.MolarConc) / (Kd + C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ C0 Variable:.:P2 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy15 )
	{
		k1	1.9;

		Expression "( (k1 * C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:Pn 1 ] [ C0 Variable:.:P2 0 ];
	}
	
	Process ExpressionFluxProcess( R_toy16 )
	{
		k2	1.3;

		Expression "( (-1 * k2 * C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:Pn 1 ] [ C0 Variable:.:Pn 0 ];
	}
	
	
}

