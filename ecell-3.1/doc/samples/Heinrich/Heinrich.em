
@{
Voli = 1e-13
N_A = 6.02214e+23
}

Stepper ODE45Stepper( DES01 )
# Stepper ODEStepper( DES01 )
{
	# ; no properties
}

System System( / )
{
	StepperID	DES01;
}

System System( /CELL )
{
	StepperID	DES01;
}

System System( /CELL/CYTOPLASM )
{
	StepperID	DES01;

	Variable Variable( SIZE )
	{
		Value	@( Voli );
	}

	Variable Variable( A13P2G )
	{
		Value	@( Voli * N_A * 0.0005082 );
	}

	Variable Variable( A23P2G )
	{
		Value	@( Voli * N_A * 5.0834 );
	}

	Variable Variable( PEP )
	{
		Value	@( Voli * N_A * 0.020502 );
	}

	Variable Variable( AMP )
	{
		Value	@( Voli * N_A * 0.080139 );
	}

	Variable Variable( ADP )
	{
		Value	@( Voli * N_A * 0.2190 );
	}

	Variable Variable( ATP )
	{
		Value	@( Voli * N_A * 1.196867 );
	}

	Process ExpressionFluxProcess( HK_PFK )
	{
		k	3.20;
		Km	1.0;
		nH	4.0;

		Expression	"k * S0.MolarConc / ( 1.0 + pow( S0.MolarConc / Km, nH ) )  * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:ATP -2 ]
					[ P0 :.:A13P2G +2 ]
					[ P1 :.:ADP +2 ];
	}

	Process ExpressionFluxProcess( P2GM )
	{
		k	1500;

		Expression "k * S0.MolarConc * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:A13P2G -1 ]
					[ P0 :.:A23P2G +1 ];
	}

	Process ExpressionFluxProcess( P2Gase )
	{
		k	0.15;

		Expression "k * S0.MolarConc * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:A23P2G -1 ]
					[ P0 :.:PEP +1 ];
	}

	Process ExpressionFluxProcess( PGK )
	{
		k	1.57e+4;

		Expression "k * S0.MolarConc * S1.MolarConc * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:A13P2G -1 ]
					[ S1 :.:ADP -1 ]
					[ P0 :.:PEP +1 ]
					[ P1 :.:ATP +1 ];
	}

	Process ExpressionFluxProcess( PK )
	{
		isReversible	1;

		k	559;

		Expression "k * S0.MolarConc * S1.MolarConc * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:PEP -1 ]
					[ S1 :.:ADP -1 ]
					[ P0 :.:ATP +1 ];
	}

	Process ExpressionFluxProcess( AK )
	{
		isReversible	1;

		q	2.0;
		A	1.0;

		Expression "A * ( S0.MolarConc * S1.MolarConc - q * P0.MolarConc * P0.MolarConc ) * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:AMP -1 ]
					[ S1 :.:ATP -1 ]
					[ P0 :.:ADP +2 ];
	}

	Process ExpressionFluxProcess( ATPase )
	{
		k	1.46;

		Expression "k * S0.MolarConc * self.getSuperSystem().SizeN_A";

		VariableReferenceList	[ S0 :.:ATP -1 ]
					[ P0 :.:ADP +1 ];
	}
}
