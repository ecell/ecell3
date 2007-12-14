#
# Drosophila.em
#

@{
VOLUME = 1e-18
N_A = 6.02e+23
}

Stepper ODEStepper( DE )
{
	# no property
}

System System( / )
{
	StepperID	DE;
	Variable Variable( SIZE ) { Value 0.000000000000001; }
}

System System( /CELL )
{
	StepperID	DE;
}	
	
System System( /CELL/CYTOPLASM )
{
	StepperID	DE;
	
	Variable Variable( SIZE ) 
        {	
                Value   @(VOLUME);
        }

	Variable Variable( M )
	{
		Value	@(3.61328202E-01 * N_A * VOLUME);
	}
	
	Variable Variable( Pn )
	{
		Value	@(6.21367E-01 * N_A * VOLUME);
	}
	
	Variable Variable( P0 )
	{
		Value	@(3.01106835E-01 * N_A * VOLUME);
	}
	
	Variable Variable( P1 )
	{
		Value	@(3.01106835E-01 * N_A * VOLUME);
	}
	
	Variable Variable( P2 )
	{
		Value	@(3.61328202E-01 * N_A * VOLUME);
	}
	
	Process PythonProcess( R_toy1 )
	{
		IsContinuous	1;
		InitializeMethod "vs = 0.76; KI = 1;";
		FireMethod "self.setFlux( (vs*KI) / (KI + (C0.MolarConc * C0.MolarConc * C0.MolarConc) ) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:M 1 ] [ C0 Variable:.:Pn ];
	}
	
	Process PythonProcess( R_toy2 )
	{
		IsContinuous	1;
		InitializeMethod "vm = 0.65; Km = 0.5;";
		FireMethod "self.setFlux( (vm * S0.MolarConc) / ( Km + S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ S0 Variable:.:M -1 ];
	}
	
	Process PythonProcess( R_toy3 )
	{
		IsContinuous	1;
		InitializeMethod "Ks = 0.38;";
		FireMethod "self.setFlux( (Ks * C0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ C0 Variable:.:M 0 ];
	}
	
	Process PythonProcess( R_toy4 )
	{
		IsContinuous	1;
		InitializeMethod "V1 = 3.2; K1 = 2;";
		FireMethod "self.setFlux( ( V1 * S0.MolarConc) / (K1 + S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ S0 Variable:.:P0 -1 ];
	}
	
	Process PythonProcess( R_toy5 )
	{
		IsContinuous	1;
		InitializeMethod "V2 = 1.58; K2 = 2;";
		FireMethod "self.setFlux( (V2 * S0.MolarConc) / (K2 + S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P0 1 ] [ S0 Variable:.:P1 -1 ];
	}
	
	Process PythonProcess( R_toy6 )
	{
		IsContinuous	1;
		InitializeMethod "V3 = 5; K3 = 2;";
		FireMethod "self.setFlux( ( V3 * S0.MolarConc) / (K3 + S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ S0 Variable:.:P1 -1];
	}
	
	Process PythonProcess( R_toy7 )
	{
		IsContinuous	1;
		InitializeMethod "V4 = 2.5; K4 = 2;";
		FireMethod "self.setFlux( (V4 * S0.MolarConc) / (K4 + S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P1 1 ] [ S0 Variable:.:P2 -1 ];
	}
			
	Process PythonProcess( R_toy8 )
	{
		IsContinuous	1;
		InitializeMethod "k1 = 1.9;";
		FireMethod "self.setFlux( ( k1 * S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:Pn 1 ] [ S0 Variable:.:P2 -1 ];
	}
	
	Process PythonProcess( R_toy9 )
	{
		IsContinuous	1;
		InitializeMethod "k2 = 1.3;";
		FireMethod "self.setFlux( (k2 * S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ P0 Variable:.:P2 1 ] [ S0 Variable:.:Pn -1 ];
	}
	
	Process PythonProcess( R_toy10 )
	{
		IsContinuous	1;
		InitializeMethod "vd = 0.95; Kd = 0.2;";
		FireMethod "self.setFlux( ( vd * S0.MolarConc) / (Kd + S0.MolarConc) * self.getSuperSystem().SizeN_A )";

		VariableReferenceList	[ S0 Variable:.:P2 -1 ];
	}	
	
}

