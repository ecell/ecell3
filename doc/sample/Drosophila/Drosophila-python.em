
# created by eml2em program
# from file: droTest.eml, date: Sun Oct 13 15:04:54 2002
#

# Stepper FixedODE1Stepper( DE )
Stepper ODEStepper( DE )
{
	# no property
}

System CompartmentSystem( / )
{
	StepperID	DE;
	Variable Variable( SIZE ) { Value 0.000000000000001; }
}

System CompartmentSystem( / )
{
	StepperID	DE;

}

System CompartmentSystem( /CELL )
{
	StepperID	DE;

}

System CompartmentSystem( /CELL/CYTOPLASM )
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
	
	Process PythonProcess( R_toy1 )
	{
		IsContinuous 1;
		InitializeMethod "vs = 0.76; KI = 1";
		FireMethod "self.setFlux(((vs * KI)/(KI + (C0.MolarConc * C0.MolarConc * C0.MolarConc))) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:M 1 ] [ C0 :.:Pn 0 ];
	}
	
	Process PythonProcess( R_toy2 )
	{
		IsContinuous 1;
		InitializeMethod "vm = 0.65; Km = 0.5";
		FireMethod "self.setFlux(((-1 * vm * P0.MolarConc)/(Km + P0.MolarConc)) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:M 1 ];
	}
	
	Process PythonProcess( R_toy3 )
	{
		IsContinuous 1;
		InitializeMethod "Km = 0.38";
		FireMethod "self.setFlux((Km * C0.MolarConc) * self.getSuperSystem().SizeN_A)";
		VariableReferenceList	[ P0 :.:P0 1 ] [ C0 :.:M 0 ];
	}
	
	Process PythonProcess( R_toy4 )
	{
		IsContinuous 1;
		InitializeMethod "V1 = 3.2; K1 = 2";
		FireMethod "self.setFlux(((-1 * V1 * C0.MolarConc) / (K1 + C0.MolarConc)) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P0 1 ] [ C0 :.:P0 0 ];
	}
	
	Process PythonProcess( R_toy5 )
	{
		IsContinuous 1;
		InitializeMethod "V2 = 1.58; K2 = 2";
		FireMethod "self.setFlux(((V2 * C0.MolarConc) / (K2 + C0.MolarConc)) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P0 1 ] [ C0 :.:P1 0 ];
	}
	
	Process PythonProcess( R_toy6 )
	{
		IsContinuous 1;
		InitializeMethod "V1 = 3.2; K1 = 2";
		FireMethod "self.setFlux(((V1 * C0.MolarConc) / (K1 + C0.MolarConc)) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P1 1 ] [ C0 :.:P0 0 ];
	}
	
	Process PythonProcess( R_toy7 )
	{
		IsContinuous 1;
		InitializeMethod "V2 = 1.58; K2 = 2";
		FireMethod "self.setFlux(((-1 * V2 * C0.MolarConc) / (K2 + C0.MolarConc)) * self.getSuperSystem().SizeN_A)";
		VariableReferenceList	[ P0 :.:P1 1 ] [ C0 :.:P1 0 ];
	}
	
	Process PythonProcess( R_toy8 )
	{
		IsContinuous 1;
		InitializeMethod "V3 = 5; K3 = 2";
		FireMethod "self.setFlux(((-1 * V3 * C0.MolarConc) / (K3 + C0.MolarConc)) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P1 1 ] [ C0 :.:P1 0 ];
	}
	
	Process PythonProcess( R_toy9 )
	{
		IsContinuous 1;
		InitializeMethod "V4 = 2.5; K4 = 2";
		FireMethod "self.setFlux(((V4 * C0.MolarConc) / (K4 + C0.MolarConc)) * self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P1 1 ] [ C0 :.:P2 0 ];
	}
	
	Process PythonProcess( R_toy10 )
	{
		IsContinuous 1;
		InitializeMethod "V3 = 5; K3 = 2";
		FireMethod "self.setFlux(((V3 * C0.MolarConc) / (K3 + C0.MolarConc)) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P2 1 ] [ C0 :.:P1 0 ];

	}
	
	Process PythonProcess( R_toy11 )
	{
		IsContinuous 1;
		InitializeMethod "V4 = 2.5; K4 = 2";
		FireMethod "self.setFlux(((-1 * V4 * C0.MolarConc) / (K4 + C0.MolarConc)) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P2 1 ] [ C0 :.:P2 0 ];
	}
	
	Process PythonProcess( R_toy12 )
	{
		IsContinuous 1;
		InitializeMethod "K1 = 1.9";
		FireMethod "self.setFlux((-1 * K1 * C0.MolarConc) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P2 1 ] [ C0 :.:P2 0 ];
	}
	
	Process PythonProcess( R_toy13 )
	{
		IsContinuous 1;
		InitializeMethod "k2 = 1.3";
		FireMethod "self.setFlux((k2 * C0.MolarConc) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P2 1 ] [ C0 :.:Pn 0 ];
	}
	
	Process PythonProcess( R_toy14 )
	{
		IsContinuous 1;
		InitializeMethod "vd = 0.95; Kd = 0.2";
		FireMethod "self.setFlux(((-1 * vd * C0.MolarConc) / (Kd + C0.MolarConc)) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:P2 1 ] [ C0 :.:P2 0 ];
	}
	
	Process PythonProcess( R_toy15 )
	{
		IsContinuous 1;
		InitializeMethod "k1 = 1.9";
		FireMethod "self.setFlux((k1 * C0.MolarConc) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:Pn 1 ] [ C0 :.:P2 0 ];
	}
	
	Process PythonProcess( R_toy16 )
	{
		IsContinuous 1;
		InitializeMethod "k2 = 1.3";
		FireMethod "self.setFlux((-1 * k2 * C0.MolarConc) *  self.getSuperSystem().SizeN_A)";

		VariableReferenceList	[ P0 :.:Pn 1 ] [ C0 :.:Pn 0 ];
	}
	
	
}

