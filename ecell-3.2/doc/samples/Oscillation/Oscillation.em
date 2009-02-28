Stepper FixedODE1Stepper( DefaultStepper )
{
	MaxStepInterval	INF;
	MinStepInterval	0;
	Priority	0;
	StepInterval	0.001;
}

System System( / )
{
	Name	"The Root System";
	StepperID	DefaultStepper;

	Variable Variable( SIZE )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Value	1;
	}
}

System System( /cell )
{
	Name	"E. coli cell";
	StepperID	DefaultStepper;

	Variable Variable( TetR )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Name	"Tetracycline repressor";
		Value	0.1;
	}
	
	Variable Variable( tetR )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Name	"TetR transcripts";
		Value	0.1;
	}
	
	Variable Variable( cI )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Name	"CI transcripts";
		Value	0.1;
	}
	
	Variable Variable( CI )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Name	"CI protain";
		Value	0.1;
	}
	
	Variable Variable( lacI )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Name	"LacI transcripts";
		Value	0.1;
	}
	
	Variable Variable( SIZE )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Value	1E-15;
	}
	
	Variable Variable( LacI )
	{
		DiffusionCoeff	0;
		Fixed	0;
		Name	"LacI protain";
		Value	0.1;
	}
	
	Process ExpressionFluxProcess( Degrade_TetR )
	{
		Expression	"Kd * S0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Degradation of TetR protain";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ S0 :/cell:TetR -1 1 ];
		Kd	5;
	}
	
	Process ExpressionFluxProcess( Transcript_tetR )
	{
		Expression	"( alpha ) / ( C0.MolarConc ^ n + 1 ) * self.getSuperSystem().SizeN_A";
		Name	"Transcription of TetR gene";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ C0 :/cell:LacI 0 1 ] [ P0 :/cell:tetR 1 1 ];
		alpha	50;
		n	2.1;
	}
	
	Process ExpressionFluxProcess( Translate_TetR )
	{
		Expression	"Ks * C0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Translation of TetR";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ C0 :/cell:tetR 0 1 ] [ P0 :/cell:TetR 1 1 ];
		Ks	5;
	}
	
	Process ExpressionFluxProcess( Degrade_tetR )
	{
		Expression	"Kd * S0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Degradation of tetR transcripts";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ S0 :/cell:tetR -1 1 ];
		Kd	1;
	}
	
	Process ExpressionFluxProcess( Degrade_cI )
	{
		Expression	"Kd * S0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Degradation of CI transcripts";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ S0 :/cell:cI -1 1 ];
		Kd	1;
	}
	
	Process ExpressionFluxProcess( Translate_CI )
	{
		Expression	"Ks * C0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Translation of CI";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ C0 :/cell:cI 0 1 ] [ P0 :/cell:CI 1 1 ];
		Ks	5;
	}
	
	Process ExpressionFluxProcess( Degrade_LacI )
	{
		Expression	"Kd * S0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Degradation of LacI protain";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ S0 :/cell:LacI -1 1 ];
		Kd	5;
	}
	
	Process ExpressionFluxProcess( Degrade_lacI )
	{
		Expression	"Kd * S0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Degradation of LacI transcripts";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ S0 :/cell:lacI -1 1 ];
		Kd	1;
	}
	
	Process ExpressionFluxProcess( Translate_Lacl )
	{
		Expression	"Ks* C0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Translation of LacI";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ C0 :/cell:lacI 0 1 ] [ P0 :/cell:LacI 1 1 ];
		Ks	5;
	}
	
	Process ExpressionFluxProcess( Degrade_CI )
	{
		Expression	"Kd * S0.MolarConc  * self.getSuperSystem().SizeN_A";
		Name	"Degradation of CI protain";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ S0 :/cell:CI -1 1 ];
		Kd	5;
	}
	
	Process ExpressionFluxProcess( Transcript_cI )
	{
		Expression	"( alpha ) / ( C0.MolarConc ^ n + 1 ) * self.getSuperSystem().SizeN_A";
		Name	"Transcription of CI";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ C0 :/cell:TetR 0 1 ] [ P0 :/cell:cI 1 1 ];
		alpha	49;
		n	2.1;
	}
	
	Process ExpressionFluxProcess( Transcript_lacI )
	{
		Expression	"( alpha ) / ( C0.MolarConc ^ n + 1 ) * self.getSuperSystem().SizeN_A";
		Name	"Transctiption of LacI";
		Priority	0;
		StepperID	DefaultStepper;
		VariableReferenceList	[ C0 :/cell:CI 0 1 ] [ P0 :/cell:lacI 1 1 ];
		alpha	50;
		n	2.1;
	}
}
