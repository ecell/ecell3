
# created by eml2em program
# from file: LTD.eml, date: Mon Oct 14 19:51:27 2002
#

Stepper Fehlberg23Stepper( DE_1 )
#Stepper DormandPrince547MStepper( DE_1 )
#Stepper FixedRungeKutta4Stepper( DE_1 )
{
	# no property
}

Stepper DiscreteTimeStepper( DT_1 )
{
	StepInterval 0.001;
}


Stepper SlaveStepper( SLAVE_1 )
{
	# no property
}


System System( / )
{
	StepperID	DE_1;
	Name	"The culture medium";
}

System System( /CELL )
{
	StepperID	DE_1;
	Name	"The cell";
}

System System( /CELL/MEMBRANE )
{
	StepperID	DE_1;
	Volume	0.000000000001;
	Name	"The menbrane";
}

System System( /CELL/CYTOPLASM )
{
	StepperID	DE_1;
	Name	"The cytoplasm";
	Volume	0.000000000001;

	Variable Variable( Ca_DAG_PKC )
	{
		Value	0.0;
		Name	Ca_DAG_PKC;
	}
	
	Variable Variable( DAG_AA_PKC )
	{
		Value	0.0;
		Name	DAG_AA_PKC;
	}
	
	Variable Variable( AA_PKC )
	{
		Value	0.0;
		Name	AA_PKC;
	}
	
	Variable Variable( Ca_AA_PKC )
	{
		Value	0.0;
		Name	Ca_AA_PKC;
	}
	
	Variable Variable( AMPAR )
	{
		Value	301107.0;
		Name	AMPAR;
	}
	
	Variable Variable( AMPAR_P )
	{
		Value	0.0;
		Name	AMPAR_P;
	}
	
	Variable Variable( PP2A )
	{
		Value	1625977.0;
		Name	PP2A;
	}
	
	Variable Variable( Raf )
	{
		Value	301107.0;
		Name	Raf;
	}
	
	Variable Variable( Raf_P )
	{
		Value	0.0;
		Name	Raf_P;
	}
	
	Variable Variable( CRHR )
	{
		Value	301107.0;
		Name	CRHR;
	}
	
	Variable Variable( CRF )
	{
		Value	60221.0;
		Name	CRF;
	}
	
	Variable Variable( N )
	{
		Value	0.0;
		Name	N;
	}
	
	Variable Variable( CRHR_CRF )
	{
		Value	0.0;
		Name	CRHR_CRF;
	}
	
	Variable Variable( Lyn_activate )
	{
		Value	66244.0;
		Name	Lyn_activate;
	}
	
	Variable Variable( AA )
	{
		Value	0.0;
		Name	AA;
	}
	
	Variable Variable( APC )
	{
		Value	18066410.0;
		Name	APC;
	}
	
	Variable Variable( GC )
	{
		Value	1806641.0;
		Name	GC;
	}
	
	Variable Variable( NO )
	{
		Value	0.0;
		Name	NO;
	}
	
	Variable Variable( NO_GC )
	{
		Value	0.0;
		Name	NO_GC;
	}
	
	Variable Variable( GMP5 )
	{
		Value	0.0;
		Name	GMP5;
	}
	
	Variable Variable( GTP )
	{
		Value	6022137.0;
		Name	GTP;
	}
	
	Variable Variable( CGMP )
	{
		Value	0.0;
		Name	CGMP;
	}
	
	Variable Variable( PDE )
	{
		Value	3011068.0;
		Name	PDE;
	}
	
	Variable Variable( PKG )
	{
		Value	1505534.0;
		Name	PKG;
	}
	
	Variable Variable( cGMP_PKG )
	{
		Value	0.0;
		Name	cGMP_PKG;
	}
	
	Variable Variable( G_Sub_P )
	{
		Value	0.0;
		Name	G_Sub_P;
	}
	
	Variable Variable( G_Sub )
	{
		Value	6443686.0;
		Name	G_Sub;
	}
	
	Variable Variable( PP2A_G_Sub_P )
	{
		Value	0.0;
		Name	PP2A_G_Sub_P;
	}

	Variable Variable( MEK )
	{
		Value	301107.0;
		Name	MEK;
	}
	
	Variable Variable( MEK_P )
	{
		Value	0.0;
		Name	MEK_P;
	}
	
	Variable Variable( MEK_PP )
	{
		Value	0.0;
		Name	MEK_PP;
	}
	
	Variable Variable( MAPK_P )
	{
		Value	0.0;
		Name	MAPK_P;
	}
	
	Variable Variable( MAPK_PP )
	{
		Value	0.0;
		Name	MAPK_PP;
	}
	
	Variable Variable( MAPK )
	{
		Value	602214.0;
		Name	MAPK;
	}
	
	Variable Variable( MKP1 )
	{
		Value	1927.0;
		Name	MKP1;
	}
	
	Variable Variable( PLA2 )
	{
		Value	240885.0;
		Name	PLA2;
	}
	
	Variable Variable( Ca )
	{
		Value	0.0;
		Name	Ca;
	}
	
	Variable Variable( Ca_PLA2 )
	{
		Value	0.0;
		Name	Ca_PLA2;
	}
	
	Variable Variable( DAG )
	{
		Value	0.0;
		Name	DAG;
	}
	
	Variable Variable( DAG_Ca_PLA2 )
	{
		Value	0.0;
		Name	DAG_Ca_PLA2;
	}
	
	Variable Variable( PIP2 )
	{
		Value	6022137.0;
		Name	PIP2 ;
	}
	
	Variable Variable( PIP2_PLA2 )
	{
		Value	0.0;
		Name	PIP2_PLA2;
	}
	
	Variable Variable( Ca_PIP2_PLA2 )
	{
		Value	0.0;
		Name	Ca_PIP2_PLA2;
	}
	
	Variable Variable( PLA2_P )
	{
		Value	0.0;
		Name	PLA2_P;
	}
	
	Variable Variable( PKC )
	{
		Value	602214.0;
		Name	PKC;
	}
	
	Variable Variable( Ca_PKC )
	{
		Value	0.0;
		Name	Ca_PKC;
	}
	
	Variable Variable( DAG_PKC )
	{
		Value	0.0;
		Name	DAG_PKC;
	}
	
	Variable Variable( mGluR_Gq )
	{
		Value	180664.0;
		Name	mGluR_Gq;
	}
	
	Variable Variable( Glu )
	{
		Value	0.0;
		Name	Glu;
	}
	
	Variable Variable( Glu_mGluR_Gq )
	{
		Value	0.0;
		Name	Glu_mGluR_Gq;
	}
	
	Variable Variable( GTP_Ga )
	{
		Value	0.0;
		Name	GTP_Ga;
	}
	
	Variable Variable( GDP_Ga )
	{
		Value	0.0;
		Name	GDP_Ga;
	}
	
	Variable Variable( Gbc )
	{
		Value	0.0;
		Name	Gbc;
	}
	
	Variable Variable( Glu_mGluR )
	{
		Value	0.0;
		Name	Glu_mGluR;
	}
	
	Variable Variable( GTP_Ga_PLC )
	{
		Value	0.0;
		Name	GTP_Ga_PLC;
	}
	
	Variable Variable( PLC )
	{
		Value	481771.0;
		Name	PLC;
	}
	
	Variable Variable( IP3 )
	{
		Value	0.0;
		Name	IP3;
	}
	
	Variable Variable( Gabc )
	{
		Value	0.0;
		Name	Gabc;
	}
	
	Variable Variable( MgluR )
	{
		Value	0.0;
		Name	MgluR;
	}
	
	Variable Variable( PKCactive )
	{
		Value	0.0;
		Name	PKCactive;
	}
	
	Variable Variable( MAPKactive )
	{
		Value	602214.0;
		Name	MAPKactive;
	}
	
	Process MassActionProcess( Ca_DAG_PKC_N )
	{
		Name	Ca_DAG_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_DAG_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( DAG_AA_PKC_N )
	{
		Name	DAG_AA_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:DAG_AA_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( AA_PKC_N )
	{
		Name	AA_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AA_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( Ca_AA_PKC_N )
	{
		Name	Ca_AA_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_AA_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P1 )
	{
		Name	AMPAR_AMPAR_P1;
		Km	0.0000035;
		Kcat	1.5;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AMPAR -1 ] [ P0 Variable:/CELL/CYTOPLASM:AMPAR_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:Ca_DAG_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P2 )
	{
		Name	AMPAR_AMPAR_P2;
		Km	0.0000035;
		Kcat	1.5;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AMPAR -1 ] [ P0 Variable:/CELL/CYTOPLASM:AMPAR_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:DAG_AA_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P3 )
	{
		Name	AMPAR_AMPAR_P3;
		Km	0.0000035;
		Kcat	1.5;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AMPAR -1 ] [ P0 Variable:/CELL/CYTOPLASM:AMPAR_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:AA_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P4 )
	{
		Name	AMPAR_AMPAR_P4;
		Km	0.0000035;
		Kcat	1.5;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AMPAR -1 ] [ P0 Variable:/CELL/CYTOPLASM:AMPAR_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:Ca_AA_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_P_AMPAR )
	{
		Name	AMPAR_P_AMPAR;
		Km	0.00001565;
		Kcat	6;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AMPAR_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:AMPAR 1 ] [ C0 Variable:/CELL/CYTOPLASM:PP2A];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P1 )
	{
		Name	Raf_Raf_P1;
		Km	0.0000116;
		Kcat	0.0335;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:Ca_DAG_PKC];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P2 )
	{
		Name	Raf_Raf_P2;
		Km	0.0000116;
		Kcat	0.0335;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:DAG_AA_PKC];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P3 )
	{
		Name	Raf_Raf_P3;
		Km	0.0000116;
		Kcat	0.0335;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:AA_PKC];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P4 )
	{
		Name	Raf_Raf_P4;
		Km	0.0000116;
		Kcat	0.0335;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:Ca_AA_PKC];
	}
	
	Process MassActionProcess( CRHR_CRHR_CRF )
	{
		Name	CRHR_CRHR_CRF;
		k	100000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:CRHR -1 ] [ S1 Variable:/CELL/CYTOPLASM:CRF -1 ] [ P0 Variable:/CELL/CYTOPLASM:CRHR_CRF 1 ];
	}
	
	Process MassActionProcess( CRHR_CRF_CRHR )
	{
		Name	CRHR_CRF_CRHR;
		k	0.00001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:CRHR_CRF -1 ] [ P0 Variable:/CELL/CYTOPLASM:CRHR 1 ] [ P1 Variable:/CELL/CYTOPLASM:CRF 1 ];
	}
	
	Process MassActionProcess( DeCRF )
	{
		Name	DeCRF;
		k	0.02;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:CRF -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P22 )
	{
		Name	Raf_Raf_P22;
		Km	0.00002;
		Kcat	0.0025;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:CRHR_CRF];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P33 )
	{
		Name	Raf_Raf_P33;
		Km	0.00003;
		Kcat	0.08;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:Lyn_activate];
	}
	
	Process MassActionProcess( DeLyn_activate )
	{
		Name	DeLyn_activate;
		k	0.001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Lyn_activate -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( AA_APC )
	{
		Name	AA_APC;
		k	0.001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AA -1 ] [ P0 Variable:/CELL/CYTOPLASM:APC 1 ];
	}
	
	Process MassActionProcess( NO_N )
	{
		Name	NO_N;
		k	0.636;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:NO -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( GC__NO_NO_GC )
	{
		Name	GC__NO_NO_GC;
		k	3000000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GC -1 ] [ S1 Variable:/CELL/CYTOPLASM:NO -1 ] [ P0 Variable:/CELL/CYTOPLASM:NO_GC 1 ];
	}
	
	Process MassActionProcess( NO_GC_GC__NO )
	{
		Name	NO_GC_GC__NO;
		k	0.75;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:NO_GC -1 ] [ P0 Variable:/CELL/CYTOPLASM:GC 1 ] [ P1 Variable:/CELL/CYTOPLASM:NO 1 ];
	}
	
	Process MichaelisUniUniProcess( GTP_CGMP )
	{
		Name	GTP_CGMP;
		Km	0.000045;
		Kcat	7.35;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GTP -1 ] [ P0 Variable:/CELL/CYTOPLASM:CGMP 1 ] [ C0 Variable:/CELL/CYTOPLASM:NO_GC];
	}
	
	Process MichaelisUniUniProcess( CGMP_GMP5 )
	{
		Name	CGMP_GMP5;
		Km	0.000002;
		Kcat	3.87;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:CGMP -1 ] [ P0 Variable:/CELL/CYTOPLASM:GMP5 1 ] [ C0 Variable:/CELL/CYTOPLASM:PDE];
	}
	
	Process MassActionProcess( CGMP__PKG_cGMP_PKG )
	{
		Name	CGMP__PKG_cGMP_PKG;
		k	100000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:CGMP -1 ] [ S1 Variable:/CELL/CYTOPLASM:PKG -1 ] [ P0 Variable:/CELL/CYTOPLASM:cGMP_PKG 1 ];
	}
	
	Process MassActionProcess( cGMP_PKG_CGMP__PKG )
	{
		Name	cGMP_PKG_CGMP__PKG;
		k	0.005;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:cGMP_PKG -1 ] [ P0 Variable:/CELL/CYTOPLASM:CGMP 1 ] [ P1 Variable:/CELL/CYTOPLASM:PKG 1 ];
	}
	
	Process MichaelisUniUniProcess( G_Sub_G_Sub_P )
	{
		Name	G_Sub_G_Sub_P;
		Km	0.0000002;
		Kcat	0.72;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:G_Sub -1 ] [ P0 Variable:/CELL/CYTOPLASM:G_Sub_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:cGMP_PKG];
	}
	
	Process MassActionProcess( G_Sub_P_G_Sub )
	{
		Name	G_Sub_P_G_Sub;
		k	0.0001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:G_Sub_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:G_Sub 1 ];
	}
	
	Process MassActionProcess( G_Sub_P__PP2A_PP2A_G_Sub_P )
	{
		Name	G_Sub_P__PP2A_PP2A_G_Sub_P;
		k	10000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:G_Sub_P -1 ] [ S1 Variable:/CELL/CYTOPLASM:PP2A -1 ] [ P0 Variable:/CELL/CYTOPLASM:PP2A_G_Sub_P 1 ];
	}
	
	Process MassActionProcess( PP2A_G_Sub_P )
	{
		Name	PP2A_G_Sub_P;
		k	0.0027;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PP2A_G_Sub_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:G_Sub_P 1 ] [ P1 Variable:/CELL/CYTOPLASM:PP2A 1 ];
	}
	
	
	Process MichaelisUniUniProcess( Raf_P_Raf )
	{
		Name	Raf_P_Raf;
		Km	0.0000157;
		Kcat	6;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Raf_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:Raf 1 ] [ C0 Variable:/CELL/CYTOPLASM:PP2A];
	}
	
	Process MichaelisUniUniProcess( MEK_MEK_P )
	{
		Name	MEK_MEK_P;
		Km	0.000000398;
		Kcat	0.105;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MEK -1 ] [ P0 Variable:/CELL/CYTOPLASM:MEK_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:Raf_P];
	}
	
	Process MichaelisUniUniProcess( MEK_P_MEK_PP )
	{
		Name	MEK_P_MEK_PP;
		Km	0.000000398;
		Kcat	0.105;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MEK_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:MEK_PP 1 ] [ C0 Variable:/CELL/CYTOPLASM:Raf_P];
	}
	
	Process MichaelisUniUniProcess( MEK_P_MEK )
	{
		Name	MEK_P_MEK;
		Km	0.0000157;
		Kcat	6;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MEK_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:MEK 1 ] [ C0 Variable:/CELL/CYTOPLASM:PP2A];
	}
	
	Process MichaelisUniUniProcess( MEK_PP_MEK_P )
	{
		Name	MEK_PP_MEK_P;
		Km	0.0000157;
		Kcat	6;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MEK_PP -1 ] [ P0 Variable:/CELL/CYTOPLASM:MEK_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:PP2A];
	}
	
	Process MichaelisUniUniProcess( MAPK_MAPK_P )
	{
		Name	MAPK_MAPK_P;
		Km	0.0000000463;
		Kcat	0.15;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MAPK _1 ] [ P0 Variable:/CELL/CYTOPLASM:MAPK_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:MEK_PP];
	}
	
	Process MichaelisUniUniProcess( MAPK_P_MAPK_PP )
	{
		Name	MAPK_P_MAPK_PP;
		Km	0.0000000463;
		Kcat	0.15;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MAPK_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:MAPK_PP 1 ] [ C0 Variable:/CELL/CYTOPLASM:MEK_PP];
	}
	
	Process MichaelisUniUniProcess( MAPK_P_MAPK )
	{
		Name	MAPK_P_MAPK;
		Km	0.000000167;
		Kcat	1;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MAPK_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:MAPK 1 ] [ C0 Variable:/CELL/CYTOPLASM:MKP1];
	}
	
	Process MichaelisUniUniProcess( MAPK_PP_MAPK_P )
	{
		Name	MAPK_PP_MAPK_P;
		Km	0.000000167;
		Kcat	1;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MAPK_PP -1 ] [ P0 Variable:/CELL/CYTOPLASM:MAPK_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:MKP1];
	}
	
	Process MassActionProcess( PLA2_Ca_PLA2 )
	{
		Name	PLA2_Ca_PLA2;
		k	10;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PLA2 -1 ] [ S1 Variable:/CELL/CYTOPLASM:Ca -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_PLA2 1 ];
	}
	
	Process MassActionProcess( Ca_PLA2_PLA2 )
	{
		Name	Ca_PLA2_PLA2;
		k	0.000001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_PLA2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:PLA2 1 ] [ P1 Variable:/CELL/CYTOPLASM:Ca 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA1 )
	{
		Name	APC_AA1;
		Km	0.00002;
		Kcat	54;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:APC -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA 1 ] [ C0 Variable:/CELL/CYTOPLASM:Ca_PLA2];
	}
	
	Process MassActionProcess( Ca_PLA2_DAG_Ca_PLA2 )
	{
		Name	Ca_PLA2_DAG_Ca_PLA2;
		k	100;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_PLA2 -1 ] [ S1 Variable:/CELL/CYTOPLASM:DAG -1 ] [ P0 Variable:/CELL/CYTOPLASM:DAG_Ca_PLA2 1 ];
	}
	
	Process MassActionProcess( DAG_Ca_PLA2_Ca_PLA2 )
	{
		Name	DAG_Ca_PLA2_Ca_PLA2;
		k	0.0002;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:DAG_Ca_PLA2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_PLA2 1 ] [ P1 Variable:/CELL/CYTOPLASM:DAG 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA2 )
	{
		Name	APC_AA2;
		Km	0.00002;
		Kcat	60;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:APC -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA 1 ] [ C0 Variable:/CELL/CYTOPLASM:DAG_Ca_PLA2];
	}
	
	Process MassActionProcess( PLA2_PIP2_PLA2 )
	{
		Name	PLA2_PIP2_PLA2;
		k	1.0;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PLA2 -1 ] [ S1 Variable:/CELL/CYTOPLASM:PIP2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:PIP2_PLA2 1 ];
	}
	
	Process MassActionProcess( PIP2_PLA2_PLA2 )
	{
		Name	PIP2_PLA2_PLA2;
		k	0.0004;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PIP2_PLA2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:PLA2 1 ] [ P1 Variable:/CELL/CYTOPLASM:PIP2 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA3 )
	{
		Name	APC_AA3;
		Km	0.00002;
		Kcat	11.04;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:APC -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA 1 ] [ C0 Variable:/CELL/CYTOPLASM:PIP2_PLA2];
	}
	
	Process MassActionProcess( PIP2_PLA2_Ca_PIP2_PLA2 )
	{
		Name	PIP2_PLA2_Ca_PIP2_PLA2;
		k	12000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PIP2_PLA2 -1 ] [ S1 Variable:/CELL/CYTOPLASM:Ca -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_PIP2_PLA2 1 ];
	}
	
	Process MassActionProcess( Ca_PIP2_PLA2_PIP2_PLA2 )
	{
		Name	Ca_PIP2_PLA2_PIP2_PLA2;
		k	0.012;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_PIP2_PLA2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:PIP2_PLA2 1 ] [ P1 Variable:/CELL/CYTOPLASM:Ca 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA4 )
	{
		Name	APC_AA4;
		Km	0.00002;
		Kcat	36;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:APC -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA 1 ] [ C0 Variable:/CELL/CYTOPLASM:Ca_PIP2_PLA2];
	}
	
	Process MichaelisUniUniProcess( PLA2_PLA2_P )
	{
		Name	PLA2_PLA2_P;
		Km	0.0000256;
		Kcat	20;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PLA2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:PLA2_P 1 ] [ C0 Variable:/CELL/CYTOPLASM:MAPK_PP];
	}
	
	Process MichaelisUniUniProcess( APC_AA5 )
	{
		Name	APC_AA5;
		Km	0.00002;
		Kcat	120;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:APC -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA 1 ] [ C0 Variable:/CELL/CYTOPLASM:PLA2_P];
	}
	
	Process MassActionProcess( PLA2_P_PLA2 )
	{
		Name	PLA2_P_PLA2;
		k	0.17;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PLA2_P -1 ] [ P0 Variable:/CELL/CYTOPLASM:PLA2 1 ];
	}
	
	
	Process MassActionProcess( PKC_Ca_PKC )
	{
		Name	PKC_Ca_PKC;
		k	100000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PKC -1 ] [ S1 Variable:/CELL/CYTOPLASM:Ca -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_PKC 1 ];
	}
	
	Process MassActionProcess( Ca_PKC_PKC )
	{
		Name	Ca_PKC_PKC;
		k	1;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:PKC 1 ] [ P1 Variable:/CELL/CYTOPLASM:Ca 1 ];
	}
	
	Process MassActionProcess( Ca_PKC_Ca_DAG_PKC )
	{
		Name	Ca_PKC_Ca_DAG_PKC;
		k	100000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_PKC -1 ] [ S1 Variable:/CELL/CYTOPLASM:DAG -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_DAG_PKC 1 ];
	}
	
	Process MassActionProcess( Ca_DAG_PKC_Ca_PKC )
	{
		Name	Ca_DAG_PKC_Ca_PKC;
		k	0.05;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_DAG_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_PKC 1 ] [ P1 Variable:/CELL/CYTOPLASM:DAG 1 ];
	}
	
	Process MassActionProcess( PKC_DAG_PKC )
	{
		Name	PKC_DAG_PKC;
		k	500;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PKC -1 ] [ S1 Variable:/CELL/CYTOPLASM:DAG -1 ] [ P0 Variable:/CELL/CYTOPLASM:DAG_PKC 1 ];
	}
	
	Process MassActionProcess( DAG_PKC_PKC )
	{
		Name	DAG_PKC_PKC;
		k	0.00025;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:DAG_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:PKC 1 ] [ P1 Variable:/CELL/CYTOPLASM:DAG 1 ];
	}
	
	Process MassActionProcess( DAG_PKC_DAG_AA_PKC )
	{
		Name	DAG_PKC_DAG_AA_PKC;
		k	10;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:DAG_PKC -1 ] [ S1 Variable:/CELL/CYTOPLASM:AA -1 ] [ P0 Variable:/CELL/CYTOPLASM:DAG_AA_PKC 1 ];
	}
	
	Process MassActionProcess( DAG_AA_PKC_DAG_PKC )
	{
		Name	DAG_AA_PKC_DAG_PKC;
		k	0.0005;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:DAG_AA_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:DAG_PKC 1 ] [ P1 Variable:/CELL/CYTOPLASM:AA 1 ];
	}
	
	Process MassActionProcess( PKC_AA_PKC )
	{
		Name	PKC_AA_PKC;
		k	100;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PKC -1 ] [ S1 Variable:/CELL/CYTOPLASM:AA -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA_PKC 1 ];
	}
	
	Process MassActionProcess( AA_PKC_PKC )
	{
		Name	AA_PKC_PKC;
		k	0.005;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:AA_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:PKC 1 ] [ P1 Variable:/CELL/CYTOPLASM:AA 1 ];
	}
	
	Process MassActionProcess( Ca_PKC_Ca_AA_PKC )
	{
		Name	Ca_PKC_Ca_AA_PKC;
		k	1500;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_PKC -1 ] [ S1 Variable:/CELL/CYTOPLASM:AA -1 ] [ P0 Variable:/CELL/CYTOPLASM:Ca_AA_PKC 1 ];
	}
	
	Process MassActionProcess( Ca_AA_PKC_Ca_PKC )
	{
		Name	Ca_AA_PKC_Ca_PKC;
		k	0.075;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_AA_PKC -1 ] [ P0 Variable:/CELL/CYTOPLASM:AA 1 ] [ P1 Variable:/CELL/CYTOPLASM:Ca_PKC 1 ];
	}
	
	Process MassActionProcess( mGluR_Gq_Glu_mGluR_Gq )
	{
		Name	mGluR_Gq_Glu_mGluR_Gq;
		k	10000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:mGluR_Gq -1 ] [ S1 Variable:/CELL/CYTOPLASM:Glu -1 ] [ P0 Variable:/CELL/CYTOPLASM:Glu_mGluR_Gq 1 ];
	}
	
	Process MassActionProcess( Glu_mGluR_Gq_mGluR_Gq )
	{
		Name	Glu_mGluR_Gq_mGluR_Gq;
		k	0.00006;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Glu_mGluR_Gq -1 ] [ P0 Variable:/CELL/CYTOPLASM:mGluR_Gq 1 ] [ P1 Variable:/CELL/CYTOPLASM:Glu 1 ];
	}
	
	Process MassActionProcess( Glu_mGluR_Gq_GTP_Gbc_Glu )
	{
		Name	Glu_mGluR_Gq_GTP_Gbc_Glu;
		k	0.1;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Glu_mGluR_Gq -1 ] [ P0 Variable:/CELL/CYTOPLASM:GTP_Ga 1 ] [ P1 Variable:/CELL/CYTOPLASM:Gbc 1 ] [ P2 Variable:/CELL/CYTOPLASM:Glu_mGluR 1 ];
	}
	
	Process MassActionProcess( GTP_Gbc_Glu_Glu_mGluR_Gq )
	{
		Name	GTP_Gbc_Glu_Glu_mGluR_Gq;
		k	0.0000000001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GTP_Ga -1 ] [ S1 Variable:/CELL/CYTOPLASM:Gbc -1 ] [ S2 Variable:/CELL/CYTOPLASM:Glu_mGluR -1 ] [ P0 Variable:/CELL/CYTOPLASM:Glu_mGluR_Gq 1 ];
	}
	
	Process MassActionProcess( GTP_Ga_GDP_Ga )
	{
		Name	GTP_Ga_GDP_Ga;
		k	0.1;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GTP_Ga -1 ] [ P0 Variable:/CELL/CYTOPLASM:GDP_Ga 1 ];
	}
	
	Process MassActionProcess( GTP_Ga_GTP_Ga_PLC )
	{
		Name	GTP_Ga_GTP_Ga_PLC;
		k	1000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GTP_Ga -1 ] [ S1 Variable:/CELL/CYTOPLASM:PLC -1 ] [ P0 Variable:/CELL/CYTOPLASM:GTP_Ga_PLC 1 ];
	}
	
	Process MassActionProcess( GTP_Ga_PLC_GTP_Ga )
	{
		Name	GTP_Ga_PLC_GTP_Ga;
		k	0.0000397;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GTP_Ga_PLC -1 ] [ P0 Variable:/CELL/CYTOPLASM:GTP_Ga 1 ] [ P1 Variable:/CELL/CYTOPLASM:PLC 1 ];
	}
	
	Process MichaelisUniUniProcess( PIP2_IP3 )
	{
		Name	PIP2_IP3;
		Km	0.0000005;
		Kcat	48;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PIP2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:IP3 1 ] [ C0 Variable:/CELL/CYTOPLASM:GTP_Ga_PLC];
	}
	
	Process MichaelisUniUniProcess( PIP2_DAG )
	{
		Name	PIP2_DAG;
		Km	0.0000005;
		Kcat	48;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:PIP2 -1 ] [ P0 Variable:/CELL/CYTOPLASM:DAG 1 ] [ C0 Variable:/CELL/CYTOPLASM:GTP_Ga_PLC];
	}
	
	Process MassActionProcess( DeIP3 )
	{
		Name	DeIP3;
		k	10;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:IP3 -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( GDP_Ga_Gabc )
	{
		Name	GDP_Ga_Gabc;
		k	500;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:GDP_Ga -1 ] [ S1 Variable:/CELL/CYTOPLASM:Gbc -1 ] [ P0 Variable:/CELL/CYTOPLASM:Gabc 1 ];
	}
	
	Process MassActionProcess( Gabc_GDP_Ga )
	{
		Name	Gabc_GDP_Ga;
		k	0.0000005;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Gabc -1 ] [ P0 Variable:/CELL/CYTOPLASM:GDP_Ga 1 ] [ P1 Variable:/CELL/CYTOPLASM:Gbc 1 ];
	}
	
	Process MassActionProcess( Glu_mGluR_mGluR_Glu )
	{
		Name	Glu_mGluR_mGluR_Glu;
		k	0.1;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Glu_mGluR -1 ] [ P0 Variable:/CELL/CYTOPLASM:MgluR 1 ] [ P1 Variable:/CELL/CYTOPLASM:Glu 1 ];
	}
	
	Process MassActionProcess( DeGlu )
	{
		Name	DeGlu;
		k	0.001;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Glu -1 ] [ P0 Variable:/CELL/CYTOPLASM:N 1 ];
	}
	
	Process MassActionProcess( MgluR_mGluR_Gq )
	{
		Name	MgluR_mGluR_Gq;
		k	1000;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MgluR -1 ] [ S1 Variable:/CELL/CYTOPLASM:Gabc -1 ] [ P0 Variable:/CELL/CYTOPLASM:mGluR_Gq 1 ];
	}
	
	Process MassActionProcess( mGluR_Gq_MgluR )
	{
		Name	mGluR_Gq_MgluR;
		k	0.00167;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:mGluR_Gq -1 ] [ P0 Variable:/CELL/CYTOPLASM:MgluR 1 ] [ P1 Variable:/CELL/CYTOPLASM:Gabc 1 ];
	}
	

	Process MakesignalPProcess( ADD )
	{
		Name	add;
		StepperID DT_1;	
		add	0.000000012;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:NO] [ P0 Variable:/CELL/CYTOPLASM:NO 1 ];
	}


	Process MakesignalPProcess( ADDCa )
	{
		Name	addCa;
		StepperID DT_1;	
		add	0.00000055;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca] [ P0 Variable:/CELL/CYTOPLASM:Ca 1 ];
	}

	Process MakesignalPProcess( ADDGlu )
	{
		Name	addGlu;
		StepperID DT_1;	
		add	0.0000012;
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Glu] [ P0 Variable:/CELL/CYTOPLASM:Glu 1 ];
	}
	
	Process PKCactivePProcess( PKC_PKCactive )
	{
		Name	PKC_PKC_active;
		StepperID SLAVE_1;	
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:Ca_DAG_PKC] [ S1 Variable:/CELL/CYTOPLASM:DAG_AA_PKC] [ S2 Variable:/CELL/CYTOPLASM:AA_PKC] [ S3 Variable:/CELL/CYTOPLASM:Ca_AA_PKC] [ P0 Variable:/CELL/CYTOPLASM:PKCactive 1 ];
	}
	
	Process MAPPProcess( MAPact )
	{
		Name	MAPact;
		StepperID SLAVE_1;	
		VariableReferenceList	[ S0 Variable:/CELL/CYTOPLASM:MAPK_P] [ S1 Variable:/CELL/CYTOPLASM:MAPK_PP] [ P0 Variable:/CELL/CYTOPLASM:MAPKactive 1 ];
	}
	
	
}


