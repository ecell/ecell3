
# created by eml2em program
# from file: LTD.eml, date: Mon Oct 14 19:51:27 2002
#

#Stepper FixedRungeKutta4Stepper( DE_1 )
#Stepper ODE23Stepper( DE_1 )
Stepper ODE45Stepper( DE_1 )
{
	# no property
}

Stepper DiscreteTimeStepper( DT_1 )
#Stepper PassiveStepper( DT_1 )
{
	StepInterval 0.01;
}


Stepper PassiveStepper( PASSIVE_1 )
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
	Name	"The menbrane";
}

System System( /CELL/CYTOPLASM )
{
	StepperID	DE_1;
	Name	"The cytoplasm";

	Variable Variable( SIZE )
	{
		Value	0.000000000001;
	}

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
	
	Process MassActionFluxProcess( Ca_DAG_PKC_N )
	{
		Name	Ca_DAG_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 :.:Ca_DAG_PKC -1 ]
	 [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( DAG_AA_PKC_N )
	{
		Name	DAG_AA_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 :.:DAG_AA_PKC -1 ] [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( AA_PKC_N )
	{
		Name	AA_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 :.:AA_PKC -1 ] [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( Ca_AA_PKC_N )
	{
		Name	Ca_AA_PKC_N;
		k	0.004;
		VariableReferenceList	[ S0 :.:Ca_AA_PKC -1 ] [ P0 :.:N 1 ];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P1 )
	{
		Name	AMPAR_AMPAR_P1;
		KmS	0.0000035;
		KcF	1.5;
		VariableReferenceList	[ S0 :.:AMPAR -1 ] [ P0 :.:AMPAR_P 1 ] [ C0 :.:Ca_DAG_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P2 )
	{
		Name	AMPAR_AMPAR_P2;
		KmS	0.0000035;
		KcF	1.5;
		VariableReferenceList	[ S0 :.:AMPAR -1 ] [ P0 :.:AMPAR_P 1 ] [ C0 :.:DAG_AA_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P3 )
	{
		Name	AMPAR_AMPAR_P3;
		KmS	0.0000035;
		KcF	1.5;
		VariableReferenceList	[ S0 :.:AMPAR -1 ] [ P0 :.:AMPAR_P 1 ] [ C0 :.:AA_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_AMPAR_P4 )
	{
		Name	AMPAR_AMPAR_P4;
		KmS	0.0000035;
		KcF	1.5;
		VariableReferenceList	[ S0 :.:AMPAR -1 ] [ P0 :.:AMPAR_P 1 ] [ C0 :.:Ca_AA_PKC];
	}
	
	Process MichaelisUniUniProcess( AMPAR_P_AMPAR )
	{
		Name	AMPAR_P_AMPAR;
		KmS	0.00001565;
		KcF	6;
		VariableReferenceList	[ S0 :.:AMPAR_P -1 ] [ P0 :.:AMPAR 1 ] [ C0 :.:PP2A];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P1 )
	{
		Name	Raf_Raf_P1;
		KmS	0.0000116;
		KcF	0.0335;
		VariableReferenceList	[ S0 :.:Raf -1 ] [ P0 :.:Raf_P 1 ] [ C0 :.:Ca_DAG_PKC];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P2 )
	{
		Name	Raf_Raf_P2;
		KmS	0.0000116;
		KcF	0.0335;
		VariableReferenceList	[ S0 :.:Raf -1 ] [ P0 :.:Raf_P 1 ] [ C0 :.:DAG_AA_PKC];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P3 )
	{
		Name	Raf_Raf_P3;
		KmS	0.0000116;
		KcF	0.0335;
		VariableReferenceList	[ S0 :.:Raf -1 ] [ P0 :.:Raf_P 1 ] [ C0 :.:AA_PKC];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P4 )
	{
		Name	Raf_Raf_P4;
		KmS	0.0000116;
		KcF	0.0335;
		VariableReferenceList	[ S0 :.:Raf -1 ] [ P0 :.:Raf_P 1 ] [ C0 :.:Ca_AA_PKC];
	}
	
	Process MassActionFluxProcess( CRHR_CRHR_CRF )
	{
		Name	CRHR_CRHR_CRF;
		k	100000;
		VariableReferenceList	[ S0 :.:CRHR -1 ] [ S1 :.:CRF -1 ] [ P0 :.:CRHR_CRF 1 ];
	}
	
	Process MassActionFluxProcess( CRHR_CRF_CRHR )
	{
		Name	CRHR_CRF_CRHR;
		k	0.00001;
		VariableReferenceList	[ S0 :.:CRHR_CRF -1 ] [ P0 :.:CRHR 1 ] [ P1 :.:CRF 1 ];
	}
	
	Process MassActionFluxProcess( DeCRF )
	{
		Name	DeCRF;
		k	0.02;
		VariableReferenceList	[ S0 :.:CRF -1 ] [ P0 :.:N 1 ];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P22 )
	{
		Name	Raf_Raf_P22;
		KmS	0.00002;
		KcF	0.0025;
		VariableReferenceList	[ S0 :.:Raf -1 ] [ P0 :.:Raf_P 1 ] [ C0 :.:CRHR_CRF];
	}
	
	Process MichaelisUniUniProcess( Raf_Raf_P33 )
	{
		Name	Raf_Raf_P33;
		KmS	0.00003;
		KcF	0.08;
		VariableReferenceList	[ S0 :.:Raf -1 ] [ P0 :.:Raf_P 1 ] [ C0 :.:Lyn_activate];
	}
	
	Process MassActionFluxProcess( DeLyn_activate )
	{
		Name	DeLyn_activate;
		k	0.001;
		VariableReferenceList	[ S0 :.:Lyn_activate -1 ] [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( AA_APC )
	{
		Name	AA_APC;
		k	0.001;
		VariableReferenceList	[ S0 :.:AA -1 ] [ P0 :.:APC 1 ];
	}
	
	Process MassActionFluxProcess( NO_N )
	{
		Name	NO_N;
		k	0.636;
		VariableReferenceList	[ S0 :.:NO -1 ] [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( GC__NO_NO_GC )
	{
		Name	GC__NO_NO_GC;
		k	3000000;
		VariableReferenceList	[ S0 :.:GC -1 ] [ S1 :.:NO -1 ] [ P0 :.:NO_GC 1 ];
	}
	
	Process MassActionFluxProcess( NO_GC_GC__NO )
	{
		Name	NO_GC_GC__NO;
		k	0.75;
		VariableReferenceList	[ S0 :.:NO_GC -1 ] [ P0 :.:GC 1 ] [ P1 :.:NO 1 ];
	}
	
	Process MichaelisUniUniProcess( GTP_CGMP )
	{
		Name	GTP_CGMP;
		KmS	0.000045;
		KcF	7.35;
		VariableReferenceList	[ S0 :.:GTP -1 ] [ P0 :.:CGMP 1 ] [ C0 :.:NO_GC];
	}
	
	Process MichaelisUniUniProcess( CGMP_GMP5 )
	{
		Name	CGMP_GMP5;
		KmS	0.000002;
		KcF	3.87;
		VariableReferenceList	[ S0 :.:CGMP -1 ] [ P0 :.:GMP5 1 ] [ C0 :.:PDE];
	}
	
	Process MassActionFluxProcess( CGMP__PKG_cGMP_PKG )
	{
		Name	CGMP__PKG_cGMP_PKG;
		k	100000;
		VariableReferenceList	[ S0 :.:CGMP -1 ] [ S1 :.:PKG -1 ] [ P0 :.:cGMP_PKG 1 ];
	}
	
	Process MassActionFluxProcess( cGMP_PKG_CGMP__PKG )
	{
		Name	cGMP_PKG_CGMP__PKG;
		k	0.005;
		VariableReferenceList	[ S0 :.:cGMP_PKG -1 ] [ P0 :.:CGMP 1 ] [ P1 :.:PKG 1 ];
	}
	
	Process MichaelisUniUniProcess( G_Sub_G_Sub_P )
	{
		Name	G_Sub_G_Sub_P;
		KmS	0.0000002;
		KcF	0.72;
		VariableReferenceList	[ S0 :.:G_Sub -1 ] [ P0 :.:G_Sub_P 1 ] [ C0 :.:cGMP_PKG];
	}
	
	Process MassActionFluxProcess( G_Sub_P_G_Sub )
	{
		Name	G_Sub_P_G_Sub;
		k	0.0001;
		VariableReferenceList	[ S0 :.:G_Sub_P -1 ] [ P0 :.:G_Sub 1 ];
	}
	
	Process MassActionFluxProcess( G_Sub_P__PP2A_PP2A_G_Sub_P )
	{
		Name	G_Sub_P__PP2A_PP2A_G_Sub_P;
		k	10000;
		VariableReferenceList	[ S0 :.:G_Sub_P -1 ] [ S1 :.:PP2A -1 ] [ P0 :.:PP2A_G_Sub_P 1 ];
	}
	
	Process MassActionFluxProcess( PP2A_G_Sub_P )
	{
		Name	PP2A_G_Sub_P;
		k	0.0027;
		VariableReferenceList	[ S0 :.:PP2A_G_Sub_P -1 ] [ P0 :.:G_Sub_P 1 ] [ P1 :.:PP2A 1 ];
	}
	
	
	Process MichaelisUniUniProcess( Raf_P_Raf )
	{
		Name	Raf_P_Raf;
		KmS	0.0000157;
		KcF	6;
		VariableReferenceList	[ S0 :.:Raf_P -1 ] [ P0 :.:Raf 1 ] [ C0 :.:PP2A];
	}
	
	Process MichaelisUniUniProcess( MEK_MEK_P )
	{
		Name	MEK_MEK_P;
		KmS	0.000000398;
		KcF	0.105;
		VariableReferenceList	[ S0 :.:MEK -1 ] [ P0 :.:MEK_P 1 ] [ C0 :.:Raf_P];
	}
	
	Process MichaelisUniUniProcess( MEK_P_MEK_PP )
	{
		Name	MEK_P_MEK_PP;
		KmS	0.000000398;
		KcF	0.105;
		VariableReferenceList	[ S0 :.:MEK_P -1 ] [ P0 :.:MEK_PP 1 ] [ C0 :.:Raf_P];
	}
	
	Process MichaelisUniUniProcess( MEK_P_MEK )
	{
		Name	MEK_P_MEK;
		KmS	0.0000157;
		KcF	6;
		VariableReferenceList	[ S0 :.:MEK_P -1 ] [ P0 :.:MEK 1 ] [ C0 :.:PP2A];
	}
	
	Process MichaelisUniUniProcess( MEK_PP_MEK_P )
	{
		Name	MEK_PP_MEK_P;
		KmS	0.0000157;
		KcF	6;
		VariableReferenceList	[ S0 :.:MEK_PP -1 ] [ P0 :.:MEK_P 1 ] [ C0 :.:PP2A];
	}
	
	Process MichaelisUniUniProcess( MAPK_MAPK_P )
	{
		Name	MAPK_MAPK_P;
		KmS	0.0000000463;
		KcF	0.15;
		VariableReferenceList	[ S0 :.:MAPK _1 ] [ P0 :.:MAPK_P 1 ] [ C0 :.:MEK_PP];
	}
	
	Process MichaelisUniUniProcess( MAPK_P_MAPK_PP )
	{
		Name	MAPK_P_MAPK_PP;
		KmS	0.0000000463;
		KcF	0.15;
		VariableReferenceList	[ S0 :.:MAPK_P -1 ] [ P0 :.:MAPK_PP 1 ] [ C0 :.:MEK_PP];
	}
	
	Process MichaelisUniUniProcess( MAPK_P_MAPK )
	{
		Name	MAPK_P_MAPK;
		KmS	0.000000167;
		KcF	1;
		VariableReferenceList	[ S0 :.:MAPK_P -1 ] [ P0 :.:MAPK 1 ] [ C0 :.:MKP1];
	}
	
	Process MichaelisUniUniProcess( MAPK_PP_MAPK_P )
	{
		Name	MAPK_PP_MAPK_P;
		KmS	0.000000167;
		KcF	1;
		VariableReferenceList	[ S0 :.:MAPK_PP -1 ] [ P0 :.:MAPK_P 1 ] [ C0 :.:MKP1];
	}
	
	Process MassActionFluxProcess( PLA2_Ca_PLA2 )
	{
		Name	PLA2_Ca_PLA2;
		k	10;
		VariableReferenceList	[ S0 :.:PLA2 -1 ] [ S1 :.:Ca -1 ] [ P0 :.:Ca_PLA2 1 ];
	}
	
	Process MassActionFluxProcess( Ca_PLA2_PLA2 )
	{
		Name	Ca_PLA2_PLA2;
		k	0.000001;
		VariableReferenceList	[ S0 :.:Ca_PLA2 -1 ] [ P0 :.:PLA2 1 ] [ P1 :.:Ca 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA1 )
	{
		Name	APC_AA1;
		KmS	0.00002;
		KcF	54;
		VariableReferenceList	[ S0 :.:APC -1 ] [ P0 :.:AA 1 ] [ C0 :.:Ca_PLA2];
	}
	
	Process MassActionFluxProcess( Ca_PLA2_DAG_Ca_PLA2 )
	{
		Name	Ca_PLA2_DAG_Ca_PLA2;
		k	100;
		VariableReferenceList	[ S0 :.:Ca_PLA2 -1 ] [ S1 :.:DAG -1 ] [ P0 :.:DAG_Ca_PLA2 1 ];
	}
	
	Process MassActionFluxProcess( DAG_Ca_PLA2_Ca_PLA2 )
	{
		Name	DAG_Ca_PLA2_Ca_PLA2;
		k	0.0002;
		VariableReferenceList	[ S0 :.:DAG_Ca_PLA2 -1 ] [ P0 :.:Ca_PLA2 1 ] [ P1 :.:DAG 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA2 )
	{
		Name	APC_AA2;
		KmS	0.00002;
		KcF	60;
		VariableReferenceList	[ S0 :.:APC -1 ] [ P0 :.:AA 1 ] [ C0 :.:DAG_Ca_PLA2];
	}
	
	Process MassActionFluxProcess( PLA2_PIP2_PLA2 )
	{
		Name	PLA2_PIP2_PLA2;
		k	1.0;
		VariableReferenceList	[ S0 :.:PLA2 -1 ] [ S1 :.:PIP2 -1 ] [ P0 :.:PIP2_PLA2 1 ];
	}
	
	Process MassActionFluxProcess( PIP2_PLA2_PLA2 )
	{
		Name	PIP2_PLA2_PLA2;
		k	0.0004;
		VariableReferenceList	[ S0 :.:PIP2_PLA2 -1 ] [ P0 :.:PLA2 1 ] [ P1 :.:PIP2 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA3 )
	{
		Name	APC_AA3;
		KmS	0.00002;
		KcF	11.04;
		VariableReferenceList	[ S0 :.:APC -1 ] [ P0 :.:AA 1 ] [ C0 :.:PIP2_PLA2];
	}
	
	Process MassActionFluxProcess( PIP2_PLA2_Ca_PIP2_PLA2 )
	{
		Name	PIP2_PLA2_Ca_PIP2_PLA2;
		k	12000;
		VariableReferenceList	[ S0 :.:PIP2_PLA2 -1 ] [ S1 :.:Ca -1 ] [ P0 :.:Ca_PIP2_PLA2 1 ];
	}
	
	Process MassActionFluxProcess( Ca_PIP2_PLA2_PIP2_PLA2 )
	{
		Name	Ca_PIP2_PLA2_PIP2_PLA2;
		k	0.012;
		VariableReferenceList	[ S0 :.:Ca_PIP2_PLA2 -1 ] [ P0 :.:PIP2_PLA2 1 ] [ P1 :.:Ca 1 ];
	}
	
	Process MichaelisUniUniProcess( APC_AA4 )
	{
		Name	APC_AA4;
		KmS	0.00002;
		KcF	36;
		VariableReferenceList	[ S0 :.:APC -1 ] [ P0 :.:AA 1 ] [ C0 :.:Ca_PIP2_PLA2];
	}
	
	Process MichaelisUniUniProcess( PLA2_PLA2_P )
	{
		Name	PLA2_PLA2_P;
		KmS	0.0000256;
		KcF	20;
		VariableReferenceList	[ S0 :.:PLA2 -1 ] [ P0 :.:PLA2_P 1 ] [ C0 :.:MAPK_PP];
	}
	
	Process MichaelisUniUniProcess( APC_AA5 )
	{
		Name	APC_AA5;
		KmS	0.00002;
		KcF	120;
		VariableReferenceList	[ S0 :.:APC -1 ] [ P0 :.:AA 1 ] [ C0 :.:PLA2_P];
	}
	
	Process MassActionFluxProcess( PLA2_P_PLA2 )
	{
		Name	PLA2_P_PLA2;
		k	0.17;
		VariableReferenceList	[ S0 :.:PLA2_P -1 ] [ P0 :.:PLA2 1 ];
	}
	
	
	Process MassActionFluxProcess( PKC_Ca_PKC )
	{
		Name	PKC_Ca_PKC;
		k	100000;
		VariableReferenceList	[ S0 :.:PKC -1 ] [ S1 :.:Ca -1 ] [ P0 :.:Ca_PKC 1 ];
	}
	
	Process MassActionFluxProcess( Ca_PKC_PKC )
	{
		Name	Ca_PKC_PKC;
		k	1;
		VariableReferenceList	[ S0 :.:Ca_PKC -1 ] [ P0 :.:PKC 1 ] [ P1 :.:Ca 1 ];
	}
	
	Process MassActionFluxProcess( Ca_PKC_Ca_DAG_PKC )
	{
		Name	Ca_PKC_Ca_DAG_PKC;
		k	100000;
		VariableReferenceList	[ S0 :.:Ca_PKC -1 ] [ S1 :.:DAG -1 ] [ P0 :.:Ca_DAG_PKC 1 ];
	}
	
	Process MassActionFluxProcess( Ca_DAG_PKC_Ca_PKC )
	{
		Name	Ca_DAG_PKC_Ca_PKC;
		k	0.05;
		VariableReferenceList	[ S0 :.:Ca_DAG_PKC -1 ] [ P0 :.:Ca_PKC 1 ] [ P1 :.:DAG 1 ];
	}
	
	Process MassActionFluxProcess( PKC_DAG_PKC )
	{
		Name	PKC_DAG_PKC;
		k	500;
		VariableReferenceList	[ S0 :.:PKC -1 ] [ S1 :.:DAG -1 ] [ P0 :.:DAG_PKC 1 ];
	}
	
	Process MassActionFluxProcess( DAG_PKC_PKC )
	{
		Name	DAG_PKC_PKC;
		k	0.00025;
		VariableReferenceList	[ S0 :.:DAG_PKC -1 ] [ P0 :.:PKC 1 ] [ P1 :.:DAG 1 ];
	}
	
	Process MassActionFluxProcess( DAG_PKC_DAG_AA_PKC )
	{
		Name	DAG_PKC_DAG_AA_PKC;
		k	10;
		VariableReferenceList	[ S0 :.:DAG_PKC -1 ] [ S1 :.:AA -1 ] [ P0 :.:DAG_AA_PKC 1 ];
	}
	
	Process MassActionFluxProcess( DAG_AA_PKC_DAG_PKC )
	{
		Name	DAG_AA_PKC_DAG_PKC;
		k	0.0005;
		VariableReferenceList	[ S0 :.:DAG_AA_PKC -1 ] [ P0 :.:DAG_PKC 1 ] [ P1 :.:AA 1 ];
	}
	
	Process MassActionFluxProcess( PKC_AA_PKC )
	{
		Name	PKC_AA_PKC;
		k	100;
		VariableReferenceList	[ S0 :.:PKC -1 ] [ S1 :.:AA -1 ] [ P0 :.:AA_PKC 1 ];
	}
	
	Process MassActionFluxProcess( AA_PKC_PKC )
	{
		Name	AA_PKC_PKC;
		k	0.005;
		VariableReferenceList	[ S0 :.:AA_PKC -1 ] [ P0 :.:PKC 1 ] [ P1 :.:AA 1 ];
	}
	
	Process MassActionFluxProcess( Ca_PKC_Ca_AA_PKC )
	{
		Name	Ca_PKC_Ca_AA_PKC;
		k	1500;
		VariableReferenceList	[ S0 :.:Ca_PKC -1 ] [ S1 :.:AA -1 ] [ P0 :.:Ca_AA_PKC 1 ];
	}
	
	Process MassActionFluxProcess( Ca_AA_PKC_Ca_PKC )
	{
		Name	Ca_AA_PKC_Ca_PKC;
		k	0.075;
		VariableReferenceList	[ S0 :.:Ca_AA_PKC -1 ] [ P0 :.:AA 1 ] [ P1 :.:Ca_PKC 1 ];
	}
	
	Process MassActionFluxProcess( mGluR_Gq_Glu_mGluR_Gq )
	{
		Name	mGluR_Gq_Glu_mGluR_Gq;
		k	10000;
		VariableReferenceList	[ S0 :.:mGluR_Gq -1 ] [ S1 :.:Glu -1 ] [ P0 :.:Glu_mGluR_Gq 1 ];
	}
	
	Process MassActionFluxProcess( Glu_mGluR_Gq_mGluR_Gq )
	{
		Name	Glu_mGluR_Gq_mGluR_Gq;
		k	0.00006;
		VariableReferenceList	[ S0 :.:Glu_mGluR_Gq -1 ] [ P0 :.:mGluR_Gq 1 ] [ P1 :.:Glu 1 ];
	}
	
	Process MassActionFluxProcess( Glu_mGluR_Gq_GTP_Gbc_Glu )
	{
		Name	Glu_mGluR_Gq_GTP_Gbc_Glu;
		k	0.1;
		VariableReferenceList	[ S0 :.:Glu_mGluR_Gq -1 ] [ P0 :.:GTP_Ga 1 ] [ P1 :.:Gbc 1 ] [ P2 :.:Glu_mGluR 1 ];
	}
	
	Process MassActionFluxProcess( GTP_Gbc_Glu_Glu_mGluR_Gq )
	{
		Name	GTP_Gbc_Glu_Glu_mGluR_Gq;
		k	0.0000000001;
		VariableReferenceList	[ S0 :.:GTP_Ga -1 ] [ S1 :.:Gbc -1 ] [ S2 :.:Glu_mGluR -1 ] [ P0 :.:Glu_mGluR_Gq 1 ];
	}
	
	Process MassActionFluxProcess( GTP_Ga_GDP_Ga )
	{
		Name	GTP_Ga_GDP_Ga;
		k	0.1;
		VariableReferenceList	[ S0 :.:GTP_Ga -1 ] [ P0 :.:GDP_Ga 1 ];
	}
	
	Process MassActionFluxProcess( GTP_Ga_GTP_Ga_PLC )
	{
		Name	GTP_Ga_GTP_Ga_PLC;
		k	1000;
		VariableReferenceList	[ S0 :.:GTP_Ga -1 ] [ S1 :.:PLC -1 ] [ P0 :.:GTP_Ga_PLC 1 ];
	}
	
	Process MassActionFluxProcess( GTP_Ga_PLC_GTP_Ga )
	{
		Name	GTP_Ga_PLC_GTP_Ga;
		k	0.0000397;
		VariableReferenceList	[ S0 :.:GTP_Ga_PLC -1 ] [ P0 :.:GTP_Ga 1 ] [ P1 :.:PLC 1 ];
	}
	
	Process MichaelisUniUniProcess( PIP2_IP3 )
	{
		Name	PIP2_IP3;
		KmS	0.0000005;
		KcF	48;
		VariableReferenceList	[ S0 :.:PIP2 -1 ] [ P0 :.:IP3 1 ] [ C0 :.:GTP_Ga_PLC];
	}
	
	Process MichaelisUniUniProcess( PIP2_DAG )
	{
		Name	PIP2_DAG;
		KmS    	0.0000005;
		KcF	48;
		VariableReferenceList	[ S0 :.:PIP2 -1 ] [ P0 :.:DAG 1 ] [ C0 :.:GTP_Ga_PLC];
	}
	
	Process MassActionFluxProcess( DeIP3 )
	{
		Name	DeIP3;
		k	10;
		VariableReferenceList	[ S0 :.:IP3 -1 ] [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( GDP_Ga_Gabc )
	{
		Name	GDP_Ga_Gabc;
		k	500;
		VariableReferenceList	[ S0 :.:GDP_Ga -1 ] [ S1 :.:Gbc -1 ] [ P0 :.:Gabc 1 ];
	}
	
	Process MassActionFluxProcess( Gabc_GDP_Ga )
	{
		Name	Gabc_GDP_Ga;
		k	0.0000005;
		VariableReferenceList	[ S0 :.:Gabc -1 ] [ P0 :.:GDP_Ga 1 ] [ P1 :.:Gbc 1 ];
	}
	
	Process MassActionFluxProcess( Glu_mGluR_mGluR_Glu )
	{
		Name	Glu_mGluR_mGluR_Glu;
		k	0.1;
		VariableReferenceList	[ S0 :.:Glu_mGluR -1 ] [ P0 :.:MgluR 1 ] [ P1 :.:Glu 1 ];
	}
	
	Process MassActionFluxProcess( DeGlu )
	{
		Name	DeGlu;
		k	0.001;
		VariableReferenceList	[ S0 :.:Glu -1 ] [ P0 :.:N 1 ];
	}
	
	Process MassActionFluxProcess( MgluR_mGluR_Gq )
	{
		Name	MgluR_mGluR_Gq;
		k	1000;
		VariableReferenceList	[ S0 :.:MgluR -1 ] [ S1 :.:Gabc -1 ] [ P0 :.:mGluR_Gq 1 ];
	}
	
	Process MassActionFluxProcess( mGluR_Gq_MgluR )
	{
		Name	mGluR_Gq_MgluR;
		k	0.00167;
		VariableReferenceList	[ S0 :.:mGluR_Gq -1 ] [ P0 :.:MgluR 1 ] [ P1 :.:Gabc 1 ];
	}
	
	Process MakesignalProcess( ADD )
	{
		Name	add;
		StepperID DT_1;	

		Impulse	0.000000012;
		Interval 1;
		Duration 300.0;

		VariableReferenceList	[ P0 :.:NO 1 ];
	}


	Process MakesignalProcess( ADDCa )
	{
		Name	addCa;
		StepperID DT_1;	

		Impulse	0.00000055;
		Interval 1;
		Duration 300.0;

		VariableReferenceList	[ P0 :.:Ca 1 ];
	}

	Process MakesignalProcess( ADDGlu )
	{
		Name	addGlu;
		StepperID DT_1;	

		Impulse	0.0000012;
		Interval 1;
		Duration 300.0;
	
		VariableReferenceList	[ P0 :.:Glu 1 ];
	}

	Process PythonProcess( PKC_PKCactive )
	{
		Name	PKC_PKC_active;
		StepperID PASSIVE_1;

		ProcessMethod "P0.Value = R0.Value + R1.Value + R2.Value + R3.Value";	
		VariableReferenceList	[ R0 :.:Ca_DAG_PKC 0 ] 
					[ R1 :.:DAG_AA_PKC 0 ] 
					[ R2 :.:AA_PKC 0 ] 
					[ R3 :.:Ca_AA_PKC 0 ] 
					[ P0 :.:PKCactive 1 ];
	}
	
	Process PythonProcess( MAPact )
	{
		Name	MAPact;
		StepperID PASSIVE_1;	

		ProcessMethod "P0.Value = R0.Value + R1.Value";

		VariableReferenceList	[ R0 :.:MAPK_P 0 ] 
					[ R1 :.:MAPK_PP 0 ]
					[ P0 :.:MAPKactive 1 ];
	}
	
	
}


