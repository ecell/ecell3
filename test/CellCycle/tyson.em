
# created by eml2em program
# from file: tyson.eml, date: Wed Oct  9 23:43:48 2002
#

Stepper Euler1Stepper( SRM_01 )
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

System System( /CELL/MEMBRANE )
{
	StepperID	SRM_01;
	Volume	unknown;
}

System System( /CELL/CYTOPLASM )
{
	StepperID	SRM_01;
	Volume	1e-18;

	Variable Variable( Cln2 )
	{
		Value	0;
	}
	
	Variable Variable( Clb2 )
	{
		Value	828044;
	}
	
	Variable Variable( Clb2t )
	{
		Value	828044;
	}
	
	Variable Variable( Clb5 )
	{
		Value	48177;
	}
	
	Variable Variable( Clb5t )
	{
		Value	48177;
	}
	
	Variable Variable( Cln3 )
	{
		Value	0;
	}
	
	Variable Variable( Sic1 )
	{
		Value	0;
	}
	
	Variable Variable( Sic1t )
	{
		Value	0;
	}
	
	Variable Variable( Swi5 )
	{
		Value	0;
	}
	
	Variable Variable( Clb2Sic1 )
	{
		Value	0;
	}
	
	Variable Variable( Clb5Sic1 )
	{
		Value	0;
	}
	
	Variable Variable( SBF )
	{
		Value	0;
	}
	
	Variable Variable( MBF )
	{
		Value	0;
	}
	
	Variable Variable( Mcm1 )
	{
		Value	566081;
	}
	
	Variable Variable( mass )
	{
		Value	903321;
	}
	
	Variable Variable( Bck2 )
	{
		Value	0;
	}
	
	Variable Variable( ORI )
	{
		Value	0;
	}
	
	Variable Variable( BUD )
	{
		Value	0;
	}
	
	Variable Variable( SPN )
	{
		Value	578125;
	}
	
	Variable Variable( Hct1t )
	{
		Value	602214;
	}
	
	Variable Variable( Hct1a )
	{
		Value	0;
	}
	
	Variable Variable( Cdc20t )
	{
		Value	481771;
	}
	
	Variable Variable( Cdc20a )
	{
		Value	46370;
	}
	
	Process CyclinsynthesisProcess( Cln2syn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Cln2 1 ] [ C0 Variable:/CELL/CYTOPLASM:SBF 0 ] [ C1 Variable:/CELL/CYTOPLASM:mass 0 ];
		k1	0;
		k2	0.05;
	}
	
	Process DegradeProcess( Cln2deg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Cln2 -1 ];
		kd	0.1;
	}
	
	Process CyclinsynthesisProcess( Clb2syn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Clb2t 1 ] [ C0 Variable:/CELL/CYTOPLASM:Mcm1 0 ] [ C1 Variable:/CELL/CYTOPLASM:mass 0 ];
		k1	0.002;
		k2	0.05;
	}
	
	Process DegradeClb2Process( Clb2deg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb2t -1 ] [ C0 Variable:/CELL/CYTOPLASM:Hct1t 0 ] [ C1 Variable:/CELL/CYTOPLASM:Hct1a 0 ] [ C2 Variable:/CELL/CYTOPLASM:Cdc20a 0 ];
		k1	0.01;
		k2	0.01;
		k3	0.05;
	}
	
	Process CyclinsynthesisProcess( Clb5syn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Clb5t 1 ] [ C0 Variable:/CELL/CYTOPLASM:MBF 0 ] [ C1 Variable:/CELL/CYTOPLASM:mass 0 ];
		k1	0.006;
		k2	0.02;
	}
	
	Process DegradeClb5Process( Clb5deg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb5t -1 ] [ C0 Variable:/CELL/CYTOPLASM:Cdc20a 0 ];
		k1	0.1;
		k2	0.25;
	}
	
	Process Bck2PProcess( !Bck )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Bck2 1 ] [ C0 Variable:/CELL/CYTOPLASM:mass 0 ];
		Bck2	0.0027;
	}
	
	Process Cln3PProcess( !Cln3cal )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Cln3 1 ] [ C0 Variable:/CELL/CYTOPLASM:mass 0 ];
		Max	0.02;
		D	1;
		J	6;
	}
	
	Process Sic1Process( Sic1syn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Sic1t 1 ] [ C0 Variable:/CELL/CYTOPLASM:Swi5 0 ];
		k1	0.02;
		k2	0.1;
	}
	
	Process DegradeProcess( Sic1deg1 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Sic1t -1 ];
		kd	0.01;
	}
	
	Process Sic1ComplexDegradeProcess( Sic1deg2 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Sic1t -1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C1 Variable:/CELL/CYTOPLASM:Bck2 0 ] [ C2 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C3 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C4 Variable:/CELL/CYTOPLASM:Clb2 0 ] [ C5 Variable:/CELL/CYTOPLASM:Sic1t 0 ];
		k1	0.3;
		e1	20;
		e2	2;
		e3	1;
		e4	0.067;
		J	0.05;
	}
	
	Process BiUniProcess( Clb2Sic1syn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Clb2Sic1 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb2 0 ] [ C1 Variable:/CELL/CYTOPLASM:Sic1 0 ];
		k	50;
	}
	
	Process DegradeProcess( Clb2Sic1deg1 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb2Sic1 -1 ];
		kd	0.05;
	}
	
	Process DegradeClb2Process( Clb2Sic1deg2 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb2Sic1 -1 ] [ C0 Variable:/CELL/CYTOPLASM:Hct1t 0 ] [ C1 Variable:/CELL/CYTOPLASM:Hct1a 0 ] [ C2 Variable:/CELL/CYTOPLASM:Cdc20a 0 ];
		k1	0.01;
		k2	0.01;
		k3	0.05;
	}
	
	Process DegradeProcess( Clb2Sic1deg3 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb2Sic1 -1 ];
		kd	0.01;
	}
	
	Process Sic1ComplexDegradeProcess( Clb2Sic1deg4 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb2Sic1 -1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C1 Variable:/CELL/CYTOPLASM:Bck2 0 ] [ C2 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C3 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C4 Variable:/CELL/CYTOPLASM:Clb2 0 ] [ C5 Variable:/CELL/CYTOPLASM:Sic1t 0 ];
		k1	0.3;
		e1	20;
		e2	2;
		e3	1;
		e4	0.067;
		J	0.05;
	}
	
	Process BiUniProcess( Clb5Sic1syn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Clb5Sic1 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C1 Variable:/CELL/CYTOPLASM:Sic1 0 ];
		k	50;
	}
	
	Process DegradeProcess( Clb5Sic1deg1 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb5Sic1 -1 ];
		kd	0.05;
	}
	
	Process DegradeClb5Process( Clb5Sic1deg2 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb5Sic1 -1 ] [ C0 Variable:/CELL/CYTOPLASM:Cdc20a 0 ];
		k1	0.1;
		k2	0.25;
	}
	
	Process DegradeProcess( Clb5Sic1deg3 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb5Sic1 -1 ];
		kd	0.01;
	}
	
	Process Sic1ComplexDegradeProcess( Clb5Sic1deg4 )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Clb5Sic1 -1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C1 Variable:/CELL/CYTOPLASM:Bck2 0 ] [ C2 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C3 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C4 Variable:/CELL/CYTOPLASM:Clb2 0 ] [ C5 Variable:/CELL/CYTOPLASM:Sic1t 0 ];
		k1	0.3;
		e1	20;
		e2	2;
		e3	1;
		e4	0.067;
		J	0.05;
	}
	
	Process Sic1Process( Cdc20tsyn )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Cdc20t 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k1	0.005;
		k2	0.06;
	}
	
	Process DegradeProcess( Cdc20tdeg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Cdc20t -1 ];
		kd	0.08;
	}
	
	Process Cdc20ActivateProcess( Cdc20act )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Cdc20a 1 ] [ C0 Variable:/CELL/CYTOPLASM:Cdc20t 0 ];
		k	1;
	}
	
	Process Cdc20InactivateProcess( Cdc20ina )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Cdc20a -1 ] [ C0 Variable:/CELL/CYTOPLASM:ORI 0 ] [ C1 Variable:/CELL/CYTOPLASM:SPN 0 ];
		k1	0.1;
		k2	10;
		kd	0.08;
	}
	
	Process Hct1ActivateProcess( Hct1act )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Hct1a 1 ] [ C0 Variable:/CELL/CYTOPLASM:Cdc20a 0 ] [ C1 Variable:/CELL/CYTOPLASM:Hct1t 0 ];
		k1	0.04;
		k2	2;
		J	0.05;
	}
	
	Process Hct1InactivateProcess( Hct1ina )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:Hct1a -1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C1 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C2 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C3 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k1	0;
		k2	0.64;
		e1	1;
		e2	0.5;
		e3	1;
		J	0.05;
	}
	
	Process MassCalculateProcess( masscalc )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:mass 1 ] [ C0 Variable:/CELL/CYTOPLASM:mass 0 ];
		m	0.005776;
	}
	
	Process DegradeProcess( ORIdeg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:ORI -1 ];
		kd	0.06;
	}
	
	Process BUDProcess( BUD )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:BUD 1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C1 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C2 Variable:/CELL/CYTOPLASM:Clb5 0 ];
		k	0.3;
		e	1;
	}
	
	Process DegradeProcess( BUDdeg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:BUD -1 ];
		kd	0.06;
	}
	
	Process SPNProcess( SPN )
	{
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:SPN 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k	0.08;
		J	0.2;
	}
	
	Process DegradeProcess( SPNdeg )
	{
		ConnectionList	[ S0 Variable:/CELL/CYTOPLASM:SPN -1 ];
		kd	0.06;
	}
	
	Process SBFcalcPProcess2( !SBF )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:SBF 1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C1 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C2 Variable:/CELL/CYTOPLASM:Bck2 0 ] [ C3 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C4 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k1	0.5;
		k2	6;
		k3	1;
		e1	75;
		e2	0.5;
		J1	0.01;
		J2	0.01;
	}
	
	Process SBFcalcPProcess2( !MBF )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:MBF 1 ] [ C0 Variable:/CELL/CYTOPLASM:Cln2 0 ] [ C1 Variable:/CELL/CYTOPLASM:Cln3 0 ] [ C2 Variable:/CELL/CYTOPLASM:Bck2 0 ] [ C3 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ C4 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k1	0.5;
		k2	6;
		k3	1;
		e1	75;
		e2	0.5;
		J1	0.01;
		J2	0.01;
	}
	
	Process Mcm1calcPProcess( !Mcm1 )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Mcm1 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k1	1;
		k2	0.15;
		J1	1;
		J2	1;
	}
	
	Process Swi5calcPProcess2( !Swi5 )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Swi5 1 ] [ C0 Variable:/CELL/CYTOPLASM:Cdc20a 0 ] [ C1 Variable:/CELL/CYTOPLASM:Clb2 0 ];
		k1	1;
		k2	0.3;
		k3	0.2;
		J1	0.1;
		J2	0.1;
	}
	
	Process DivisionPProcess( !divide )
	{
		Priority	-1;
		ConnectionList	[ C0 Variable:/CELL/CYTOPLASM:Clb2 0 ] [ P0 Variable:/CELL/CYTOPLASM:mass 1 ] [ P1 Variable:/CELL/CYTOPLASM:BUD 1 ] [ P2 Variable:/CELL/CYTOPLASM:SPN 1 ];
		m	0.005776;
	}
	
	Process ORIPProcess( !ORIP )
	{
		Priority	-1;
		ConnectionList	[ C0 Variable:/CELL/CYTOPLASM:Clb2 0 ] [ C1 Variable:/CELL/CYTOPLASM:Clb5 0 ] [ P0 Variable:/CELL/CYTOPLASM:ORI 1 ];
	}
	
	Process TotalPProcess( !clb2total2 )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Clb2 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb2t 0 ] [ C1 Variable:/CELL/CYTOPLASM:Clb2Sic1 0 ];
	}
	
	Process TotalPProcess( !clb5total2 )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Clb5 1 ] [ C0 Variable:/CELL/CYTOPLASM:Clb5t 0 ] [ C1 Variable:/CELL/CYTOPLASM:Clb5Sic1 0 ];
	}
	
	Process Sic1TotalPProcess( !sic1total2 )
	{
		Priority	-1;
		ConnectionList	[ P0 Variable:/CELL/CYTOPLASM:Sic1 1 ] [ C0 Variable:/CELL/CYTOPLASM:Sic1t 0 ] [ C1 Variable:/CELL/CYTOPLASM:Clb2Sic1 0 ] [ C2 Variable:/CELL/CYTOPLASM:Clb5Sic1 0 ];
	}
	
	
}

System System( /ENVIRONMENT )
{
	StepperID	SRM_01;
	Volume	1e-15;
}

