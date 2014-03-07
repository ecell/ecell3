#
#  This is a sample model for GMAProcess
#
#  
#  How to use GMAProcess
#
# *step1  declare stepper
#
#	Stepper ESSYNSStepper( DT_02 )
#	{	
#		TaylorOrder 1;
#	}
#   
#  you should declare ESSYNSStepper like this.
#  you shoule declare ESSYNSStepper same number of decleation GMAProcess
#
#  *step2  declare GMAProcess
#
#	Process GMAProcess( GMASystem )
#	{
#		Name	"GMAProcess";
#		GMASystemMatrix
#		[  0.4  -3    -2 [0 -1 -1] [0.5 -0.1 0] [0.75 0 -0.2] ]
#		[  3    -1.5  0  [0.5 -0.1 0] [0 0.5 0] [0 0 0] ]
#	        [  2    -5    0  [0.75 0 -0.2] [0 0 0.5] [0 0 0] ];
#		VariableReferenceList	
#		[ P0 Variable:/CELL/CYTOPLASM:X1  1 ]
#		[ P1 Variable:/CELL/CYTOPLASM:X2  1 ]
#		[ P2 Variable:/CELL/CYTOPLASM:X3  1 ];
#		
#	}
#
#  you should declare GMAProcess like this.  
#  
# 	[  0.4  -3    -2 [0 -1 -1] [0.5 -0.1 0] [0.75 0 -0.2] ]
#       
#	Forexample, this vector menans following equation.
#
#	X1 = 0.4*X1^0    *X2^-1   *X3^-1 
#	     -3 *X1^0.5  *X2^-0.1 *X3^0 
#	     -2 *X1^0.75 *X2^0    *X3^-0.2
#
#

Stepper ESSYNSStepper( DT_01 )
{
	TaylorOrder 1;
	StepInterval 0.002;
}


System System( / )
{
	StepperID	DT_01;
	Name	"The culture medium";
}

System System( /CELL )
{
	StepperID	DT_01;
	Name	"The cell";
}

System System( /CELL/CYTOPLASM )
{
	StepperID	DT_01;
	Name	"The cytoplasm";

	Variable Variable( X1 )
	{
		Value	0.5;
		Name	X1;
	}

	Variable Variable( X2 )
	{
		Value	0.5;
		Name	X2;
	}

	Variable Variable( X3 )
	{
		Value	1;
		Name	X3;
	}

	Process GMAProcess( GMASystem )
	{
		Name	"GMAProcess";	
		GMASystemMatrix
		[  0.4  -3    -2 [0 -1 -1] [0.5 -0.1 0] [0.75 0 -0.2] ]
		[  3    -1.5  0  [0.5 -0.1 0] [0 0.5 0] [0 0 0] ]
	        [  2    -5    0  [0.75 0 -0.2] [0 0 0.5] [0 0 0] ];
		VariableReferenceList	
		[ P0 Variable:/CELL/CYTOPLASM:X1  1 ]
		[ P1 Variable:/CELL/CYTOPLASM:X2  1 ]
		[ P2 Variable:/CELL/CYTOPLASM:X3  1 ];
		
	}
}
