
#
# a sample model for Differential/Algebraic Equations
# the pendulum problem, a DAE of index-1
# the coordinate (X, Y), the velocity (U, V), and T represents the tension
#

#Stepper FixedDAE1Stepper( DAES1 )
Stepper DAEStepper( DAES1 )
{
	# ; no property
}

System System( / )
{
	StepperID	DAES1;

	Variable Variable( SIZE )
	{
		Value	1e-18;
	}

	Variable Variable( X )
	{
		Value	0;
	}

	Variable Variable( Y )
	{
	    	Value	-1;
	}

	Variable Variable( U )
	{
		Value	1;
	}

	Variable Variable( V )
	{
		Value	0;
	}

	Variable Variable( T )
	{
		Value	2;
	}

	Process Differential1Process( DP1 )
	{
		VariableReferenceList	[ P0 Variable:.:X 1 ]
					[ C0 Variable:.:U 0 ];
		k	1.0;
	}

#	Process Differential1Process( DP2 )
#	{
#		VariableReferenceList	[ P0 Variable:.:Y 1 ]
#					[ C0 Variable:.:V 0 ];
#		k	1.0;
#	}

	Process Differential2Process( DP3 )
	{
		VariableReferenceList	[ P0 Variable:.:U 1 ]
					[ C0 Variable:.:X 0 ]
					[ C1 Variable:.:T 0 ];
		c	0.0;
	}

#	Process Differential2Process( DP4 )
#	{
#		VariableReferenceList	[ P0 Variable:.:V 1 ]
#					[ C0 Variable:.:Y 0 ]
#					[ C1 Variable:.:T 0 ];
#		c	-1.0;
#	}

	Process Algebraic1Process( AP1 )
	{
		VariableReferenceList	[ C0 Variable:.:X 0 ]
					[ C1 Variable:.:Y 1 ]
					[ C2 Variable:.:U 0 ]
					[ C3 Variable:.:V 1 ];
	}

	Process Algebraic2Process( AP2 )
	{
		VariableReferenceList	[ C0 Variable:.:Y 1 ]
					[ C1 Variable:.:U 0 ]
					[ C2 Variable:.:V 1 ]
					[ C3 Variable:.:T 1 ];
	}

	Process Algebraic3Process( AP3 )
	{
		VariableReferenceList	[ C0 Variable:.:X 0 ]
					[ C1 Variable:.:Y 1 ];
	}

}
