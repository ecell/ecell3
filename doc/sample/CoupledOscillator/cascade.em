#
# A coupled oscillator model with multi-timescale
#

@{

#
# Parameters
#

CASCADE = 4
SYS_PROCESS = 5
RATIO = 2
INIT_V = 0.1
## INIT_V = 0.25

# 
# Functions
#

def declareStepper( aName ):

	print "Stepper ODE23Stepper( %s )" % ( aName ) 
	print '{'
	print '	# no property'
	print '}'
	print ''

# end of declareStepper

def declareVariable( aName, aValue ):

	print "	Variable Variable( %s )" % ( aName )
	print '	{'
	print "		Value	%lf;" % ( aValue )
	print '	}'
	print ''

# end of declareVariable

def declareFOProcess( aName, \
	anEnzyme, aProduct, aConstant, aStepperID ):

	print "	Process FOProcess( %s )" % aName
	print '	{'
	print "		StepperID	%s;" % (aStepperID)
	print "		k		%lf;" % ( aConstant )
	print ''
	print "		VariableReferenceList	[C0 Variable:%s 0]" % ( anEnzyme )
	print "					[P0 Variable:%s 1];" % ( aProduct )
	print '	}'
	print ''

# end of declareFOProcess

def declareOscillator( i, aConstant, aStepperID ):

	print "System System( /SYSTEM_%d )" % i
	print '{'
	print "	StepperID %s;" % ( aStepperID )
	print ''

	declareVariable( 'SIZE', 1e-16 );

	for j in range( SYS_PROCESS ):

		declareVariable( "X%d_%d" % ( j, i*2 ), 1000000 )
		declareVariable( "X%d_%d" % ( j, i*2+1 ), 0 )

		declareFOProcess( "P%d_%d" % ( j, i*2 ), \
				  "/SYSTEM_%d:X%d_%d" % ( i, j, i*2 ), \
				  "/SYSTEM_%d:X%d_%d" % ( i, j, i*2+1 ), \
				  aConstant, aStepperID )
		declareFOProcess( "P%d_%d" % ( j, i*2+1 ), \
				  "/SYSTEM_%d:X%d_%d" % ( i, j, i*2+1 ), \
				  "/SYSTEM_%d:X%d_%d" % ( i, j, i*2 ), \
				  ( -1 * aConstant ), aStepperID )

	print '}'
	print ''

# end of declareOscillator

}@
@{

#
# Write em file
#

declareStepper( "DES_0" )

for i in range( 1, CASCADE ):
 	declareStepper( "DES_%d" % ( i ) )

}@
System System( / )
{

	StepperID	DES_0;

@{

declareVariable( 'SIZE', 1e-16 );

for i in range( CASCADE - 1 ):
	for j in range( SYS_PROCESS ):
		declareFOProcess( "CONNECT%d_%d" % ( j, i ), \
				  "/SYSTEM_%d:X%d_%d" % ( i, j, i*2 ), \
				  "/SYSTEM_%d:X%d_%d" % ( i+1, j, i*2+3 ), \
				  INIT_V * pow( RATIO, i ), "DES_%d" % ( i ) )
}@

}

@{
for i in range( CASCADE ):
	declareOscillator( i, INIT_V * pow( RATIO, i ), "DES_%d" % ( i ) )
}@
