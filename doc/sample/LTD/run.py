import ecell.ecs

loadModel( 'LTD.eml' )


for i in range( 10 ):
	message( ( getCurrentTime(), getNextEvent() ) )

#	try:
#		s = createStepperStub( getNextEvent()[1] )
#		print s.getProperty( 'StepInterval' )
#		print s.getProperty( 'NextStepInterval' )
#	except:
#		pass


	step()

run( 10 )

