# sample of testrun

from ecell.ECDDataFile import *

print "runsession.py --------------------------------> start "

# ---------------------------------------------------------
# load the model
# ---------------------------------------------------------
loadModel( _EML_ )


# ---------------------------------------------------------
# set parameter
# ---------------------------------------------------------
anEntity = createEntityStub( 'Process:/:E' )
anEntity.setProperty( 'KmS', _KmS_ )
anEntity.setProperty( 'KcF', _KcF_ )


# ---------------------------------------------------------
# create logger stubs
# ---------------------------------------------------------
S_Logger = createLoggerStub( 'Variable:/:S:Value' )
S_Logger.create()
S = createEntityStub( 'Variable:/:S' )
P_Logger = createLoggerStub( 'Variable:/:P:Value' )
P_Logger.create()
P = createEntityStub( 'Variable:/:P' )

# ---------------------------------------------------------
# run
# ---------------------------------------------------------
# print some values
message( 'KmS= \t%s' %_KmS_  )
message( 'KcF= \t%s' %_KcF_  )
message( '\n' )

message( 't= \t%s' % getCurrentTime() )
message( 'S:Value= \t%s' % S.getProperty( 'Value' ) )
message( 'P:Value= \t%s' % P.getProperty( 'Value' ) )
# run
duration = 1000
message( '\n' )
message( 'run %s sec.\n' % duration )
run( duration )
# print results
message( 't= \t%s' % getCurrentTime() )
message( 'S:Value= \t%s' % S.getProperty( 'Value' ) )
message( 'P:Value= \t%s' % P.getProperty( 'Value' ) )
message( '\n' )


# ---------------------------------------------------------
# read input files
# ---------------------------------------------------------

# -----------------------------------
# load given time-course
# -----------------------------------

message('loading given S.ecd..')
aGivenS = ECDDataFile()
aGivenS.load( '%s/S.ecd' %_DATA_ )

message('loading given P.ecd..')
aGivenP = ECDDataFile()
aGivenP.load( '%s/P.ecd' %_DATA_ )


# -----------------------------------
# save predicted time-course
# -----------------------------------
aPredictedS = ECDDataFile( S_Logger.getData(0,1000,10) )
aPredictedP = ECDDataFile( P_Logger.getData(0,1000,10) )

message('saving predicted preS.ecd..')
aPredictedS.setDataName( S_Logger.getName() )
aPredictedS.setNote( 'predicted S' )
aPredictedS.save( 'preS.ecd' )

message('saving predicted preP.ecd..')
aPredictedP.setDataName( P_Logger.getName() )
aPredictedP.setNote( 'predicted P' )
aPredictedP.save( 'preP.ecd' )


# ---------------------------------------------------------
# calculates the difference between given time-course and
# predicted one
# ---------------------------------------------------------

aDifference = 0.0
for anIndex in xrange(0,aGivenS.getSize()[1]):

	# denominator
	aDenominator = aGivenS.getData()[anIndex][1] 
	if aDenominator < 0.001:
		aDenominator = 0.001 

	# add the difference of given S and predicted S
	aDifference += (abs(aGivenS.getData()[anIndex][1]- 
					    aPredictedS.getData()[anIndex][1])/
	                    aDenominator)
	# denominator
	aDenominator = aGivenP.getData()[anIndex][1] 
	if aDenominator < 0.001:
		aDenominator = 0.001

	# add the difference of given P and predicted P
	aDifference += (abs(aGivenP.getData()[anIndex][1]- 
	                    aPredictedP.getData()[anIndex][1])/ 
	                    aDenominator)

aDifference /= aGivenS.getSize()[1]

print "aDifference = %s " %aDifference

# ---------------------------------------------------------
# write evaluation value (float) to 'result.dat'
# ga.py reads this file.
# ---------------------------------------------------------
open('result.dat','w').write(str(aDifference))

# optional procedures

# ---------------------------------------------------------
# write setting file for gnuplot
# ---------------------------------------------------------

aContents  = "set ylabel \"Quantity\"\n"
aContents += "set xlabel \"Time\"\n"
aContents += "set linestyle 1 lt 2 lw 2\n"
aContents += "set key box linestyle 1\n"
aContents += "plot \"%s/S.ecd\" title \"given S\" with lines,\\\n" %_DATA_
aContents += "     \"%s/P.ecd\" title \"given P\" with lines,\\\n" %_DATA_
aContents += "     \"preS.ecd\" title \"predicted S\" with lines,\\\n" 
aContents += "     \"preP.ecd\" title \"predicted P\" with line\n" 
aContents += "pause -1"

open('gnuplot.dat','w').write(aContents)


print "runsession.py --------------------------------> end "








