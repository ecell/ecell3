from ecell.ECDDataFile import *

# --------------------------------------------------------
# (1) load eml fil
loadModel( _EML_ )

# --------------------------------------------------------
# (2) set parameter
anEntity = createEntityStub( 'Process:/:E' )
anEntity.setProperty( 'KmS', _KmS_ )
anEntity.setProperty( 'KcF', _KcF_ )

# --------------------------------------------------------
# (3) create logger stubs
S_Logger = createLoggerStub( 'Variable:/:S:Value' )
S_Logger.create()
S = createEntityStub( 'Variable:/:S' )
P_Logger = createLoggerStub( 'Variable:/:P:Value' )
P_Logger.create()
P = createEntityStub( 'Variable:/:P' )

# --------------------------------------------------------
# (4) run
run( 1000 )

# --------------------------------------------------------
# (5) read training time-course
aTrainingS = ECDDataFile()
aTrainingS.load( '%s/S.ecd' %_DATA_ )
aTrainingP = ECDDataFile()
aTrainingP.load( '%s/P.ecd' %_DATA_ )

# --------------------------------------------------------
# (6) save predicted time-course
aPredictedS = ECDDataFile( S_Logger.getData(0,1000,10) )
aPredictedP = ECDDataFile( P_Logger.getData(0,1000,10) )

aPredictedS.setDataName( S_Logger.getName() )
aPredictedS.setNote( 'predicted S' )
aPredictedS.save( 'preS.ecd' )

aPredictedP.setDataName( P_Logger.getName() )
aPredictedP.setNote( 'predicted P' )
aPredictedP.save( 'preP.ecd' )


# --------------------------------------------------------
# (7) calculates the difference between the training - and 
#     the predicted time-course.
aDifference = 0.0
for anIndex in xrange(0,aTrainingS.getSize()[1]):

	# denominator
	aDenominator = aTrainingS.getData()[anIndex][1] 
	if aDenominator < 0.001:
		aDenominator = 0.001 

	# add the difference of predicted S and predicted S
	aDifference += (abs(aTrainingS.getData()[anIndex][1]- 
					    aPredictedS.getData()[anIndex][1])/
	                    aDenominator)
	# denominator
	aDenominator = aTrainingP.getData()[anIndex][1] 
	if aDenominator < 0.001:
		aDenominator = 0.001

	# add the difference of predicted P and predicted P
	aDifference += (abs(aTrainingP.getData()[anIndex][1]- 
	                    aPredictedP.getData()[anIndex][1])/ 
	                    aDenominator)

aDifference /= aTrainingS.getSize()[1]

# --------------------------------------------------------
# (8) write the value of fitness function to 'result.dat'
open('result.dat','w').write(str(aDifference))

# optional procedures
# write setting file for gnuplot
aContents  = "set ylabel \"Quantity\"\n"
aContents += "set xlabel \"Time\"\n"
aContents += "set linestyle 1 lt 2 lw 2\n"
aContents += "set key box linestyle 1\n"
aContents += "plot \"%s/S.ecd\" title \"predicted S\" with lines,\\\n" %_DATA_
aContents += "     \"%s/P.ecd\" title \"predicted P\" with lines,\\\n" %_DATA_
aContents += "     \"preS.ecd\" title \"predicted S\" with lines,\\\n" 
aContents += "     \"preP.ecd\" title \"predicted P\" with line\n" 
aContents += "pause -1"

open('gnuplot.dat','w').write(aContents)

# end of this file
