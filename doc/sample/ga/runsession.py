from ecell.ECDDataFile import *
import os

# When you debug this file, delete forehand # of the following
# parameters and execute,
# [%] ecell3-session runsession.py

#_KmS_ = 0.2
#_KcF_ = 1.0
#_DATA_ = 'Data'
#_EML_ = 'simple.eml'


# --------------------------------------------------------
# parameters for runsession.py
# --------------------------------------------------------

# simulation
DULATION = 1000
START_TIME = 0
INTERVAL = 10

# evaluation
MINIMUM_DENOMINATOR = 0.001

# parameter
VARIABLE_LIST_FOR_LOGGER = [ 'Variable:/:S:Value',
                             'Variable:/:P:Value' ]

# input data
TRAINING_DATA_FILE_LIST = [ 'S.ecd',
                            'P.ecd' ]

# output data
PREFIX_OF_PREDICTED_TIMECOURSE = 'pre'

# --------------------------------------------------------
# (1) load eml file
# --------------------------------------------------------
loadModel( _EML_ )

# --------------------------------------------------------
# (2) set parameter
# --------------------------------------------------------
anEntity = createEntityStub( 'Process:/:E' )
anEntity.setProperty( 'KmS', _KmS_ )
anEntity.setProperty( 'KcF', _KcF_ )

# --------------------------------------------------------
# (3) create logger stubs
# --------------------------------------------------------
aLoggerList = []

for i in range( len(VARIABLE_LIST_FOR_LOGGER) ):

	aLogger = createLoggerStub( VARIABLE_LIST_FOR_LOGGER[i] )
	aLogger.create()
	aLoggerList.append( aLogger )


# --------------------------------------------------------
# (4) run
# --------------------------------------------------------
run( DULATION )

# --------------------------------------------------------
# (5) read training time-course
# --------------------------------------------------------
aTrainingTimeCourseList = []

for i in range( len(TRAINING_DATA_FILE_LIST) ):

	aTimeCouse = ECDDataFile()
	aTimeCouse.load( _DATA_ + os.sep + TRAINING_DATA_FILE_LIST[i] )
	aTrainingTimeCourseList.append( aTimeCouse )


# --------------------------------------------------------
# (6) save predicted time-course
# --------------------------------------------------------
aPredictedTimeCouseList = []

for i in range( len(aLoggerList) ):
	
	aTimeCouse = ECDDataFile( aLoggerList[i].getData(START_TIME, \
							 DULATION, \
							 INTERVAL) )
	aTimeCouse.setDataName( aLoggerList[i].getName() )
	aTimeCouse.setNote( 'Predicted %s' %VARIABLE_LIST_FOR_LOGGER[i] )
	aTimeCouse.save( PREFIX_OF_PREDICTED_TIMECOURSE + \
			 TRAINING_DATA_FILE_LIST[i] )

	aPredictedTimeCouseList.append( aTimeCouse )



# --------------------------------------------------------
# (7) calculate the difference between the training and 
#     the prediction simulated time-course.
# --------------------------------------------------------
aDifference = 0.0

for i in range(aTrainingTimeCourseList[0].getSize()[1]):

	for j in range(len(aTrainingTimeCourseList)):

		# denominator
		aDenominator = aTrainingTimeCourseList[j].getData()[i][1] 
		if aDenominator < MINIMUM_DENOMINATOR:
			aDenominator = MINIMUM_DENOMINATOR
			
		# add the difference of predicted S and predicted S
		aDifference += (abs(aTrainingTimeCourseList[j].getData()[i][1]-
				    aPredictedTimeCouseList[j].getData()[i][1])/
				abs(aDenominator))

aDifference /= aTrainingTimeCourseList[0].getSize()[1]

# --------------------------------------------------------
# (8) write the value of fitness function to 'result.dat'
# --------------------------------------------------------
open('result.dat','w').write(str(aDifference))

# --------------------------------------------------------
# optional procedures
# --------------------------------------------------------
# write setting file for gnuplot
aContents  = "set ylabel \"Quantity\"\n"
aContents += "set xlabel \"Time\"\n"
aContents += "set linestyle 1 lt 2 lw 2\n"
aContents += "set key box linestyle 1\n"
aContents += "plot"
for i in range(len(TRAINING_DATA_FILE_LIST)):

	aContents += "     \"%s%s%s\" title \"training %s\" with lines,\\\n" \
	             %(_DATA_,\
	               os.sep,\
	               TRAINING_DATA_FILE_LIST[i],\
	               VARIABLE_LIST_FOR_LOGGER[i] )

for i in range(len(TRAINING_DATA_FILE_LIST)):

	aContents += "     \"%s%s\" title \"predicted %s\" with lines,\\\n" \
	             %( PREFIX_OF_PREDICTED_TIMECOURSE,
	                TRAINING_DATA_FILE_LIST[i],\
	                VARIABLE_LIST_FOR_LOGGER[i] )
	
aContents = aContents[:-3] + '\n'
aContents += "pause -1"

open('gnuplot.dat','w').write(aContents)


# end of this file
