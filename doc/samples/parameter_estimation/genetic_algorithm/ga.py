#a sample of Genetic Algorithm on SessionManager

from string import *
from ConfigParser import *
import sys
import shutil
import copy
import os
import random
import popen2
import re
import commands
import time

# ------------------------
# Constants
# ------------------------
MIN = 0
MAX = 1
TYPE = 2
SET_UP_OK_RETURN = 0
SET_UP_NG_RETURN = 1

# ------------------------
# Error Codes
# ------------------------
SETTINGFILE_EXIT = 101
SCRIPTFILE_EXIT = 102
RESUMEFILE_EXIT = 103

# ------------------------
# Messages
# ------------------------
SETUP_ERROR = "Setup Error!"
FATAL_ERROR = "Fatal Error!"
ERROR = "Error!"
EXITPROGRAM = "Exit program...."
START_MESSAGE = 'GA on E-Cell3 ........ start '



# ###################################################################################
class RCGA:
	'''Real Coded Genetic Algorithm 
	'''

	# ------------------------------------------------------------------
	def __init__(self, aSetting):
		'''Constructor
		aSetting   -- a Setting instance (Setting)

		'''
		self.theSetting = aSetting

		# Initialize parameters
		self.theCurrentGeneration = 0
		self.theIndividualList = []
		self.theErFileList = None
		self.theEliteIndividual = None
		self.theParameterMap = None
		self.theMutationRatio = None
		self.theEliteImprovedFlag = False

	# end of __init__



	# ------------------------------------------------------------------
	def initialize( self ):
		'''executs initizalize strategy of GA as bellow.
		- initialize random generator
		- sets up parameter
		- generates population
		- write gnuplot file

		Return None
		'''

		print START_MESSAGE

		# ----------------------------------------------------------
		# initialize random generator
		# ----------------------------------------------------------
		
		random.seed(self.theSetting['RANDOM SEED'] )
		
		# ----------------------------------------------------------
		# sets up parameter
		# ----------------------------------------------------------
		self.theParameterMap = {}
		for aParameterLine in self.theSetting['PARAMETER']:
			anElementList = aParameterLine
			aParameter = anElementList[0]
			aMin = atof(anElementList[1])
			aMax = atof(anElementList[2])
			aType = anElementList[3]
			if aType == 'int':
				aMin = int(aMin)
				aMax = int(aMax)
			self.theParameterMap[aParameter] = [ aMin, aMax, aType ]


		# ----------------------------------------------------------
		# generates population
		# ----------------------------------------------------------
		for anIndex in xrange(0,self.theSetting['POPULATION']):

			aCode = RealCodedIndividual( self.theSetting ) 
			self.theIndividualList.append( aCode )

		# set parameter map
		self.theIndividualList[0].setParameterMap( self.theParameterMap )

		# generate population
		for anIndividual in self.theIndividualList:
			anIndividual.constructRandomly()

		# ----------------------------------------------------------
		# resume 
		# ----------------------------------------------------------

		# when resume file is specified, read resume file
		if self.theSetting['RESUME FILE'] != None:

			sys.stdout.write( "Reading resume file ... %s\n" \
			                   %self.theSetting['RESUME FILE'] )

			# read resume data
			aGenoType = self.readResumeFile( self.theSetting['RESUME FILE'] )

			# set resume data to first individual
			self.theIndividualList[0].setGenoType(aGenoType)


		# ----------------------------------------------------------
		# write gnuplot file
		# ----------------------------------------------------------
		# evaluate
		aContents  = "set ylabel \"Evaluated Value\"\n"
		aContents += "set xlabel \"Generation\"\n"
		aContents += "plot \"%s\" with linespoints\n" \
		              %self.theSetting['EVALUATED VALUE FILE']
		aContents += "pause -1"

		open(self.theSetting['EVALUATED VALUE GNUPLOT'],'w').write(aContents)

		# mutation
		aContents  = "set ylabel \"Mutation ratio %\"\n"
		aContents += "set xlabel \"Generation\"\n"
		aContents += "plot \"%s\" with linespoints\n" \
		              %self.theSetting['MUTATION VALUE FILE']
		aContents += "pause -1"

		open(self.theSetting['MUTATION VALUE GNUPLOT'],'w').write(aContents)


	# ------------------------------------------------------------------
	def readResumeFile( self, aFile ):
		'''read resume file
		Return resume data (dict)
        key is parameter symbol
        value is parameter value
		'''

		aGenoType = {}
		aSearch = re.compile('^(\S)+\s+=\s+(\S+)')

		aCounter = 0

		for aLine in open(aFile,'r').readlines():

			# delete comment after #
			anIndex = find( aLine, '#' ) 
			if anIndex != -1:
				aLine = aLine[:anIndex]


			# get parameter symbol and value
			aResult = aSearch.match(aLine)
			if aResult != None:
				aParam, anEqual, aValue = split( aResult.group() )
				aValue = atof( aValue )
				aMessage = " [%s] <--- [%s]\n" %(aParam,aValue)
				sys.stdout.write(aMessage)
				sys.stdout.flush()

			aCounter += 1
			aGenoType[aParam] = aValue

		if aCounter != len(aGenoType):
			aMessage = "%s : same parameter is found in resume input file.\n" %(ERROR)
			sys.stdout.write(aMessage)
			sys.stdout.flush()
			sys.exit(RESUMEFILE_EXIT)


		# validate resume file.
		aParameterMap = {}
		copy.deepcopy( Individual.theParameterMap, aParameterMap )

		aKeyDict = copy.deepcopy( Individual.theParameterMap )

		for aType in aGenoType.keys():
			try:
				del aKeyDict[aType]
			except KeyError:
				aMessage = "%s : %s no such parameter resume input file.\n" %(ERROR, aType)
				sys.stdout.write(aMessage)
				sys.stdout.flush()
				sys.exit(RESUMEFILE_EXIT)
		

		if len(aKeyDict)!=0:
			aMessage = "%s : %s must be set in resume input file.\n" \
			            %(ERROR, str(aKeyDict.keys()))
			sys.stdout.write(aMessage)
			sys.stdout.flush()
			sys.exit(RESUMEFILE_EXIT)

		return aGenoType


	# ------------------------------------------------------------------
	def run( self ):
		'''run a main loop of estimation
		Return : None
		'''

		while(True):

			self.theCurrentGeneration += 1
			self.__executeEvaluateStrategy()
			self.__executeEliteStrategy()

			# clear all jobs
			# clearJob(jobid=-1)
			for job in getSessionProxies():
				job.clearJobDirectory()

			clearAllSessionProxies()

			print "[%s]\t%s" %(self.theCurrentGeneration, \
			       self.theEliteIndividual.getEvaluatedValue())

			#if atoi(self.theSetiting['MAX GENERATION']) <= self.theCurrentGeneration:
			if self.theSetting['MAX GENERATION'] <= self.theCurrentGeneration:
				print "reached max generation"
				break

			if self.theEliteIndividual.getEvaluatedValue() <= self.theSetting['FINISH CONDITION']:
				print "fufilled finish condition"
				break

			self.__executeSelectionStrategy()
			self.__executeCrossoverStrategy()
			self.__executeMutationStrategy()

	# ------------------------------------------------------------------
	def report( self ):
		'''Display the results of GA

		Return None
		'''

		print "----------------------------------------"
		print "|   Result                             |"
		print "----------------------------------------"
		sys.stdout.write( "%s" %self.theEliteIndividual )


	# ------------------------------------------------------------------
	def __executeEvaluateStrategy( self ):
		'''Execute evaluate strategy

		[1] Register ess files to SessionManager
		[2] Run ess files using SessionManager
		[3] Wait until all ess files finish
		[4] Read evaluated values

		Return None
		'''

		# -----------------------------------------------------
		# [1] Register ess files to SessionManager
		# -----------------------------------------------------
		anIndex = 0
		aParameterMapList = []
		aValueList = []

		for anIndividual in self.theIndividualList:

			#aParameterMap = anIndividual.getGenoType()

			aJobIDList = []

			for anIndex in xrange(0, len(self.theSetting['EXTRA DIRS']) ):

				anEssFile = self.theSetting['ESS FILE']
				anExtraFiles = [ self.theSetting['EXTRA DIRS'][anIndex]]
				               #  self.theSetting['EML FILES'][anIndex] ]
				#if self.theCurrentGeneration == 1:
				#	if self.theSetting['RESUME FILE'] != None:

				#sys.exit(0)

				anArgument = {}

				for aKey in anIndividual.getGenoType().keys():
					anArgument[aKey] = anIndividual.getGenoType()[aKey]

				#print anArgument

				#anArgument[self.theSetting['EML KEY']] = self.theSetting['EML FILES'][anIndex]
				anArgument[self.theSetting['DIR KEY']] = self.theSetting['EXTRA DIRS'][anIndex]

				aJobID = registerEcellSession( anEssFile, anArgument, anExtraFiles )

				aJobIDList.append( aJobID )
				anIndividual.setJobIDList( aJobIDList )


#		aJobNumber = len(getQueuedJobList())
		aJobNumber = len(list(getQueuedSessionProxyList()))
#		print getQueuedSessionProxyList()

		# -----------------------------------------------------
		# [2] Run ess files using SessionManager
		# -----------------------------------------------------
		run(block=False)

		# -----------------------------------------------------
		# [3] Wait until all ess files finish
		# -----------------------------------------------------
		aLoopCounter = 0
		aPreRunningSet = set([])
		while(True):

			update()
			time.sleep(1)

			sys.stdout.write("\r")
			sys.stdout.write("(%s/%s)" %(len(list(getFinishedSessionProxyList())),aJobNumber))

			if aLoopCounter % 6 == 1:
				sys.stdout.write(" .    ")
			elif aLoopCounter % 6 == 2:
				sys.stdout.write(" ..   ")
			elif aLoopCounter % 6 == 3: 
				sys.stdout.write(" ...  ")
			elif aLoopCounter % 6 == 4: 
				sys.stdout.write(" .... ")
			elif aLoopCounter % 6 == 5: 
				sys.stdout.write(" .....")
			else:
				sys.stdout.write("      ")

			sys.stdout.flush()
			aLoopCounter += 1

			if isFinished() == True:
				sys.stdout.write("\r")
				sys.stdout.write("(%s/%s)" %(len(list(getFinishedSessionProxyList())),aJobNumber))
				sys.stdout.write(" .....")
				break
 			#elif aLoopCounter > 3 * 600000:
			elif aLoopCounter > 4 * 6000:
                                aRunningJobSet = []
				for anIndividual in self.theIndividualList:
					for aJobID in anIndividual.getJobList():
						if getSessionProxy(aJobID).getStatus() == 1: # RUNNING = 1
							aRunningJobSet.append(aJobID)
				aRunningJobSet = set(aRunningJobSet)

				aStuckJobSet = aPreRunningSet & aRunningJobSet 
				for aStuckJob in aStuckJobSet:
					getSessionProxy(aStuckJob).stop()
					sys.stdout.write("waiting time expired.")

				aPreRunningSet = aRunningJobSet
				aLoopCounter = 0	
			# elif aLoopCounter > 180:
 			# 	stop(0)
 			# 	sys.stdout.write("waiting time expired.")
 			# 	break
						


		# -----------------------------------------------------
		# [4] Read evaluated values
		# -----------------------------------------------------
		for anIndividual in self.theIndividualList:

			anEvaluatedValue = 0

			for aJobID in anIndividual.getJobList():

				#print aJobID

				#print getJobDirectory(aJobID)
				aResultFile = "%s%sresult.dat" %(getJobDirectory(aJobID), os.sep)

				if os.access(aResultFile,os.R_OK) == False:
					sys.stdout.flush()
					sys.stderr.write("\n%s: could not find %s. \n%s must write %s.\n" \
					                 %(ERROR,
					                   str(aResultFile),
					                   getSessionProxy(aJobID).getScriptFileName(),
					                   str(aResultFile)))
					sys.stderr.write("see %s%s%s.\n" 
					                 %(getJobDirectory(aJobID),
					                   os.sep,
					                   getStderrFileName()))
					sys.stderr.write("If you set \'TMP REMOVABLE\' is True, change it to False and try again.\n")
					sys.stderr.flush()
					
					#sys.exit(SCRIPTFILE_EXIT)
					aResult = 1e+100

				else:
	
					aResult = open( aResultFile, 'r').read()
					aResult = atof(aResult)
				anEvaluatedValue += aResult

			anEvaluatedValue /= len(anIndividual.getJobList())
			anIndividual.setEvaluatedValue(anEvaluatedValue)


	# ------------------------------------------------------------------
	def __executeEliteStrategy( self ):
		'''Execute elite strategy

		[1] Find the inidividual who has best evaluated value as elite
		[2] Save elite individual
		Return None
		'''


		# --------------------------------------------------------------
		# [1] Find the inidividual who has best evaluated value as elite
		# --------------------------------------------------------------
		aValueList = []
		for anIndividual in self.theIndividualList:
			aValueList.append(anIndividual.getEvaluatedValue())


		self.theIndividualList.sort()
		aMinValue = self.theIndividualList[0].getEvaluatedValue()

		aMinIndex = 0
		for aValue in aValueList:
			if aValue == aMinValue:
				break
			else:
				pass
			aMinIndex += 1


		# --------------------------------------------------------------
		# [2] Save elite individual
		# --------------------------------------------------------------
		# First generation,
		if type(self.theEliteIndividual) == type(None):

			# -----------------------------------------------------
			# save elite individual
			# -----------------------------------------------------
			self.theEliteIndividual = copy.deepcopy( self.theIndividualList[0] )
			self.theEliteImprovedFlag = True


			# -----------------------------------------------------
			# copy the directories related to elite individual
			# -----------------------------------------------------
			for aJobID in self.theEliteIndividual.getJobList():
				aSrcDir = getSessionProxy(aJobID).getJobDirectory()

				aDstDir = "%s%s%s" %(self.theSetting['ELITE DIR'],\
				                     os.sep,\
				                     os.path.basename(aSrcDir[:-len(os.sep)]))
				shutil.copytree(aSrcDir,aDstDir)

		# Second generation or after
		else:

			# -----------------------------------------------------
			# When the best individual is better than the elite,
			# save best evaluation value as the elite's value
			# -----------------------------------------------------
			if self.theIndividualList[0] < self.theEliteIndividual:

				# ---------------------
				# save elite individual
				# ---------------------
				self.theEliteIndividual = copy.deepcopy( self.theIndividualList[0] )
				self.theEliteImprovedFlag = True

				# ---------------------
				# delete elite directory
				# ---------------------
				#print os.listdir(self.theSetting['ELITE DIR'])
				for aDir in os.listdir(self.theSetting['ELITE DIR']):
					if aDir == '.' or aDir == '..':
						continue
					aDir = "%s%s%s" %(self.theSetting['ELITE DIR'],
					                  os.sep,
					                  aDir)
					shutil.rmtree(aDir)

				# ---------------------
				# copy the directories related to elite individual
				# ---------------------
				for aJobID in self.theEliteIndividual.getJobList():
					aSrcDir = getSessionProxy(aJobID).getJobDirectory()
					aDstDir = "%s%s%s" %(self.theSetting['ELITE DIR'],\
					                     os.sep,\
				                         os.path.basename(aSrcDir[:-len(os.sep)]))
					shutil.copytree(aSrcDir,aDstDir)


			# -----------------------------------------------------
			# When the elite is better than the best individual,
			# replace worst individual with elite 
			# -----------------------------------------------------
			else:
				self.theIndividualList[len(self.theIndividualList)-1] = \
				                           copy.deepcopy( self.theEliteIndividual )
				self.theEliteImprovedFlag = False


		# -----------------------------------------------------------------------
		# Write evaluated value
		# -----------------------------------------------------------------------
		aContents = "%s\t%s\n" %(self.theCurrentGeneration,
		                         self.theEliteIndividual.getEvaluatedValue())

		if self.theCurrentGeneration == 1:
			open( self.theSetting['EVALUATED VALUE FILE'], 'w').write(aContents)
		else:
			open( self.theSetting['EVALUATED VALUE FILE'], 'a').write(aContents)

		# -----------------------------------------------------------------------
		# Write elite individual
		# -----------------------------------------------------------------------
		aContents = "[	%s	]-----------------------------------\n" %(self.theCurrentGeneration)
		aContents += str(self.theEliteIndividual)

		if self.theCurrentGeneration == 1:
			open( self.theSetting['ELITE FILE'], 'w').write(aContents)
		else:
			open( self.theSetting['ELITE FILE'], 'a').write(aContents)

		#sys.stdout.write(aContents)
		#sys.stdout.flush()

		#self.theEcellSessionManager.saveEliteDirectory(anIndexOfElite)

		#print "[GAEstimator.executeEliteStrategy]---------------------------------e"



	# ------------------------------------------------------------------
	def __executeSelectionStrategy( self ):
		'''Sxecute selection strategy

		This strategy is executed according to the following procedures.
		[1] eliminate the specific case
		[2] calculate selection probability
		[3] copy individuals according to selection probability

		Return None
		'''

		# --------------------------------------------------------------
		# [1] eliminate the specific case
		# --------------------------------------------------------------

		# When the intdividual is < 2, this strategy is meaningless.
		# Then do nothing.
		if len(self.theIndividualList) < 2:
			return None

		# --------------------------------------------------------------
		# [2] calculate selection probability
		# --------------------------------------------------------------

		# initialize a list of Pi (selection probability)
		aCopyNumberList = []

		# calculate copy numbers
		for anIndex in xrange(0,len(self.theIndividualList)):
			aP = self.theSetting['ETA PLUS']
			aP = aP-(self.theSetting['ETA PLUS']-self.theSetting['ETA MINUS'])*anIndex/(len(self.theIndividualList)-1)
			aP = int(round(aP))
			aCopyNumberList.append(aP)
	
		# --------------------------------------------------------------
		# [3] copy individuals according to selection probability
		# --------------------------------------------------------------

		# inidialize individual list buffer
		anIndividualListBuffer = []
		
		# copy individuals
		anIndex = 0
		for aCopyNumber in aCopyNumberList:
			for aDummy in xrange(0,aCopyNumber):
				anIndividualListBuffer.append( copy.deepcopy(self.theIndividualList[anIndex]) )
			anIndex += 1

		# replace individual list
		self.theIndividualList = anIndividualListBuffer


	# ------------------------------------------------------------------
	def __executeCrossoverStrategy( self ):
		'''Execute crossover strategy
		This sample support only SPX crossover method.

		Return None
		'''

		# call spx
		self.__spx()


	# ------------------------------------------------------------------
	def __spx(self):
		'''Simplex crossover

		This strategy is executed according to the following procedures.
		[1] Choose m parents Pk (i=1,2,...,m) according to the generational 
		    model used and calculate their center of gravity G, see (5).
		[2] Generate random number rk, see (6)
		[3] Calculate xk, see (7)
		[4] Calculate Ck, see (8)
		[5] Generate an offspring C, see (9)

		Return None
		'''

		aM = self.theSetting['M']
		anUpsilon = atof(self.theSetting['UPSILON'])
		aChildrenList = []

		for anIndividual in self.theIndividualList:

			# ---------------------------------------------------------------------
			# [1] Choose m parents Pk (i=1,2,...,m) according to the generational 
			#    model used and calculate their center of gravity G, see (5).
			# ---------------------------------------------------------------------

			aParentList = []

			for anIndex in xrange(0,aM):
				aRandomIndex = random.randint(0,len(self.theIndividualList)-1)
				aParentList.append( self.theIndividualList[aRandomIndex] )

			aG ={}
			aParent = aParentList[0]
			for aFullID in aParent.getGenoType().keys():
				aG[aFullID] = aParent.getGenoType()[aFullID]

			for aParent in aParentList[1:]:
				for aFullID in aParent.getGenoType().keys():
					aG[aFullID] += aParent.getGenoType()[aFullID]

			for aFullID in aParent.getGenoType():
				aG[aFullID] /= len(aParentList)

			# ---------------------------------------------------------------------
			# [2] Generate random number rk, see (6)
			# ---------------------------------------------------------------------
			anU = random.random()
			aR = []

			for aK in xrange(0,aM-1):
				aR.append( pow(anU,1.0/(aK+1.0)) )

			# ---------------------------------------------------------------------
			# [3] Calculate xk, see (7)
			# ---------------------------------------------------------------------
			aX = []

			for aK in xrange(0,aM):
				aXk = {}
				for aFullID in aParentList[aK].getGenoType().keys():
					aXk[aFullID] = aG[aFullID] + \
					               anUpsilon * ( aParentList[aK].getGenoType()[aFullID] - aG[aFullID] )
				aX.append(aXk)

			# ---------------------------------------------------------------------
			# [4] Calculate Ck, see (8)
			# ---------------------------------------------------------------------
			aC = []

			aCk0 = {}
			for aFullID in aParentList[0].getGenoType().keys():
				aCk0[aFullID] = 0.0

			aC.append(aCk0)

			for aK in xrange(1,aM):
				aCk = {}
				for aFullID in aParentList[aK].getGenoType().keys():
					aCk[aFullID] = aR[aK-1] * ( aX[aK-1][aFullID] - aX[aK][aFullID] + aC[aK-1][aFullID] )
			
				aC.append(aCk)

			# ---------------------------------------------------------------------
			# [5] Generate an offspring C, see (9)
			# ---------------------------------------------------------------------
			for aK in xrange(0,aM):
				aCk = {}
				for aFullID in aParentList[aK].getGenoType().keys():
					aCk[aFullID] = aX[aK][aFullID] + aC[aK][aFullID] 

				aChild = copy.deepcopy( self.theIndividualList[0] )
				aChild.setGenoType( aCk )
				aChild.setEvaluatedValue( None )
				aChildrenList.append( aChild )

				if len(aChildrenList) == len(self.theIndividualList):
					break

			if len(aChildrenList) == len(self.theIndividualList):
				break


		anIndex = 0
		for aChild in self.theIndividualList:
			anIndex +=1
		anIndex = 0
		for aChild in aChildrenList:
			anIndex +=1
		self.theIndividualList = aChildrenList



	# ------------------------------------------------------------------
	def __executeMutationStrategy( self ):
		'''Execute mutation strategy
		Mutate all individuals according to a probability.
		At first the M0 is used as the initial mutation value.
		When the best evaluated value is not changed compared to previous one,
		the mutation ratio is raised by multiplying constantk k(>1.0).
		The mutation ration is fixed MMAX, when it reaches upper limit.
		When the best evaluated value is improved, it is changed to M0.
		'''

		# When first generation
		if self.theMutationRatio == None:

			# save initial value to instance attribute
			self.theMutationRatio = self.theSetting['M0']

		# Second generation and after
		else:

			# When the elite value is improved, set mutation rate
			# as initial value m0
			if self.theEliteImprovedFlag == True:
				self.theMutationRatio = self.theSetting['M0']

			# When the elite value is not improved, multiple 
			# mutation rate by k
			else:

				self.theMutationRatio *= self.theSetting['K']
				
				# When the mutation ratio > the maximum,
				# set it as the maximum mmax
				if self.theMutationRatio > self.theSetting['MMAX']:
					self.theMutationRatio = self.theSetting['MMAX']

		for anIndividual in self.theIndividualList:
			anIndividual.mutate(self.theMutationRatio)

		# -----------------------------------------------------------------------
		# Write mutation value to file
		# -----------------------------------------------------------------------
		aContents = "%s\t%s\n" %(self.theCurrentGeneration,
		                         self.theMutationRatio)

		if self.theCurrentGeneration == 1:
			open( self.theSetting['MUTATION VALUE FILE'], 'w').write(aContents)
		else:
			open( self.theSetting['MUTATION VALUE FILE'], 'a').write(aContents)



# ###################################################################################
class Individual:
	'''Individual class
	has geno type, evaluated value and job ID list.
	'''

	theParameterMap = None    # parameter map
	theSetting = None         # Setting instance


	# ----------------------------------------------------------
	def __init__( self, aSetting ):
		'''Constructor

		aSetting  -- Setting instance
		'''

		Individual.theSetting = aSetting
		self.theGenoType = None
		self.theEvaluatedValue = None
		self.theJobIDList = None


	# ----------------------------------------------------------
	def __cmp__(self,other):
		'''Overwrite __cmp__ method
		The instances of this class are compared according to 
		the evaluated value

		other  --  Individual instance

		Return boolean : the result of comparison 
		'''

		try:
			return cmp(self.theEvaluatedValue,other.theEvaluatedValue)
		except:
			return False


	# ----------------------------------------------------------
	def setJobIDList( self, aJobIDList ):
		'''set a job id list

		aJobIDList  -- a list of lib id (list)

		Return None
		'''

		# set job id list
		self.theJobIDList = aJobIDList



	# ----------------------------------------------------------
	def getJobList( self ):
		'''get the job id list

		Return list of int : job id list
		'''

		# return job id
		return self.theJobIDList 


	# ----------------------------------------------------------
	def setParameterMap( self, aParameterMap ):
		'''Class Method
		Set parameter map

		aParameterMap  --  a parameter map (dict)

		Return None
		'''

		# set parameter map
		Individual.theParameterMap = aParameterMap


	# register class method
	setParameterMap = classmethod(setParameterMap)


	# ----------------------------------------------------------
	def getParameterMap( self ):
		'''Class Method
		Retunr parameter map

		Return dict : parameter map
		'''

		return Individual.theParameterMap

	# register class method
	getParameterMap = classmethod(getParameterMap)


	# ----------------------------------------------------------
	def constructRandomly( self ):
		'''ABSTRACT : This method must be overwrote in subclass
		initialize genotype randomly
		raise NotImplementedError
		'''
                                                                                                         
		# When this method is not implemented in sub class,
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



	# ----------------------------------------------------------
	def getGenoType(self):
		'''Return genotype

		Return dict : genotype
		'''

		# return the genotype
		return self.theGenoType


	# ----------------------------------------------------------
	def setEvaluatedValue(self,anEvaluatedValue):
		'''Set evaluated value

		anEvaluatedValue -- an evaluated value (float)

		Return None
		'''

		# set an evaluated value
		self.theEvaluatedValue = anEvaluatedValue


	# ----------------------------------------------------------
	def getEvaluatedValue(self):
		'''Return evaluated value

		Return float : the evaluated value
		'''

		# return the evaluated value
		return self.theEvaluatedValue



# ###################################################################################
class RealCodedIndividual(Individual):
	'''Individual for Real Coded GA
	'''

	# ----------------------------------------------------------
	def __init__( self, aSetting ):
		'''Constructor
		'''

		# call superclass's constructor
		Individual.__init__(self,aSetting)


	# ----------------------------------------------------------
	def constructRandomly( self ):
		'''initialize genotype randomly

		[1] Generate random number between minimum limit and 
		    maximum one.
		[2] If the type of genotype is integer, round it. 
		[3] Save it.

		Return None
		'''

		# initialize genotype dictonary
		self.theGenoType = {}

		for aFullID in Individual.theParameterMap.keys():
			
			# --------------------------------------------------
			# [1] Generate random number between minimum limit and 
		  	#    maximum one.
			# --------------------------------------------------
			aRandomNumber = random.random()
			aMax = Individual.theParameterMap[aFullID][MAX]
			aMin = Individual.theParameterMap[aFullID][MIN]
			aRandomNumber = (aMax-aMin)*aRandomNumber+aMin
			aType = Individual.theParameterMap[aFullID][TYPE]

			# --------------------------------------------------
			# [2] If the type of genotype is integer, round it. 
			# --------------------------------------------------
			if aType == 'int':
				aRandomNumber = int(round(aRandomNumber))

			# --------------------------------------------------
			# [3] Save it.
			# --------------------------------------------------
			self.theGenoType[aFullID]=aRandomNumber



	# ----------------------------------------------------------
	def __str__(self):
		'''Overwrite __str__ method
		[1] Print evaluated value
		[2] Print geno type
		[3] Print job id list

		Return str : str to be displayed
		'''

		# ------------------------------------
		# [1] Print evaluated value
		# ------------------------------------
		aBuffer = "(Evaluated Value)	= %s\n" %self.theEvaluatedValue

		# ------------------------------------
		# [2] Print geno type
		# ------------------------------------
		if type(self.theGenoType) == dict:
			for aKey in self.theGenoType:
				aBuffer += "%s	= %s\n" %(aKey,self.theGenoType[aKey])
		elif type(self.theGenoType) == list:
			for anElement in self.theGenoType:
				aBuffer += "(%s)\n" %(anElement)
		else:
			aBuffer += "%s\n" %str(self.theGenoType)

		# ------------------------------------
		# [3] Print job id list
		# ------------------------------------
		aBuffer += "job id = %s\n" %str(self.theJobIDList)

		return aBuffer


	# ----------------------------------------------------------
	def setGenoType(self,aGenoType):
		'''Set genotype
		Overwrite super class's method
		[1] Change the value between minimum limit and maximum one.
		[2] If the type of genotype is integer, round it. 
		[3] Set it

		aGenoType -- a dictorynary of genotype (dict)

		Return None
		'''

		for aKey in aGenoType.keys():

			# ------------------------------------------------------------
			# [1] Change the value between minimum limit and maximum one.
			# ------------------------------------------------------------
			if aGenoType[aKey] < Individual.theParameterMap[aKey][MIN]:
				aGenoType[aKey] = Individual.theParameterMap[aKey][MIN]
			elif aGenoType[aKey] > Individual.theParameterMap[aKey][MAX]:
				aGenoType[aKey] = Individual.theParameterMap[aKey][MAX]

			# ------------------------------------------------------------
			# [2] If the type of genotype is integer, round it. 
			# ------------------------------------------------------------
			if Individual.theParameterMap[aKey][TYPE] == 'int':
				aGenoType[aKey] = int(round(aGenoType[aKey]))

		# ------------------------------------------------------------
		# [3] Set it
		# ------------------------------------------------------------
		self.theGenoType = aGenoType



	# ----------------------------------------------------------
	def mutate( self, aMutationRatio ):
		'''
		Overwrite super class's method
		'''

		anIndex = 0
		for aFullID in Individual.theParameterMap.keys():
			
			#if random.random() < self.theMutationRatio/100.0:
			if random.random() < aMutationRatio/100.0:
				#print "mutate -- (%s)"% anIndex

				aRandomNumber = random.random()
				aMax = Individual.theParameterMap[aFullID][MAX]
				aMin = Individual.theParameterMap[aFullID][MIN]
				aRandomNumber = (aMax-aMin)*aRandomNumber+aMin
				self.theGenoType[aFullID] = aRandomNumber

			anIndex += 1



# ###################################################################################
class Setting(ConfigParser):
	'''Setting class
	- does not use sections, user can access values by only option
	- parses setting file
	- validates values
	'''

	# ----------------------------------------------------------
	def __init__(self):
		'''Constructor
		'''

		# calls super class's constructor
		ConfigParser.__init__(self)
		self.theValues = {}

	# end of __init__


	# ----------------------------------------------------------
	def __setitem__(self,aKey,aValue):
		'''overwrite __setitem__
		aKey    --  key (str)
		aValue  --  value (any)

		Return None
		'''

		self.theValues[aKey] = aValue


	# ----------------------------------------------------------
	def __getitem__(self,aKey):
		'''overwrite __getitem__
		aKey    --  key (str)

		Return any : value conneced to the key
		'''

		return self.theValues[aKey] 


	# ----------------------------------------------------------
	def __get(self,aKey):
		'''Return the value specified by aKey
		aKey    --  key (str)

		Return any  : value conneced to the key
		       None : When the key is not found
		'''

		# search value in all sections
		for aSect in self.sections():
			if self.has_option(aSect,aKey) == True:
				return self.get(aSect,aKey)

		# if there is no value connected to the key
		return None

	# end of __getitem__


	# ----------------------------------------------------------
	def read(self,aSettingFile):
		'''read setting file
		aSettingFile - setting file (str)
		Return : None
		'''

		# -----------------------------------------------
		# check argument
		# -----------------------------------------------
		if aSettingFile == None:
			aMessage = "%s: setting file must be specified\n" %FATAL_ERROR 
			sys.stderr.write(aMessage)
			sys.stderr.flush()
			sys.exit(SETTINGFILE_EXIT)

		# -----------------------------------------------
		# check the accessibility of setting file
		# -----------------------------------------------
		if os.access(aSettingFile,os.R_OK) == False:
			aMessage = "Error: can't read %s\n" %aSettingFile
			sys.stderr.write(aMessage)
			sys.exit(SETTINGFILE_EXIT)


		# -----------------------------------------------
		# read setting file
		# -----------------------------------------------
		ConfigParser.read(self,aSettingFile)


		# -----------------------------------------------
		# validate all items read from setting file
		# -----------------------------------------------
		self.__validate()


	# end of read



	# ----------------------------------------------------------
	def __validate(self):
		'''Validate all parameters written in setting file.
		At first each value is read as str.
		If those str values can be converted to expected type,
		save the converted values to Setting instance.
		If not, print the number of error and exit program.

		Return : None
		'''

		# initialize error counter
		aResult = 0

		#[Fundamental]

		# ==========================================================
		# [Seed]
		# ==========================================================

		# seed
		aResult += self.__isInteger('RANDOM SEED',aMin=0)


		# ==========================================================
		# [GA]
		# ==========================================================

		# max generation
		aResult += self.__isInteger('MAX GENERATION',aMin=1)

		# population
		aResult += self.__isInteger('POPULATION',aMin=1)

		# finish condition
		aResult += self.__isFloat('FINISH CONDITION',aMin=0.00)


		# ==========================================================
		# [Input files]
		# ==========================================================

		# key of eml
		#aResult += self.__isStr('EML KEY')

		# eml file names
		#aResult += self.__isList('EML FILES')

		# ess file name
		aResult += self.__isReadableFile('ESS FILE')

		# key of directory
		aResult += self.__isStr('DIR KEY')

		# extra directory names
		aResult += self.__isList('EXTRA DIRS')


		# ------------------------------------------------
		# parameter
		# ------------------------------------------------

		# parameters
		aResult += self.__isList('PARAMETERS')
		aResult += self.__checkParameterFormat()

		# ess file name
		aResult += self.__isReadableFileOrNone('RESUME FILE')

		# ==========================================================
		# [Temporaty directory]
		# ==========================================================

		# working directory
		aResult += self.__isWritableDir('TMP DIRECTORY')

		# tmp dire removable
		aResult += self.__isBoolean('TMPDIR REMOVABLE')

		# ==========================================================
		# [Output files]
		# ==========================================================

		# elite file
		aResult += self.__isWritableFile('ELITE FILE')

		# elite directory
		aResult += self.__isWritableDir('ELITE DIR')

		# evaluated value file
		aResult += self.__isWritableFile('EVALUATED VALUE FILE')

		# evaluated value gnuplot file
		aResult += self.__isWritableFile('EVALUATED VALUE GNUPLOT')

		# mutation value file
		aResult += self.__isWritableFile('MUTATION VALUE FILE')

		# mutation value gnuplot file
		aResult += self.__isWritableFile('MUTATION VALUE GNUPLOT')


		# ==========================================================
		# [Environment]
		# ==========================================================
		# sge
		aResult += self.__isBoolean('USE SGE')

		# dulation time
		aResult += self.__isInteger('DURATION TIME',aMin=1)

		# max cpu
## 		aResult += self.__isInteger('MAX CPU',aMin=1,aMax=100)
		aResult += self.__isInteger('MAX CPU',aMin=1,aMax=400)

		# retry max count
		aResult += self.__isInteger('RETRY MAX COUNT',aMin=0)


		# ==========================================================
		# [Advanced]
		# ==========================================================

		# code
		aResult += self.__isStr('CODE')
		if self['CODE'] != 'Real':
			aMessage = "%s: \"CODE\" is set as \"%s\", current version supports only \"Real\"\n"  %(SETUP_ERROR,self['CODE'])
			sys.stderr.write(aMessage)
			aResult += SET_UP_NG_RETURN 


		# m
		aResult += self.__isInteger('M', aMin=2, aMax=100)

		# eta+
		aResult += self.__isFloat('ETA PLUS', aMin=0.00)

		# eta-
		aResult += self.__isFloat('ETA MINUS', aMin=0.00)

		# upsilon
		aResult += self.__isFloat('UPSILON', aMin=0.00)

		# m0
		aResult += self.__isFloat('M0', aMin=0.00, aMax=100.0)

		# k
		aResult += self.__isFloat('K', aMin=1.00, aMax=100.0)

		# mmax
		aResult += self.__isFloat('MMAX', aMin=0.00, aMax=100.0)


		# ==========================================================
		# check the result flag
		# ==========================================================

		if aResult != 0:
			aMessage  = "" 
			if aResult == 1:
				aMessage += "%s error was found. eixt this program.\n" %aResult
			else:
				aMessage += "%s errors were found. exit this program.\n" %aResult
			sys.stderr.write(aMessage)
			sys.exit(SETTINGFILE_EXIT)

	# end of def __validate
			

	# ----------------------------------------------------------
	def __isInteger(self,aKey,aMin=None,aMax=None):
		'''check the type of value is interger or not.
		aKey      -- a key of value (str)
		aMin  -- the minimum limit of value (int)
		aMax  -- the maximum limit of value (int)

		Return int : the number of error
		'''


		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 
		

		#print self.__get(aKey)
		anIntValue = None

		try:
			anIntValue = atoi(self.__get(aKey))
		except ValueError:
			aMessage = "%s: %s is set as \"%s\", but it must be int. \n" \
			           %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		if aMin != None:
			if anIntValue < aMin:
				aMessage = "%s: %s is set as \"%s\", but it must be >= %s. \n" \
			   	            %(SETUP_ERROR,aKey,self.__get(aKey),aMin)
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN 

		if aMax != None:
			if aMax < anIntValue:
				aMessage = "%s: %s is set as \"%s\", but it must be <= %s. \n" \
			   	            %(SETUP_ERROR,aKey,self.__get(aKey),aMax)
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN 

		self[aKey] = anIntValue
		return SET_UP_OK_RETURN 


	# ----------------------------------------------------------
	def __isFloat(self,aKey,aMin=None,aMax=None):
		'''check the type of value is float or not.
		aKey      -- a key of value (str)
		aMin  -- the minimum limit of value
		aMax  -- the maximum limit of value

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		# -----------------------------------------------------------------
		# check existance
		# -----------------------------------------------------------------
		try:
			aFloatValue = atof(self.__get(aKey))
		except ValueError:
			aMessage = "%s: \"%s\" is set as \"%s\", but it must be float. \n" \
			           %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 
		else:
			self[aKey] = aFloatValue


		# -----------------------------------------------------------------
		# check minimum
		# -----------------------------------------------------------------
		if aMin != None:
			if aFloatValue < aMin:
				aMessage = "%s: \"%s\" is set as \"%s\", but it must be >= %s. \n" \
			   	            %(SETUP_ERROR,aKey,self.__get(aKey),aMin)
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN 

		# -----------------------------------------------------------------
		# check maximum
		# -----------------------------------------------------------------
		if aMax != None:
			if aMax < aFloatValue:
				aMessage = "%s: \"%s\" is set as \"%s\", but it must be <= %s. \n" \
			   	            %(SETUP_ERROR,aKey,self.__get(aKey),aMax)
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN 

		return SET_UP_OK_RETURN 

	# ----------------------------------------------------------
	def __isBoolean(self,aKey):
		'''check the type of value is boolean or not.
		aKey      -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		# -----------------------------------------------------------------
		# check True
		# -----------------------------------------------------------------
		if self.__get(aKey) == 'True' or \
		   self.__get(aKey) == 'TRUE' or \
		   self.__get(aKey) == 'T' or \
		   self.__get(aKey) == 't':
			self[aKey] = True
			return SET_UP_OK_RETURN 


		# -----------------------------------------------------------------
		# check False
		# -----------------------------------------------------------------
		if self.__get(aKey) == 'False' or \
		   self.__get(aKey) == 'FALSE' or \
		   self.__get(aKey) == 'F' or \
		   self.__get(aKey) == 'f':
			self[aKey] = False
			return SET_UP_OK_RETURN 

		aMessage = "%s: \"%s\" is set as \"%s\", but it must be True/False. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey))
		sys.stderr.write(aMessage)
		return SET_UP_NG_RETURN 


	# ----------------------------------------------------------
	def __isStr(self,aKey):
		'''check the type of value is str or not.
		aKey      -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		aStr = self.__get(aKey)

		#print aStr

		# -----------------------------------------------
		# checks type
		# -----------------------------------------------
		if type(aStr) != str:
			aMessage = "%s: \"%s\" is set as \"%s\", but it must be str. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		self[aKey] = aStr

		return SET_UP_OK_RETURN 

	# ----------------------------------------------------------
	def __checkReadableFileList(self, aKey, aFileList):
		'''Check the files can be read or not
		aKey      -- a key of value (str)
		aFileList -- a file list to be checked (str of list)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		aNGNumber = 0

		for aFile in aFileList:

			# -----------------------------------------------
			# check the existance of aFile
			# -----------------------------------------------
			# When the file does not exist,
			if os.path.isfile(aFile) == False:
				aMessage = "%s: could not find \"%s\" set in \"%s\". \n" \
			           %(SETUP_ERROR,aFile,aKey)
				sys.stderr.write(aMessage)
				aNGNumber += 1

			# When the file exists,
			else:

				# -----------------------------------------------
				# check the read permission of aFile
				# -----------------------------------------------
				if os.access( aFile, os.R_OK ) == False:
					aMessage = "%s: could not read \"%s\" set in \"%s\". \n" \
			           %(SETUP_ERROR,aFile,aKey)
					sys.stderr.write(aMessage)
					aNGNumber += 1

		return aNGNumber


	# ----------------------------------------------------------
	def __isReadableFileOrNone(self,aKey):

		#print "----or-not---"
		#print self.__get(aKey) 

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None or \
		   self.__get(aKey) == "" or self.__get(aKey) == "None":

			self[aKey] = None
			return SET_UP_OK_RETURN

		else:
			return self.__isReadableFile(aKey)


	# ----------------------------------------------------------
	def __isReadableFile(self,aKey):
		'''Check the file can be read or not
		aKey      -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		#aFile = os.path.abspath( self.__get(aKey) )
		aFile = self.__get(aKey) 

		# -----------------------------------------------
		# checks the existance of aFile
		# -----------------------------------------------
		if os.path.isfile(aFile) == False:
			aMessage = "%s: \"%s\" is set as \"%s\", but there is not such file. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		# -----------------------------------------------
		# checks the read permission of aFile
		# -----------------------------------------------
		if os.access( aFile, os.R_OK ) == False:
			aMessage = "%s: \"%s\" is set as \"%s\", but it has no read permission. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		self[aKey] = aFile 
		return SET_UP_OK_RETURN


	# ----------------------------------------------------------
	def __isWritableFile(self,aKey):
		'''Check the file can be write or not
		aKey      -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		aFile = os.path.abspath( self.__get(aKey) )
		#aFile = self.__get(aKey) 

		# checks dir
		# to do

		if os.path.isfile(aFile) == False:

			aParentDir = os.path.dirname(aFile)
			if os.access( aParentDir, os.R_OK ) == False:
				aMessage = "%s: \"%s\" is set as \"%s\", but it has no write permission of parrent directory. \n" \
		           %(SETUP_ERROR,aKey,aFile)
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN

		else:

			if os.access( aFile, os.R_OK ) == False:
				aMessage = "%s: \"%s\" is set as \"%s\", but it has no write permission of it. \n" \
		           %(SETUP_ERROR,aKey,aFile)
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN


		self[aKey] = aFile
		return SET_UP_OK_RETURN


	# ----------------------------------------------------------
	def __isCorrectPermissionFile(self,aKey,aPermission):
		'''Check the permission of the file 
		aKey        -- a key of value (str)
		aPermission -- a Permission (os.R_OK/os.W_OK/os.X_OK)

		Return int : the number of error
		'''

		aFile = self.__get(aKey)

		# -----------------------------------------------
		# checks type
		# -----------------------------------------------
		if self.__isStr(aKey) == SET_UP_NG_RETURN:
			return SET_UP_NG_RETURN

		if aPermission == os.R_OK:
			self[aKey] = aFile
			return SET_UP_OK_RETURN

		# -----------------------------------------------
		# checks the existance of aFile
		# -----------------------------------------------
		if os.path.isfile(aFile) == False:
			aMessage = "%s: \"%s\" is set as \"%s\", but there is not such file. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		# -----------------------------------------------
		# checks the read permission of aFile
		# -----------------------------------------------
		if os.access( aFile, aPermission ) == False:
			aMessage = "%s: \"%s\" is set as \"%s\", but it has no %s permission. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey),PERMISSION[str(aPermission)] )
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		#print aFile

		self[aKey] = aFile
		return SET_UP_OK_RETURN

	# ----------------------------------------------------------
	def __checkParameterFormat(self):
		'''Check the format of paremeters

		Return int : the number of error
		'''

		try:
			aParameterList = self['PARAMETERS']
		except KeyError:
			aMessage = "%s: PARAMETERS are not set. \n" %(SETUP_ERROR)
			sys.stderr.write(aMessage)
			return SET_UP_OK_RETURN

		aConvertedParameterList = []
		aNGNumber = 0

		#print aParameterList

		for aParameter in aParameterList:
			# example.
			# Process:/E:KmS 0.1 10.0 float
			aMatch = re.match('(\S+)\s+(\S+)\s+(\S+)\s+(\S+)',aParameter)
			if aMatch == None:
				aNGNumber += 1
			else:
				aFullID = aMatch.group(1)        # ex. aFullID = "/CELL/CYTOPLASM/R:Km"
				aList = re.split(':',aFullID)    # ex. aList = ['/CELL/CYTOPLASM/R','Km']

				# check the number of ':'. It must be 3.
				#if len(aList) != 3:
				#	aMessage = "%s: About \"PARAMETER\", FullPN is set as \"%s, " %(SETUP_ERROR,aFullID)
				#	aMessage += "but its format is wrong. There must be two \':\' in the FullPN.\n" 
				#	sys.stderr.write(aMessage)
				#	aNGNumber += 1

				aType = aMatch.group(4)
				if ( aType == 'int' or aType == 'float' ) == False:
					aMessage = "%s: About \"PARAMETER\", Type is set as \"%s\", " \
					           %(SETUP_ERROR,aType)
					aMessage += "but it must be \"int\" or \"float\". \n" 
					sys.stderr.write(aMessage)
					aNGNumber += 1
					return aNGNumber

				# -------------------------------------
				# int
				# -------------------------------------
				if aType == 'int':
					aMin = aMatch.group(2)
					try:
						aMin = atoi(aMin)
					except ValueError:
						aMessage = "%s: About \"PARAMETER\", Minmum number is set as \"%s\", " \
						           %(SETUP_ERROR,aMin)
						aMessage += "but it must be int. \n" 
						sys.stderr.write(aMessage)
						aNGNumber += 1

					aMax = aMatch.group(3)
					try:
						aMax = atoi(aMax)
					except ValueError:
						aMessage = "%s: About \"PARAMETER\", Maximum number is set as \"%s\", " \
						           %(SETUP_ERROR,aMin)
						aMessage += "but it must be int. \n" 
						sys.stderr.write(aMessage)
						aNGNumber += 1

				# -------------------------------------
				# float
				# -------------------------------------
				else:

					aMin = aMatch.group(2)
					try:
						aMin = atof(aMin)
					except ValueError:
						aMessage = "%s: About \"PARAMETER\", Minmum number is set as \"%s\", " \
						           %(SETUP_ERROR,aMin)
						aMessage += "but it must be float. \n" 
						sys.stderr.write(aMessage)
						aNGNumber += 1

					aMax = aMatch.group(3)
					try:
						aMax = atof(aMax)
					except ValueError:
						aMessage = "%s: About \"PARAMETER\", Maximum number is set as \"%s\", " \
						           %(SETUP_ERROR,aMin)
						aMessage += "but it must be float. \n" 
						sys.stderr.write(aMessage)
						aNGNumber += 1

				# -------------------------------------
				# check relation of aMin and aMax
				# -------------------------------------

				if (aMin <= aMax) == False: 
					aMessage = "%s: About \"PARAMETER\", Minimum number and Maximum number are set as \"%s\" and \"%s\", " \
					           %(SETUP_ERROR,aMin,aMax)
					aMessage += "but Minumum number must be <= Maximum number \n" 
					sys.stderr.write(aMessage)
					aNGNumber += 1

				aConvertedParameterList.append((aFullID,aMin,aMax,aType))

		self['PARAMETER'] = aConvertedParameterList

		#print self['PARAMETER'] 

		if aNGNumber != 0:
			return aNGNumber
		else:
			return SET_UP_OK_RETURN


	# ----------------------------------------------------------
	def __checkReadablePermissionOfDirList(self,aKey):
		'''Check the directories can be read or not
		aKey        -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 


		aDirList = self[aKey]
		aNGNumber = 0


		for anIndex in xrange(0,len(aDirList)):

			#aDirList[anIndex] = os.path.abspath(aDirList[anIndex])

			# -----------------------------------------------
			# checks the existance of directory
			# -----------------------------------------------
			if os.path.isdir(aDirList[anIndex]) == False:
				aMessage = "%s: One of \"%s\" is set as \"%s\", but there is no such directory. \n" \
			                   %(SETUP_ERROR,
			                     aKey,
				                 os.path.basename(aDirList[anIndex][:-len(os.sep)]))
				sys.stderr.write(aMessage)
				aNGNumber += SET_UP_NG_RETURN

			else:

				# -----------------------------------------------
				# checks the permission of directory
				# -----------------------------------------------
				if os.access( aDirList[anIndex], os.R_OK + os.X_OK ) == False:
					aMessage = "%s: \"%s\" is set as \"%s\", but it has no read and execute permission of the directory. \n" \
			   	        %(SETUP_ERROR,
					      aKey,
				          os.path.basename(aDirList[anIndex][:-len(os.sep)]) )
					sys.stderr.write(aMessage)
					aNGNumber += SET_UP_NG_RETURN
	

		if aNGNumber != 0:
			return aNGNumber
		else:
			return SET_UP_OK_RETURN


	# ----------------------------------------------------------
	def __isReadableDir(self,aKey):
		'''Check the directoriy can be read or not
		aKey        -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 
	
		aDir = self.__get(aKey)

		# -----------------------------------------------
		# checks type
		# -----------------------------------------------
		if self.__isStr(aKey) == SET_UP_NG_RETURN:
			return SET_UP_NG_RETURN

		# -----------------------------------------------
		# checks the existance of directory
		# -----------------------------------------------
		if os.path.isdir(aDir) == False:
			aMessage = "%s: \"%s\" is set as \"%s\", but there is no parrent directory. \n" \
		                   %(SETUP_ERROR,aKey,self.__get(aKey))
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		# -----------------------------------------------
		# checks the permission of directory
		# -----------------------------------------------
		if os.access( aDir, os.R_OK + os.X_OK ) == False:
			aMessage = "%s: \"%s\" is set as \"%s\", but it has no read and execut permission of parrent directory of it. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey) )
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		return SET_UP_OK_RETURN


	# ----------------------------------------------------------
	def __isWritableDir(self,aKey):
		'''Check the directoriy can be wrote or not
		aKey        -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 
	
		#aDir = os.path.abspath( self.__get(aKey) )
		aDir = self.__get(aKey) 

		# -----------------------------------------------
		# checks type
		# -----------------------------------------------
		if self.__isStr(aKey) == SET_UP_NG_RETURN:
			return SET_UP_NG_RETURN

		if os.path.isabs( aDir ):

			aParentDir = os.path.dirname(aDir)

			# -----------------------------------------------
			# checks the existance of parent directory
			# -----------------------------------------------
			if os.path.isdir(aParentDir) == False:
				aMessage = "%s: %s is set as \"%s\", but there is no parrent directory. \n" \
		   	                %(SETUP_ERROR,aKey,self.__get(aKey))
				sys.stderr.write(aMessage)
				return SET_UP_NG_RETURN

		aDir = os.path.abspath(aDir)
		aParentDir = os.path.dirname(aDir)

		# -----------------------------------------------
		# checks the read permission of aParentDir
		# -----------------------------------------------
		if os.access( aParentDir, os.R_OK + os.W_OK + os.X_OK ) == False:
			aMessage = "%s: %s is set as \"%s\", but it has no read and write and execut permission of parrent directory of it. \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey) )
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		# -----------------------------------------------
		# rm old directory
		# -----------------------------------------------
		anOldDir = aDir + ".old"
		try:
			if os.path.isdir(aDir) == True and os.path.isdir(anOldDir) == True:
				shutil.rmtree(anOldDir)
		except:
			aMessage = "%s: %s is set as \"%s\". tried to the directory to backup directory, but it could not be removed \"%s.old\". \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey),self.__get(aKey) )
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		# -----------------------------------------------
		# mv directory to old directory
		# -----------------------------------------------
		try:
			if os.path.isdir(aDir) == True:
				shutil.copytree(aDir,anOldDir)    # Note:(1)
				shutil.rmtree(aDir)               # Note:(2)
				# Those two lines (1) and (2) should be changed shutil.move()
				# using Python 2.3 or later
		except:
			aMessage = "%s: %s is set as \"%s\". tried to the directory to backup directory, but it could not be moved to \"%s.old\". \n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey),self.__get(aKey) )
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN


		# -----------------------------------------------
		# create directory 
		# -----------------------------------------------
		try:
			os.mkdir(aDir)
		except:
			aMessage = "%s: %s is set as \"%s\", but it could not be created.\n" \
		           %(SETUP_ERROR,aKey,self.__get(aKey) )
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN

		return SET_UP_OK_RETURN


	# ----------------------------------------------------------
	def __isList(self,aKey):
		'''Check the type of value is list or not
		aKey        -- a key of value (str)

		Return int : the number of error
		'''

		# ------------------------------------------
		# check the existance
		# ------------------------------------------
		if aKey == None or self.__get(aKey) == None:
			aMessage = "%s: %s is not set. \n" \
			           %(SETUP_ERROR,aKey)
			sys.stderr.write(aMessage)
			return SET_UP_NG_RETURN 

		aLines = self.__get(aKey)

		if aLines == None:
			return SET_UP_OK_RETURN

		# construct list from multiple lines
		aLines = split(aLines,'\n')

		# delete space and #.*$ of each element 
		#for anIndex in xrange(0,len(aList)):
		aList = []
		for aLine in aLines:

			anIndex = find(aLine,'#')
			#print " aLine=%s, anIndex=%s" %(aLine,anIndex)
			if anIndex != -1:
				aLine = aLine[:anIndex]

			aLine = strip(aLine)

			if aLine != '':
				aList.append(aLine)

		#print aList

			
		# save it as a list
		if type(aList) != list:
			aList = [aList]

		self[aKey] = aList

		return SET_UP_OK_RETURN


# ###################################################################################

def main():

	try:
		aSettingFile = SETTING
	except NameError:
		aSettingFile = 'setting.txt'

	# -----------------------------------------------------
	# Read setting file
	# -----------------------------------------------------
	aSetting = Setting()
	aSetting.read(aSettingFile)


	# -----------------------------------------------------
	# Set up SessionManager
	# If you'd like to modify the property of SessionManager
	# modify the following lines.
	# -----------------------------------------------------

	# set environment
	if aSetting['USE SGE'] == True:
		setEnvironment('SGE')
	else:
		setEnvironment('Local')

	# set concurrency
	setConcurrency( aSetting['MAX CPU'] )

	# set tmp directory
	setTmpRootDir( aSetting['TMP DIRECTORY'] )

	# set update interval
	setUpdateInterval( aSetting['DURATION TIME'] )

	# set tmp directory removable
	setTmpDirRemovable( aSetting['TMPDIR REMOVABLE'] )

	# set retry max count
	setRetryMaxCount( aSetting['RETRY MAX COUNT'])
#	setRetryLimit( aSetting['RETRY MAX COUNT'])


	# -----------------------------------------------------
	# create GA instance and run it
	# -----------------------------------------------------

	# create instance
	anEstimator = RCGA(aSetting)

	# call initialize method at first
	anEstimator.initialize()

	# run main loop
	anEstimator.run()

	# display the result
	anEstimator.report()


if __name__ == '__builtin__':
	main()



