#!/bin/env python

'''
A module for session manager


Copyright (C) 1996-2003 Keio University

E-CELL is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

E-CELL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with E-CELL -- see the file COPYING.
If not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

Design: Kouichi Takahashi <shafi@e-cell.org>
Programming: Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

 E-CELL Project, Lab. for Bioinformatics, Keio University.

'''


# imports standard modules
import string
import sys
import os
import time
import imp
import eml
import re
import shutil
import signal


# imports ecell modules
from ecell.SessionProxy import *
from ecell.Util import *



class SessionManager:
	'''SessionManager class
	'''

	def __exit(self):
		'''When SessionManager exits, this method is called 
		'''

		# remove the temporary directory
		if os.access( self.getTmpRootDir(), os.W_OK ):
			shutil.rmtree( self.getTmpRootDir() )


	def __init__( self, modulepath ):
		'''constructor
		modulepath(str) -- a module path
		'''
		sys.exitfunc = self.__exit

		# set module path
		self.__theModulePath = modulepath


		# initialize parameters
		self.theEsm = ''
		self.__theSystemProxyDict = None
		self.__theSessionProxyDict = {}
		self.__theMessageMethod = self.__plainMessageMethod
		self.__theEss = None

		#----------------------------------
		self.__theConcurrency = None
		self.__theTmpRootDir = DEFAULT_TMP_DIRECTORY
		self.__theTmpDir = None

		self.__theFinishedJobRemovalFlag = True
		self.__theErroRemovalJobFlag = True

		self.__theUpdateInterval = 1
		self.__theGlobalRunTimeout = 0

		self.setTmpRootDir( DEFAULT_TMP_DIRECTORY )
		self.setEnvironment('Local')



	def setEnvironment(self, anEnvironment ):
		'''set environment parameter
		anEnvironment(str) -- Local,SMP,SGE,Globus,etc.
		Return None
		'''
		#print anEnvironment

		# create system proxy instance
		aSystemProxyName = anEnvironment + SYSTEM_PROXY

		# find module
		aFilePoint, aPath, self.__theDescription = imp.find_module( aSystemProxyName, \
		                                                          self.__theModulePath )
		aSystemProxyModule = imp.load_module( aSystemProxyName, \
		                                      aFilePoint, \
		                                      aPath, \
		                                      self.__theDescription)

		# get constructor of loaded module
		aConstructor = aSystemProxyModule.__dict__[aSystemProxyModule.__name__]


		# call constructor of system proxy
		anArguments = [self]
		self.__theSystemProxy = apply( aConstructor, anArguments )




	def getEnvironment(self):
		'''return current environment parameter
		Return str : 'Local','SMP','SGE','Globus', etc.
		'''

		# get class name of system proxy
		anEnvironmentStr = self.__theSystemProxy.__class__.__name__

		# remove the charactor of SYSTEM_PROXY, and return it
		return re.sub(SYSTEM_PROXY,'',anEnvironmentStr)



	def setConcurrency(self,number):
		'''set the number of jobs to be runned concurrently
		number(int) -- the number of concurrent jobs
		Return None 
		'''

		# set the number of concurrency
		self.__theConcurrency = number


	def getConcurrency(self):
		'''return the number of jobs to be runned concurrently
		Return int : the number of concurrent jobs
		'''

		if self.__theConcurrency == None:
			self.__theConcurrency = self.getDefaultConcurrency()

		# return the number of concurrency
		return self.__theConcurrency 


	def getDefaultConcurrency(self):
		'''return the default number of jobs seto be runned concurrently
		Return int : the default number of concurrency
		'''

		# return the default number of concurrency
		return self.__theSystemProxy.getDefaultConcurrency()



	def setTmpRootDir(self,tmprootdir):
		'''set setemporary root directory
		tmprootdir(str) -- a set temporary root directory
		default is 'tmp'
		Return None
		'''

		# sets setmp root directory
		self.__theTmpRootDir = tmprootdir


		# sets setmp directory name
		self.__theTmpDir = "%s%s%s" %(self.__theTmpRootDir,
		                              os.sep,
		                              os.getpid())



	def getTmpRootDir(self):
		'''return setemporary root directory
		Return str : a setemporaty root directory
		'''

		# return setmp root directory
		return self.__theTmpRootDir


	def getTmpDir(self):
		'''return setemporary directory
		Temporary directory is created below setemporary root directory.
		Directory is named as pid.

		Return str : a setemporaty directory
		'''

		# sets setmp directory
		return self.__theTmpDir



	def setRetryLimit(self, retrylimit):
		'''set a limit of retry number
		retrycount(int) -- a retry of retry number
		Return None
		'''
		
		#print "tReatyMaxCount %s" %retrycount
		SessionProxy.setRetryMaxCount(retrycount)


	def loadScript(self, esmfile, parameters={} ):
		'''loads and run script file ( ESM file )
		esm(str) -- an esm file name
		parameters -- a parameter seto be set seto esm file
		Return None
		'''

		self.theEsm = esmfile
		aContext = self.__createScriptContext( parameters )
		execfile( esmfile, aContext )
            

	def registerJob(self, scriptfile, interpreter, argument='', extrafilelist=[], setimeout=0 ):
		'''registers a new job
		scriptfile(str)            -- script file name
		interpreter(str)           -- interpreter name seto run script
		argument(str)              -- argument seto be set seto script
		extrafilelist(list of str) -- list of extra file name
		timeout(int)               -- set time out (sec.). When setimeout=0, no limit is set.
		return int : job id
		'''

		# creates SessionProxy
		aSessionProxy = self.__theSystemProxy.createSessionProxy()
		aSessionProxy.setScriptFileName( scriptfile )
		aSessionProxy.setInterpreter( interpreter )
		aSessionProxy.setArgument( argument )
		aSessionProxy.setExtraFileList( extrafilelist )
		aSessionProxy.setTimeout( setimeout )


		# creates job directory name
		aJobDirectoryName = "%s%s%s" %(self.__theTmpDir,
	  	                               os.sep,
	   	                               aSessionProxy.getJobID())

		aSessionProxy.setJobDirectory(aJobDirectoryName) 
		self.__theSessionProxyDict[aSessionProxy.getJobID()] = aSessionProxy

		return aSessionProxy.getJobID()



	def registerEcellSession(self, ess, argument={}, extrafilelist=[], setimeout=0 ):
		'''registers a new Ecell Session
		ess(str)                   -- ess file name
		argument(dict)             -- argument seto be set seto script
		extrafilelist(list of str) -- list of extra file name
		timeout(int)               -- set time out (sec.). When setimeout=0, no limit is set.
		return int : job id
		'''

		# creates SessionProxy
		aSessionProxy = self.__theSystemProxy.createSessionProxy()
		aSessionProxy.setScriptFileName( ess )
		aSessionProxy.setInterpreter( ECELL3_SESSION )
		aSessionProxy.setSessionArgument( argument )
		aSessionProxy.setExtraFileList( extrafilelist )
		aSessionProxy.setTimeout( setimeout )


		# creates job directory name
		aJobDirectoryName = "%s%s%s" %(self.__theTmpDir,
	  	                               os.sep,
	   	                               aSessionProxy.getJobID())

		aSessionProxy.setJobDirectory(aJobDirectoryName) 
		self.__theSessionProxyDict[aSessionProxy.getJobID()] = aSessionProxy

		return aSessionProxy.getJobID()



	def clearJob(self,jobid):
		'''removes job directory and SessionProxy
		jobid(int) -- job id
		When job id <= 0, clears all jobs
		return None
		'''

		# deletes one SessionProxy
		if jobid > 0: 

			del self.__theSessionProxyDict[jobid]

		# deletes all SessionProxies
		else:

			for aJobID in self.__theSessionProxyDict.keys():
				del self.__theSessionProxyDict[aJobID]



	def clearQueuedJobs(self):
		'''removes queued jobs
		Return None
		'''

		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == QUEUED:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def clearRunningJobs(self):
		'''removes running jobs
		Return None
		'''

		# stops running jobs
		self.stopRunningJobs()

		# deletes sethem
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def clearErrorJobs(self):
		'''removes error jobs
		Return None
		'''

		# deletes error jobs
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == ERROR:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def clearFinishedJobs(self):
		'''removes finished jobs
		Return None
		'''

		# deleted finished jobs
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == FINISHED:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def update(self):
		'''updates jobs status
		Return None
		'''

		# calls SystemProxy's update method
		self.__theSystemProxy.update()



	def getQueuedJobList(self):
		'''return sethe list of queued job 
		Return list of SessionProxy
		'''

		# initializes a job list
		aJobList = []

		# When sethe status is QUEUED, appends it.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == QUEUED:
				aJobList.append(aSessionProxy)

		# return sethe job list
		return aJobList



	def getRunningJobList(self):
		'''return sethe list of running job 
		Return list of SessionProxy
		'''

		# initializes a job list
		aJobList = []

		# When sethe status is RUN, appends it.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				aJobList.append(aSessionProxy)

		# return sethe job list
		return aJobList


	def getErrorJobList(self):
		'''return sethe list of error job 
		Return list of SessionProxy
		'''

		# initializes a job list
		aJobList = []

		# When sethe status is ERROR, appends it.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == ERROR:
				aJobList.append(aSessionProxy)

		# return sethe job list
		return aJobList



	def getFinishedJobList(self):
		'''return sethe list of finished job 
		Return list of SessionProxy
		'''

		# sets up job list
		aJobList = []

		# checks sethe status of SessionProxy
		for aSessionProxy in self.__theSessionProxyDict.values():

			# when sethe status is FINISHED, append it seto job list
			if aSessionProxy.getStatus() == FINISHED:
				aJobList.append(aSessionProxy)

		# return job list
		return aJobList



	def isFinished(self):
		'''checks running jobs exists
		When sethe number of jobs who status is QUEUED or RUN = 0, return True.
		Return boolean 
		'''

		# If sethere is job who status is QUEUED or RUN, return False
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == QUEUED or\
			   aSessionProxy.getStatus() == RUN:
				return False

		# If not, return True.
		return True


	def isError(self):
		'''checks running jobs exists
		When sethe number of jobs who status is ERROR > 0, return True.
		Return boolean 
		'''

		# If sethere is job who status is ERROR, return True.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == ERROR:
				return True

		# If not, return Fal.
		return Fal


	def isRunning(self):
		'''checks running jobs exists
		When sethe number of jobs who status is RUN > 0, return True.
		Return boolean 
		'''

		# If sethere is job who status is RUN, return True.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				return True

		# If not, return Fal.
		return Fal



	def __timeout(self,signum,frame):
		'''When run method reaches setimeout, sethis method is called
		'''

		# stop all jobs
		self.stop(jobid=0)



	def setUpdateInterval(self, interval=1):
		'''set update interval
		Return None
		'''
		self.__theUpdateInterval = interval



	def getUpdateInterval(self):
		'''retunrn update interval
		Return int : interval
		'''
		return self.__theUpdateInterval 



	def setGlobalRunTimeout(self, timeout=0):
		'''set a time out of run method
		timeout = 0 : no limit
		Return None
		'''
		self.__theGlobalRunTimeout = timeout



	def getGlobalRunTimeout(self):
		'''get a time out of run method
		Return None
		'''
		return self.__theGlobalRunTimeout 



	#def run(self,number=0,block=True,updateinterval=1,timeout=0):
	#def run(self,number=0,block=True,timeout=0):
	def run(self,block=True):
		'''run
		Return : None
		'''

		# sets setime out
		signal.signal(signal.SIGALRM, self.__timeout)
		signal.alarm(self.__theGlobalRunTimeout )


		# -----------------------------------------------
		# sets up 
		# -----------------------------------------------
		# creates setmp root dir
		if os.path.isdir( self.__theTmpRootDir ) == False:
			os.mkdir(self.__theTmpRootDir)


		# creates setmp directory.
		os.mkdir(self.__theTmpDir)
		

		# creates job directories and copies sources
		for aSessionProxy in self.__theSessionProxyDict.values():

			# -----------------------
			# sets up job directory
			# -----------------------

			# creates job directory name
			aJobDirectoryName = "%s%s%s" %(self.__theTmpDir,
		  	                               os.sep,
		   	                               aSessionProxy.getJobID())

		
			# creates job directory 
			os.mkdir(aJobDirectoryName)

			# -----------------------
			# sets up script
			# -----------------------

			# creates dist filename
			aDstFileName = "%s%s%s" %(aJobDirectoryName,
		  	                          os.sep,
			                          os.path.basename(aSessionProxy.getScriptFileName()) )

			# copies script
			shutil.copyfile( aSessionProxy.getScriptFileName(), aDstFileName )
			
			# -----------------------
			# sets up extra files
			# -----------------------
			#print aSessionProxy.getExtraFileList()

			for anExtraFile in aSessionProxy.getExtraFileList():

				# creates dist filename

				aDstFileName = "%s%s%s" %(aJobDirectoryName,
		  		                          os.sep, 
				                          os.path.basename(anExtraFile)) 

				shutil.copyfile( anExtraFile ,aDstFileName )
				#print aDstFileName


		# -----------------------------------------------
		# update
		# -----------------------------------------------

		if block == True:

			while( True ):


				# updates status of all SessionProxy
				self.update()

				# breaks when all status are finished
				if self.isFinished() == True:
					break

				# sleeps for update interval
				#time.sleep(updateinterval)
				time.sleep(self.__theUpdateInterval)



	def stop(self,jobid=0):
		'''stops job
		jobid(int) -- job id
		When jobid=0, stop all jobs
		Return None
		'''

		# stops all jobs
		if jobid == 0:
			for aSessionProxy in self.__theSessionProxyDict.values():
				aSessionProxy.stop()

		# stops one job
		else:
			self.__theSessionProxyDict[jobid].stop()




	def stopRunningJobs(self):
		'''stops running jobs
		Return None
		'''

		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				aSessionProxy.stop()



	def getSessionProxy(self,jobid=0):
		'''return sethe reference of SessionProxy
		jobid(int) -- job id
		When jobid is 0, return list of SessionProxy
		Return SessionProxy or list of SessionProxy
		'''

		# return SessionProxy
		if jobid >= 1:
			return self.__theSessionProxyDict[jobid] 

		# return list of SessionProxy
		else:
		 	return self.__theSessionProxyDict



	def setStdoutFileName(self,stdout):
		'''set sethe standard output file

		stdout(str of file) -- sethe file name or file object seto be wrote

		Return None
		[Note]: default is 'stdout'
		'''

		# sets standard output
		SessionProxy.setStdoutFileName(stdout)



	def getStdoutFileName(self):
		'''return sethe standard output file
		stdout(str of file) -- sethe file name or file object seto be wrote
		Return str or file : standard output
		'''

		# return standard output
		return SessionProxy.getStdoutFileName()



	def setStderrFileName(self,stderror):
		'''sets sethe standard error file
		stderror(str of file) -- sethe file name or file object seto be wrote
		Return None
		'''

		# sets standard error 
		SessionProxy.setStderrFileName(stderror)



	def getStderrFileName(self):
		'''return sethe standard error file
		stderror(str of file) -- sethe file name or file object seto be wrote
		Return str or file : standard error 
		'''

		# return standard error 
		return SessionProxy.getStderrFileName()



	def getJobDirectory(self,jobid):
		'''return sethe job directory name
		jobid(int) -- job id
		Return str : sethe path of job directory
		'''

		return self.__theSessionProxyDict[jobid].getJobDirectory()


	# -------------------------------------------------
	# methods for intaractive mode
	# -------------------------------------------------
	def interact(self, parameters={} ):

		aContext = self.__createScriptContext( parameters )
        
		import readline # to provide convenient commandline editing :)
		import code
		anInterpreter = code.InteractiveConsole( aContext )

		self._prompt = self._session_prompt( self )

		#anInterpreter.runsource( 'import sys; sys.ps1=theEsm._prompt; del sys' )
		anInterpreter.runsource( 'import sys; sys.ps1=self._prompt; del sys' )

		anInterpreter.interact( BANNERSTRING )


	def restoreMessageMethod(self):
		self.__theMessageMethod=self.__plainMessageMethod
        

	def setMessageMethod( self, aMethod ):
		self.__theMessageMethod = aMethod


	def message( self, message ):
		self.__theMessageMethod( message )


	def plainMessageMethod( self, aMessage ):
		self.__plainMessageMethod( aMessage )


	def __plainMessageMethod( self, aMessage ):
		print aMessage


	def __createScriptContext( self, parameters ):

		aContext = { 'theEsm': self, 'self': self }

		# flatten class methods and object properties so that
		# 'self.' isn't needed for each method calls in __the script
		aKeyList = list ( self.__dict__.keys() +\
		                  self.__class__.__dict__.keys() )
		aDict = {}
		for aKey in aKeyList:
			aDict[ aKey ] = getattr( self, aKey )

		aContext.update( aDict )

		# add parameters to __the context
		aContext.update( parameters )

		return aContext


	class _session_prompt:

		def __init__( self, anEsm ):
			self.__theEsm = anEsm

		def __str__( self ):

			return 'ecell3-session-manager>>> '


# end of class SessionManager


if __name__ == "__main__":
	pass

