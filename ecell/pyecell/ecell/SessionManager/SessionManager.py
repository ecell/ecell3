#!/bin/env python

'''
A module for session manager


Copyright (C) 2001-2004 Keio University

E-Cell is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

E-Cell is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with E-Cell -- see the file COPYING.
If not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.

Design: Kouichi Takahashi <shafi@e-cell.org>
Programming: Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

 E-Cell Project, Lab. for Bioinformatics, Keio University.

'''


# imports standard modules
import string
import sys
import os
import time
import imp
import ecell.eml
import re
import shutil
import signal


# imports ecell modules
from SessionProxy import *
from Util import *


class SessionManager:
	'''SessionManager class
	Provide API to execute multiple jobs concurrently.
	You can access SessionProxy directory to control each job.
	'''

	def __exit( self ):
		'''When SessionManager exits, this method is called.
		In this method, delete tmp root directory.
		Running jobs will be deleted from each SessionProxy instance.
		This method should be substituded to sys.exitfunc.
		'''

		if self.__theTmpRemovable == True:
			# remove the temporary directory
			if os.access( self.getTmpRootDir(), os.W_OK ):
				shutil.rmtree( self.getTmpRootDir() )

	
		self.__theSessionProxyDict = None
		self.__theSystemProxy = None


	def __init__( self, modulepath, concurrency, environment ):
		'''Constructor
		Initialize parameters
		modulepath(str) -- a module path
		concurrency(int) -- a concurrency
		environment(str) -- an environment to be used
		'''

		# exit functions
		sys.exitfunc = self.__exit

		# set module path
		self.__theModulePath = modulepath

		# initialize parameters
		self.theEsm = ''
		self.__theSystemProxyDict = None
		self.__theSessionProxyDict = {}
		self.__theMessageMethod = self.__plainMessageMethod

		self.__theConcurrency = concurrency
		self.__theTmpRootDir = DEFAULT_TMP_DIRECTORY
		self.__theTmpDir = None
		self.__theFinishedJobRemovableFlag = True
		self.__theErroRemovableJobFlag = True
		self.__theUpdateInterval = 1
		self.__theGlobalRunTimeout = 0

		self.__theTmpRemovable = True

		# set default temporary directory
		# Temporary directory is created in run method.
		self.setTmpRootDir( DEFAULT_TMP_DIRECTORY )

		# let default environment
		self.setEnvironment( environment )


	def setEnvironment( self, environment ):
		'''Set environment parameter.
		Try to import the module whole name is environment + SystemProxy
		For example, if the 'environment' is Local, try to LocalSystemProxy
		and call constroctor of it.

		environment(str) -- Local,SMP,SGE,Globus,etc.
		Return None
		'''

		# create system proxy instance
		aSystemProxyName = environment + SYSTEM_PROXY

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



	def getEnvironment( self ):
		'''Return current environment parameter as str.

		Return str : 'Local','SMP','SGE','Globus', etc.
		'''

		# get class name of system proxy
		anEnvironmentStr = self.__theSystemProxy.__class__.__name__

		# remove the charactor of SYSTEM_PROXY, and return it
		return re.sub(SYSTEM_PROXY,'',anEnvironmentStr)



	def setConcurrency( self, number ):
		'''Set the number of jobs to be runned concurrently.

		number(int) -- the number of concurrent jobs
		Return None 
		'''

		# set the number of concurrency
		self.__theConcurrency = number


	def getConcurrency( self ):
		'''Return the number of jobs to be runned concurrently.

		Return int : the number of concurrent jobs
		'''

		if self.__theConcurrency == None:
			self.__theConcurrency = self.getDefaultConcurrency()

		# return the number of concurrency
		return self.__theConcurrency 


	def getDefaultConcurrency( self ):
		'''Return the default number of jobs to be runned concurrently.

		Return int : the default number of concurrency
		'''

		# return the default number of concurrency
		return self.__theSystemProxy.getDefaultConcurrency()

	def setTmpRootDir( self, tmprootdir ):
		'''Set temporary root directory.
		When run method is called, tmprootdir is created on current working 
		directory.  Then, below 'tmp' directory, the directory whose name 
		is same as pid of this process is also created.
		For example, if tmprootdir is 'work' and pid is 12345, './work/12345'
		is created whtn run method is called.

		tmprootdir(str) -- a temporary root directory to be set. default is 'tmp'
		Return None
		'''

		# sets tmp root directory
		self.__theTmpRootDir = tmprootdir


		# sets tmp directory name
		self.__theTmpDir = "%s%s%s" %(self.__theTmpRootDir,
		                              os.sep,
		                              os.getpid())


	def getTmpRootDir( self ):
		'''Return the temporary root directory
		See setTmpRootDir about the detail of temporary root directory

		Return str : temporaty root directory
		'''

		# return tmp root directory
		return self.__theTmpRootDir



	def getTmpDir( self ):
		'''Return temporary directory
		Temporary directory is created below temporary root directory,
		and named pid of this process.
		See setTmpRootDir about the detail of temporary directory
		For example, if tmprootdir is 'work' and pid is 12345, 
		this method returns './work/12345'.
		
		Return str : a temporaty directory
		'''

		# return temporary directory
		return self.__theTmpDir

	
	def setTmpDirRemovable( self, deleteflag ):
		'''Set a removable flag of tmp directory.

		deleteflag(boolean)  --  True  : delete tmp and tmp root directory,
                                         when destructor is called.
		                         False : tmp and tmp root directory is not
		                                 deleted when destructor is called.
		                         True is set as default.
		Return None
		'''

		# save deleteflat to instance attribute
		self.__theTmpRemovable = deleteflag


	def getTmpDirRemovable( self ):
		'''Return the removable flag of tmp directory.

		Return boolean : True  : delete tmp and tmp root directory,
                                 when destructor is called.
		                 False : tmp and tmp root directory is not
		                         deleted when destructor is called.
		'''

		# return delete flag of tmp directory
		return self.__theTmpRemovable 


	def setRetryLimit( self, retrylimit ):
		'''Set a limit of retry number

		retrylimit(int) -- a retry of retry number
		Return None
		'''
		
		# call a class method of SessionProxy.
		SessionProxy.setRetryMaxCount(retrylimit)


	def loadScript(self, esmfile, parameters={} ):
		'''Load script file ( ESM file )

		esm(str) -- an esm file name
		parameters(dict) -- a parameter to be set to esm file
		Return None
		'''

		# save an esm file name to an instance attribute.
		self.theEsm = esmfile

		# create a context
		aContext = self.__createScriptContext( parameters )

		# execute esm file
		execfile( esmfile, aContext )
            

	def registerJob(self, scriptfile, interpreter, argument='', extrafilelist=[], timeout=0 ):
		'''registers a new job
		scriptfile(str)            -- script file name
		interpreter(str)           -- interpreter name to run script
		argument(str)              -- argument set be set to script
		extrafilelist(list of str) -- list of extra file name
		timeout(int)               -- set time out (sec.). When timeout<=0, no limit is set.
		return int : job id
		'''

		# creates SessionProxy
		aSessionProxy = self.__theSystemProxy.createSessionProxy()
		aSessionProxy.setScriptFileName( scriptfile )
		aSessionProxy.setInterpreter( interpreter )
		aSessionProxy.setArgument( argument )
		if type(extrafilelist) != list:
			extrafilelist = [extrafilelist]
		aSessionProxy.setExtraFileList( extrafilelist )
		aSessionProxy.setTimeout( timeout )


		# creates job directory name
		aJobDirectoryName = "%s%s%s" %(self.__theTmpDir,
	  	                               os.sep,
	   	                               aSessionProxy.getJobID())

		aSessionProxy.setJobDirectory(aJobDirectoryName) 
		self.__theSessionProxyDict[aSessionProxy.getJobID()] = aSessionProxy

		return aSessionProxy.getJobID()



	def registerEcellSession(self, ess, argument={}, extrafilelist=[], dmpath="", timeout=0):
		'''registers a new Ecell Session
		ess(str)                   -- ess file name
		argument(dict)             -- argument to be set to script
		extrafilelist(list of str) -- list of extra file name
		dmpath(str)            -- set the ECELL3_DM_PATH
		timeout(int)               -- set time out (sec.). When timeout=0, no limit is set.
		return int : job id
		'''

		# creates SessionProxy
		aSessionProxy = self.__theSystemProxy.createSessionProxy()
		aSessionProxy.setScriptFileName( ess )
		aSessionProxy.setInterpreter( ECELL3_SESSION )
		aSessionProxy.setSessionArgument( argument )
		if type(extrafilelist) != list:
			extrafilelist = [extrafilelist]
		aSessionProxy.setExtraFileList( extrafilelist )
		aSessionProxy.setDM_PATH(dmpath)
		aSessionProxy.setTimeout( timeout )

		# creates job directory name
		aJobDirectoryName = "%s%s%s" %(self.__theTmpDir,
	  	                               os.sep,
	   	                               aSessionProxy.getJobID())

		aSessionProxy.setJobDirectory(aJobDirectoryName) 
		self.__theSessionProxyDict[aSessionProxy.getJobID()] = aSessionProxy

		return aSessionProxy.getJobID()



	def clearJob( self, jobid ):
		'''Remove the job directory and the SessionProxy related to the jobid.

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



	def clearQueuedJobs( self ):
		'''remove queued jobs

		Return None
		'''

		# delete SessionProxy instance whose status is QUEUED
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == QUEUED:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def clearRunningJobs( self ):
		'''remove running jobs

		Return None
		'''

		# stop running jobs
		self.stopRunningJobs()

		# delete SessionProxy instance whose status is RUN
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def clearErrorJobs( self ):
		'''remove error jobs

		Return None
		'''

		# delete SessionProxy instance whose status is ERROR
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == ERROR:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def clearFinishedJobs( self ):
		'''remove finished jobs

		Return None
		'''

		# delete SessionProxy instance whose status is FINISHED
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == FINISHED:
				del self.__theSessionProxyDict[aSessionProxy.getJobID()]



	def update( self ):
		'''updates all jobs' status
		This method calls merely SystemProxy's update method.

		Return None
		'''

		# call SystemProxy's update method
		self.__theSystemProxy.update()



	def getQueuedJobList( self ):
		'''Return the list of queued job.

		Return list : the list of SessionProxy
		'''

		# initializes a job list
		aJobList = []

		# When the status is QUEUED, append it.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == QUEUED:
				aJobList.append(aSessionProxy)

		# return the job list
		return aJobList



	def getRunningJobList( self ):
		'''Return the list of running job. 

		Return list : the list of SessionProxy
		'''

		# initialize a job list
		aJobList = []

		# When the status is RUN, append it.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				aJobList.append(aSessionProxy)

		# return the job list
		return aJobList


	def getErrorJobList( self ):
		'''Return the list of error job 

		Return list : the list of SessionProxy
		'''

		# initialize a job list
		aJobList = []

		# When the status is ERROR, append it.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == ERROR:
				aJobList.append(aSessionProxy)

		# return the job list
		return aJobList



	def getFinishedJobList( self ):
		'''Return the list of finished job. 

		Return list : list of SessionProxy
		'''

		# initialize a job list
		aJobList = []

		# checks the status of SessionProxy
		for aSessionProxy in self.__theSessionProxyDict.values():

			# when the status is FINISHED, append it.
			if aSessionProxy.getStatus() == FINISHED:
				aJobList.append(aSessionProxy)

		# return job list
		return aJobList



	def isFinished( self ):
		'''Check all jobs are finished or not.
		When the number of jobs whose status is QUEUED or RUN = 0, return True.

		Return boolean 
		'''

		# If there is job who status is QUEUED or RUN, return False
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == QUEUED or\
			   aSessionProxy.getStatus() == RUN:
				return False

		# If not, return True.
		return True


	def isError( self ):
		'''Check the existance of error job.
		When the number of jobs whose status is ERROR > 0, return True.

		Return boolean 
		'''

		# If there is job who status is ERROR, return True.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == ERROR:
				return True

		# If not, return False.
		return False


	def isRunning( self ):
		'''Check the existance of running job.
		When the number of jobs who status is RUN > 0, return True.

		Return boolean 
		'''

		# If there is job who status is RUN, return True.
		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				return True

		# If not, return Fal.
		return False



	def __timeout( self, signum, frame ):
		'''When run method reaches timeout, this method is called.
		In this method, stop method is called.

		Return None
		'''

		# stop all jobs
		self.stop(jobid=0)



	def setUpdateInterval( self, interval=1 ):
		'''Set an update interval.
		When the argument block is True, update method of SystemProxy
		is called in the update interval time.
		When the argument block is False, the update interval time
		is not used.

		interval(int) -- an interval time (sec.). default is 1.
		Return None
		'''

		# save interval to instance attribute
		self.__theUpdateInterval = interval



	def getUpdateInterval( self ):
		'''Return the update interval.
		Update interval is used in run method.
		See setUpdateInterval about the detail of update interval

		Return int : interval time (sec.)
		'''

		# return the update interval
		return self.__theUpdateInterval 



	def setGlobalRunTimeout( self, timeout=0 ):
		'''Set a timeout of run method.
		When the time of run method reach the timeout,
		stop method is called. Then running jobs and queued jobs 
		are finished compulsorily and their status become ERROR.

		timeout(int) -- a timeout of run method (sec.) 
		                if timeout<=0, no limit is set. Default is 0.
		Return None
		'''

		# save timeout to instance attribute
		self.__theGlobalRunTimeout = timeout



	def getGlobalRunTimeout( self ):
		'''Return the timeout of run method.
		See setGlobalRunTimeout about the detail of timeout

		Return int : the timeout of run method (sec.)
		'''

		# return the timeout
		return self.__theGlobalRunTimeout 



	def run( self, block=True ):
		'''Execute the QUEUED jobs.
		The RUNNING or FINISHED jobs are not executed by this method.

		When block is True, this method waits until all jobs are finished.
		While some jobs are running, call update method at the intervals of
		updateinterval that is speficed setUpdateInterval.
		When the global run timeout > 0 is set through setGlobalRunTimeout
		and the running time of this method reaches it, stop method is
		called and this method is finished compulsorily.

		When block is False, you have to call update method by yourself.
		If update method does not be called, not only the status of all jobs
		are not changed but also queued jobs never be executed.

		block(boolean) -- True: This method waits until all jobs are finished.
		                  False: This method does not wait.
		Return : None
		'''

		# set timeout
		signal.signal(signal.SIGALRM, self.__timeout)
		signal.alarm(self.__theGlobalRunTimeout )


		# -----------------------------------------------
		# set up 
		# -----------------------------------------------
		# create a tmp root dir
		if os.path.isdir( self.__theTmpRootDir ) == False:
			os.mkdir(self.__theTmpRootDir)


		# create a tmp directory.
		if os.path.isdir(self.__theTmpDir) == False:
			os.mkdir(self.__theTmpDir)
		

		# create job directories and copies sources
		for aSessionProxy in self.__theSessionProxyDict.values():

			if aSessionProxy.getStatus() == FINISHED:
				continue

			# -----------------------
			# set up job directory
			# -----------------------
			# create job directory name
			aJobDirectoryName = "%s%s%s" %(self.__theTmpDir,
		  	                               os.sep,
		   	                               aSessionProxy.getJobID())

		
			# create job directory 
			if os.path.isdir(aJobDirectoryName) == False:
				os.mkdir(aJobDirectoryName)

			# -----------------------
			# set up script
			# -----------------------

			# create dist filename
			aDstFileName = "%s%s%s" %(aJobDirectoryName,
		  	                          os.sep,
			                          os.path.basename(aSessionProxy.getScriptFileName()) )

			# copie script
			shutil.copyfile( aSessionProxy.getScriptFileName(), aDstFileName )

			# -----------------------
			# sets up extra files
			# -----------------------
			for anExtraFile in aSessionProxy.getExtraFileList():

				# creates dist filename
				aDstFileName = "%s%s%s" %(aJobDirectoryName,
		  		                          os.sep, 
				                          os.path.basename(anExtraFile)) 

				# copy a directory
				if os.path.isdir( anExtraFile ) == True:
					shutil.copytree( anExtraFile ,aDstFileName )

				# copy a file
				else:
					shutil.copyfile( anExtraFile ,aDstFileName )



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
				time.sleep(self.__theUpdateInterval)



	def stop( self, jobid=0 ):
		'''stop jobs.
		Change the statuses of QUEUED or RUNNING jobs to ERROR.
		About FINISHED jobs, do nothing.

		jobid(int) -- job id
		              When jobid<=0, stop all jobs
		Return None
		'''

		# stops all jobs
		if jobid == 0:
			for aSessionProxy in self.__theSessionProxyDict.values():
				aSessionProxy.stop()

		# stops one job
		else:
			self.__theSessionProxyDict[jobid].stop()




	def stopRunningJobs( self ):
		'''Stop running jobs.
		Change all statuses of RUNNING jobs to ERROR.

		Return None
		'''

		for aSessionProxy in self.__theSessionProxyDict.values():
			if aSessionProxy.getStatus() == RUN:
				aSessionProxy.stop()



	def getSessionProxy( self, jobid=0 ):
		'''Return an instance of SessionProxy or dict of jobids.
		and instances of SessionProxy.

		jobid(int) -- job id
		              When jobid<=0, return dict of SessionProxy

		Return SessionProxy or dict of SessionProxy
		A key of the dict of SessionProxy is jobid(int).
		and a value of it is an instance of SessionProxy.
		'''

		# return SessionProxy
		if jobid >= 1:
			return self.__theSessionProxyDict[jobid] 

		# return list of SessionProxy
		else:
		 	return self.__theSessionProxyDict



	def setStdoutFileName( self, stdout ):
		'''Set the standard output file.

		stdout(str of file) -- the file name or file object 
		                       default is 'stdout'

		Return None
		'''

		# set standard output
		SessionProxy.setStdoutFileName(stdout)



	def getStdoutFileName( self ):
		'''Return the standard output file.

		stdout(str of file) -- the file name or file object.
		Return str or file : standard output
		'''

		# return standard output
		return SessionProxy.getStdoutFileName()



	def setStderrFileName( self, stderror ):
		'''Set the standard error file.

		stderr(str of file) -- the file name or file object
		                       default is 'stderr'
		Return None
		'''

		# set standard error 
		SessionProxy.setStderrFileName(stderror)



	def getStderrFileName( self ):
		'''Return the standard error file.

		stderror(str of file) -- the file name or file object
		Return str or file : standard error 
		'''

		# return standard error 
		return SessionProxy.getStderrFileName()


	def getJobDirectory( self, jobid ):
		'''Return the job directory name

		jobid(int) -- job id

		Return str : the path of the job directory
		'''

		# return the path of the job directory
		return self.__theSessionProxyDict[jobid].getJobDirectory()


	def setRetryMaxCount( self, limit ):

		SessionProxy.setRetryMaxCount(limit)


	def getStdout( self, jobid ):
		'''return the stdout of the job

		jobid(int) -- job id
		
		Return str : the stdout
		'''
		return self.__theSessionProxyDict[jobid].getStdout()


	def getStderr( self, jobid ):
		'''return the stderr of the job

		jobid(int) -- job id
		
		Return str : the stderr
		'''
		return self.__theSessionProxyDict[jobid].getStderr()


	def getSystemProxy( self ):

		return self.__theSystemProxy

	# -------------------------------------------------
	# methods for intaractive mode
	# -------------------------------------------------
	def interact(self, parameters={} ):

		aContext = self.__createScriptContext( parameters )
        
		try:
			import readline # to provide convenient commandline editing :)
		except:
			pass
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
			self.theEsm = anEsm

		def __str__( self ):

			return 'ecell3-session-manager>>> '


# end of class SessionManager


if __name__ == "__main__":
	pass

