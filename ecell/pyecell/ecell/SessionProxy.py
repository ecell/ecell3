#! /usr/bin/env python

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

import sys
import string
import os
import time
import popen2
import re


import ecell.eml
from Numeric import *
import ecell.emc
import ecell.ecs

from ecell.Util import *
from ecell.SessionManager import *

class SessionProxy:
	'''SessionProxy is an abstract class that manages one process
	'''

	# -----------------
	# class variables
	# -----------------

	__theJobID = 0
	__theStdout = DEFAULT_STDOUT
	__theStderror = DEFAULT_STDERR
	__theRetryMaxCount = 0


	# -----------------
	# class methods
	# -----------------

	def __getNextJobID(cls):
		'''returns next job ID 
		Return int : job ID
		[Note]:private method, the first job id is 1
		'''

		# counts up jod id
		SessionProxy.__theJobID += 1

		# returns job id
		return SessionProxy.__theJobID


	def setStdoutFileName(cls, stdout):
		'''sets standard output
		stdout(str or file) -- file name or file object to be wrote
		Return None
		[Note] default stdout is 'stdout'
		'''

		# sets stdout
		SessionProxy.__theStdout = stdout


	def getStdoutFileName(cls):
		'''returns standard output
		Return str or file -- standard output
		[Note] default stdout is 'stdout'
		'''

		# returns stdout
		return SessionProxy.__theStdout


	def setStderrFileName(cls, stderror):
		'''sets standard error
		stdout(str or file) -- file name or file object to be wrote
		Return None
		[Note] default stderr is 'stderr'
		'''

		SessionProxy.__theStderror = stderror


	def getStderrFileName(cls):
		'''returns standard error
		Return str or file -- standard error
		[Note] default stderr is 'stderr'
		'''

		return SessionProxy.__theStderror


	def setRetryMaxCount(cls, count):
		'''sets retry max count
		When count < 0, count is set as 0
		count(int) -- retry max count
		Return None
		[Note] default max count is 0
		'''

		if count < 0:
			count = 0

		SessionProxy.__theRetryMaxCount = count


	# registers class methods
	__getNextJobID = classmethod(__getNextJobID)
	setStdoutFileName = classmethod(setStdoutFileName)
	getStdoutFileName = classmethod(getStdoutFileName)
	setStderrFileName = classmethod(setStderrFileName)
	getStderrFileName = classmethod(getStderrFileName)
	setRetryMaxCount = classmethod(setRetryMaxCount)


	# -----------------
	# instance methods
	# -----------------

	def __init__(self):
		'''Constructor
		job id is set.
		status is set as QUEUED
		all attributes are initialized.
		'''

		# sets job id
		self.__theJobID = SessionProxy.__getNextJobID()


		# sets status as QUEUED
		self.__theStatus = QUEUED


		# initializes attributes
		self.__theTimeout = None
		self.__theScriptFileName = None
		self.__theInterpreter = None
		self.__theArgument = None
		self.__theExtraFileList = None
		self.__theRemovalStatus = False
		self.__theJobDirectory = None
		self.__theRetryCount = 0
		self.__theStartTime = 0



	def __del__(self):
		'''when destractor is called, deletes the job directory.
		Return None
		'''

		# removes job directory
		self.clearJobDirectory()

		# destractor is called automatically after this method



	def getJobID(self):
		'''returns job id
		Return int : job id
		''' 

		# returns the job id
		return self.__theJobID 



	def setScriptFileName(self,scriptfilename):
		'''sets script file name
		scriptfilename(str) -- a script file name
		Return None
		''' 

		# sets the script file name
		self.__theScriptFileName = scriptfilename



	def getScriptFileName(self):
		'''returns script file name
		Return str : script file name
		''' 

		# returns the script file name
		return self.__theScriptFileName 



	def setInterpreter(self,interpreter):
		'''sets interpreter
		interpreter(str) -- an interpreter
		Return None
		'''

		aStdout, aStdin, aStderr = popen2.popen3( "which %s" %interpreter )

		try:
			aPath = aStdout.readline()[:-1]
		except IOError:
			aPath = interpreter

		# sets the interpreter
		self.__theInterpreter = os.path.abspath( aPath )


	def getInterpreter(self):
		'''returns interpreter
		Return str : interpreter
		''' 

		# returns the interpreter
		return self.__theInterpreter 



	def setArgument(self,argument):
		'''sets argument
		argument(str) -- an argument to be set to script
		Return None
		''' 

		# sets the argument
		self.__theArgument = argument



	def getArgument(self):
		'''returns argument
		Return None
		''' 

		# returns the artument
		return self.__theArgument 



	def setSessionArgument(self,argument):
		'''sets argument to session
		argument(dict)  -- argument to be set to session
		Return None
		''' 

		# sets session argument
		self.__theSessionArgument = re.sub(':\s*',':',argument)



	def getSessionArgument(self):
		'''returns argument of session
		Return str : argument of session
		''' 

		# returns the session argument
		return self.__theSessionArgument 



	def setExtraFileList(self,extrafilelist):
		'''sets extra file list
		extrafilelist(list of str)  -- extra file list
		Return None
		''' 

		# sets the extra file list
		self.__theExtraFileList = extrafilelist



	def getExtraFileList(self):
		'''returns extra file list
		Return list of str : extra file list
		''' 

		# returns the extra file list
		return self.__theExtraFileList 



	def setTimeout(self,timeout):
		'''sets timeout
		When timeout is 0, no limit is set.
		timeout(int) -- time out (sec.)
		Return None
		'''

		# sets time out
		self.__theTimeout = timeout



	def getTimeout(self):
		'''returns timeout
		Return int : time out (sec.)
		'''

		# returns the time out
		return self.__theTimeout 



	def retry(self):
		'''ABSTRACT : This method must be overwrote in subclass
		retries job
		Return None
		raise NotImplementedError
		'''

		# When this method is not implemented in sub class, 
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



	def run(self):
		'''runs process
		Return None
		[Note] This method must be called from the first of run method of sub class.
		'''

		# sets status
		self.setStatus(RUN)

		# -----------------------------
		# checks the retry count
		# -----------------------------

		# When status is RUN, and retry count is over max retry count,
		# sets ERROR status and does nothing.
		if self.getStatus() == RUN:
		
			if self.getRetryMaxCount() != 0:
				if self.__theRetryCount > self.getRetryMaxCount():
					self.setStatus(ERROR)
					return None

		# -----------------------------
		# runs
		# -----------------------------

		# counts up retry counter
		self.__theRetryCount += 1


		# saves current time
		self.__theStartTime = time.time()



	def update(self):
		'''updates status
		Return None
		[Note] This method must be called from the first of run method of sub class.
		'''

		# When the status is RUN, and running time is > timeout,
		# calls retry method of sub class
		if self.__theStatus == RUN:

			# When the timeout is set as 0, does not retry.
			if self.__theTimeout != 0:

				# checks the running time
				if time.time() - self.__theStartTime > self.__theTimeout:

					# calls retyr method of sub class
					self.retry()



	def stop(self):
		'''ABSTRACT : This method must be overwrote in subclass
		stops job
		Return None
		raise NotImplementedError
		'''

		# When this method is not implemented in sub class, 
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



	def setStatus(self,status):
		'''sets status 
		aStatus(int) -- QUEUED,RUN,ERROR, or FINISHED
		Return None
		'''

		# sets the status
		self.__theStatus = status



	def getStatus(self):
		'''returns status 
		Return int : QUEUED,RUN,ERROR, or FINISHED
		'''

		# returns the status
		return self.__theStatus



	def getRetryCount(self):
		'''returns retry count
		Return int : retry count
		'''

		# returns the retry count
		return self.__theRetryCount



	def getRetryMaxCount(self):
		'''retunrs retry max count
		Return int : retry max count
		'''

		# returns the retry max count
		return self.__theRetryMaxCount 


	def setJobDirectory(self,jobdirectory):
		'''sets job directory
		jobdirectory(str) -- job directory
		Return None
		'''

		# saves job directory
		self.__theJobDirectory = jobdirectory


	def getJobDirectory(self):
		'''returns job directory
		Return str : job directory
		'''

		# returns job directory
		return self.__theJobDirectory


	def clearJobDirectory(self):
		'''removes job directory if there is.
		Return None
		'''

		# when the job directory exists, deletes it.
		if os.path.isdir( self.__theJobDirectory ):
			shutil.rmtree(self.__theJobDirectory)

		# if not, does nothing
		else:
			pass


# end of class SessionProxy





