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
		'''Return next job ID.
		Return int : job ID
		[Note]:private method, the first job id is 1
		'''

		# count up jod id
		SessionProxy.__theJobID += 1

		# return job id
		return SessionProxy.__theJobID


	def setStdoutFileName(cls, stdout):
		'''Set an standard output.
		stdout(str or file) -- file name or file object to be wrote
		Return None
		[Note] default stdout is 'stdout'
		'''

		# set stdout
		SessionProxy.__theStdout = stdout


	def getStdoutFileName(cls):
		'''Return the standard output.
		Return str or file -- standard output
		[Note] default stdout is 'stdout'
		'''

		# return stdout
		return SessionProxy.__theStdout


	def setStderrFileName(cls, stderror):
		'''Set a standard error.
		stdout(str or file) -- file name or file object to be wrote
		Return None
		[Note] default stderr is 'stderr'
		'''

		SessionProxy.__theStderror = stderror


	def getStderrFileName(cls):
		'''Return standard error.
		Return str or file -- standard error
		[Note] default stderr is 'stderr'
		'''

		return SessionProxy.__theStderror


	def setRetryMaxCount(cls, count):
		'''Set retry max count.
		When count < 0, count is set as 0
		count(int) -- retry max count
		Return None
		[Note] default max count is 0
		'''

		if count < 0:
			count = 0

		SessionProxy.__theRetryMaxCount = count


	# register class methods
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
		a job id is set.
		a status is set as QUEUED
		all attributes are initialized.
		'''

		# set job id
		self.__theJobID = SessionProxy.__getNextJobID()


		# set status as QUEUED
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
		'''When destractor is called, deletes the job directory.
		Return None
		'''

		# remove job directory
		self.clearJobDirectory()

		# destractor is called automatically after this method



	def getJobID(self):
		'''Return the job id
		Return int : job id
		''' 

		# return the job id
		return self.__theJobID 



	def setScriptFileName(self,scriptfilename):
		'''Set a script file name.
		scriptfilename(str) -- a script file name
		Return None
		''' 

		# Set the script file name.
		self.__theScriptFileName = scriptfilename



	def getScriptFileName(self):
		'''return the script file name
		Return str : script file name
		''' 

		# Return the script file name
		return self.__theScriptFileName 



	def setInterpreter(self,interpreter):
		'''Set an interpreter.
		interpreter(str) -- an interpreter
		Return None
		'''

		aStdout, aStdin, aStderr = popen2.popen3( "which %s" %interpreter )

		try:
			aPath = aStdout.readline()[:-1]
		except IOError:
			aPath = interpreter

		# set the interpreter
		self.__theInterpreter = os.path.abspath( aPath )


	def getInterpreter(self):
		'''Return the interpreter.
		Return str : interpreter
		''' 

		# return the interpreter
		return self.__theInterpreter 



	def setArgument(self,argument):
		'''Set an argument.
		argument(str) -- an argument to be set to script
		Return None
		''' 

		# set the argument
		self.__theArgument = argument



	def getArgument(self):
		'''Return the argument.
		Return None
		''' 

		# return the artument
		return self.__theArgument 



	def setSessionArgument(self,argument):
		'''Set an argument to session.
		argument(dict)  -- argument to be set to session
		Return None
		''' 

		# set session argument
		#self.__theSessionArgument = re.sub(':\s*',':',argument)
		self.__theSessionArgument = re.sub(',\s*', ',', re.sub(':\s*',':',str(argument)))



	def getSessionArgument(self):
		'''Returns the argument of session.
		Return str : argument of session
		''' 

		# return the session argument
		return self.__theSessionArgument 



	def setExtraFileList(self,extrafilelist):
		'''Set an extra file list.
		extrafilelist(list of str)  -- extra file list
		Return None
		''' 

		# set the extra file list
		self.__theExtraFileList = extrafilelist



	def getExtraFileList(self):
		'''Returns the extra file list.
		Return list of str : extra file list
		''' 

		# return the extra file list
		return self.__theExtraFileList 



	def setTimeout(self,timeout):
		'''Set a timeout.
		When timeout is 0, no limit is set.
		timeout(int) -- time out (sec.)
		Return None
		'''

		# set a timeout
		self.__theTimeout = timeout



	def getTimeout(self):
		'''Return the timeout.
		Return int : time out (sec.)
		'''

		# return the time out
		return self.__theTimeout 



	def retry(self):
		'''ABSTRACT : This method must be overwrote in subclass
		retry job
		Return None
		raise NotImplementedError
		'''

		# When this method is not implemented in sub class, 
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



	def run(self):
		'''run process
		Return None
		[Note] This method must be called from the run method of sub class.
		'''

		if self.getStatus() != QUEUED:
			return None

		# -----------------------------
		# check the retry count
		# -----------------------------

		# When status is RUN, and retry count is over max retry count,
		# set ERROR status and does nothing.
		#if self.getStatus() == RUN:
		
		if self.getRetryMaxCount() != 0:
			if self.__theRetryCount >= self.getRetryMaxCount():
				self.setStatus(ERROR)
				return None

		# set a status
		self.setStatus(RUN)

		# -----------------------------
		# run
		# -----------------------------

		# count up retry counter
		self.__theRetryCount += 1


		# save current time
		self.__theStartTime = time.time()



	def update(self):
		'''update status
		Return None
		[Note] This method must be called from the first of run method of sub class.
		'''

		# When the status is RUN, and running time is > timeout,
		# call retry method of sub class
		if self.__theStatus == RUN:

			# When the timeout is set as 0, do not retry.
			if self.__theTimeout != 0:

				# check the running time
				if time.time() - self.__theStartTime > self.__theTimeout:

					# call retyr method of sub class
					self.retry()



	def stop(self):
		'''ABSTRACT : This method must be overwrote in subclass
		stop job
		Return None
		raise NotImplementedError
		'''

		# When this method is not implemented in sub class, 
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



	def setStatus(self,status):
		'''Set a status.
		aStatus(int) -- QUEUED,RUN,ERROR, or FINISHED
		Return None
		'''

		# set a status
		self.__theStatus = status



	def getStatus(self):
		'''Return the status.
		Return int : QUEUED,RUN,ERROR, or FINISHED
		'''

		# return the status
		return self.__theStatus



	def getRetryCount(self):
		'''Returns the retry count.
		Return int : retry count
		'''

		# return the retry count
		return self.__theRetryCount



	def getRetryMaxCount(self):
		'''Return the retry max count.
		Return int : retry max count
		'''

		# return the retry max count
		return self.__theRetryMaxCount 


	def setJobDirectory(self,jobdirectory):
		'''Set a job directory.
		jobdirectory(str) -- job directory
		Return None
		'''

		# save job directory
		self.__theJobDirectory = jobdirectory


	def getJobDirectory(self):
		'''Return job directory.
		Return str : job directory
		'''

		# return job directory
		return self.__theJobDirectory


	def clearJobDirectory(self):
		'''Remove job directory if there is.
		Return None
		'''

		# when the job directory exists, deletes it.
		if os.path.isdir( self.__theJobDirectory ):
			shutil.rmtree(self.__theJobDirectory)

		# if not, do nothing
		else:
			pass


# end of class SessionProxy





