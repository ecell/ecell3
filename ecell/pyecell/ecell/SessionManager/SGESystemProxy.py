#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
# Designed by Koichi Takahashi <shafi@e-cell.org>
# Programmed by Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

# imports standard modules
import sys
import string
import os
import time
import signal
import re
import popen2


# imports ecell modules
from ecell.SessionManager.Util import *
from ecell.SessionManager.SGESessionProxy import *
from ecell.SessionManager.SystemProxy import *


class SGESystemProxy(SystemProxy):
	'''SGESystemProxy
	'''

	def __init__(self,sessionmanager):
		'''constructor
		sessionmanager -- an instance of SessionManager
		'''

		# Check the argument type.
		if sessionmanager.__class__.__name__ != "SessionManager":
			raise TypeError("A type of sessionmanager is \"%s\". The type of sessionmanager must be SessionManager" %type(sessionmanager))

		# Call superclass's constructor.
		SystemProxy.__init__(self,sessionmanager)

		# Get user parameter for qstat.
		self.__theUser = os.getenv('USER')

		# Check the existance of SGE command.
		# qsub
		if checkCommandExistence(SGESessionProxy.QSUB) == False:
			raise Exception("\"%s\" is not included in the $PATH environment"\
			                %SGESessionProxy.QSUB)

		# qstat
		if checkCommandExistence(SGESessionProxy.QSTAT) == False:
			raise Exception("\"%s\" is not included in the $PATH environment"\
			                %SGESessionProxy.QSTAT)

		# qhost
		if checkCommandExistence(SGESessionProxy.QHOST) == False:
			raise Exception("\"%s\" is not included in the $PATH environment"\
			                %SGESessionProxy.QHOST)

		# Initialize attributes.
		self.__theQueueList = None

	# end of def __init__(self,sessionmanager):


	def getDefaultConcurrency(self):
		'''returns default cpus
		Return int : the number of cpus
		The default number of cpu is calculated from the retuslt of qhost
		Here is the sample of standard output of qhost

		HOSTNAME             ARCH       NPROC  LOAD   MEMTOT   MEMUSE   SWAPTO   SWAPUS
		-------------------------------------------------------------------------------
		global               -              -     -        -        -        -        -
		i10                  glinux         2  0.00  1006.1M   637.3M     2.0G   111.6M
		i11                  glinux         2  0.01  1006.1M   666.7M     2.0G   144.0M
		i12                  glinux         2  0.00  1006.1M   295.6M     2.0G     2.1M
		i13                  glinux         2  0.00  1006.1M   297.2M     2.0G     2.4M

		In this case, this method returns 4 as a default number of CPUs.
		'''

		# returns the number of cpu
		aStdout = os.popen('qhost | wc -l | sed -e "s/ //g"').readline()
		if aStdout[-1] == '\n':
			aStdout = aStdout[:-1]
		
		aCpuNumber = string.atoi(aStdout) -3
		if aCpuNumber < 0:
			aCpuNumber = 0

		return aCpuNumber

	# end of def getDefaultConcurrency(self):


	def createSessionProxy(self):
		'''creates and returns new SGESessionProxy instance
		Return SGESessionProxy
		'''

		# creates and returns new SGESession Proxy instance
		return SGESessionProxy()

	# end of def createSessionProxy(self):


	def update(self):
		'''updates status
		Return None

		Updates status using the result of qstat as below.

		job-ID  prior name       user         state submit/start at     queue      master  ja-task-ID 
		---------------------------------------------------------------------------------------------
		4243651     0 test.py    hoge         Eqw   08/28/2003 11:24:26                       

		When the state is 'qw(queued/waiting)', 't(transfer)' or 'r(running)', it is set as RUN.
		When the state is 'Eqw(error)', the job is killed and its status is set as ERROR.
		In most of latter case, the interpreter can't find remote machine could not be found.
		However, above error does not be written stderr.
		'''

		# initializes a dict whose key is SGE job id and value is status
		aStatusDict = {}

		# reads the result of qstat
		aLines = os.popen("%s -u %s" %(SGESessionProxy.QSTAT,self.__theUser)).readlines()

		# When there are running jobs, gets SGE job id and status
		if len(aLines) >= 3:
			for aLine in aLines[2:]:
				aLineList = string.split(aLine)
				aStatusDict[ aLineList[0] ] = aLineList[4]


		# checks ths status of each SessionProxy
		for aSessionProxy in self.theSessionManager.getSessionProxy().values():

			# considers only running jobs
			if aSessionProxy.getStatus() == RUN:

				# gets SGE job id
				aSGEJobID = aSessionProxy.getSGEJobID()

				# there is no SGE job id in the result of qstat, the job is 
				# considered to be finished
				if aStatusDict.has_key(str(aSGEJobID)) == False:

					# read standard error file
					aStderrFile = aSessionProxy.getJobDirectory() + \
					              os.sep + aSessionProxy.getStderrFileName()
					aStderrList = open(aStderrFile,'r').readlines()

					# When something is writtend in standard error,
					if len(aStderrList) > 0 :

						if aStderrList[-1] == 'ValueError: bad marshal data\n' or \
					      aStderrList[-1] == 'EOFError: EOF read where object expected\n':

							aSessionProxy.retry()

						else:

							aSessionProxy.setStatus(ERROR)

						continue


					aSessionProxy.setStatus(FINISHED)

				# When job is running,
				else:

					# When character 'E' is included in the status,
					if string.find(aStatusDict[aSGEJobID],'E') != -1:

						aSessionProxy.stop()

					else:

						pass


		# updates all SessionProxy's status
		for aSessionProxy in self.theSessionManager.getSessionProxy().values():
			aSessionProxy.update()


		# calculates the number of jobs to be run
		aFinishedJobNumber = len(self.theSessionManager.getFinishedJobList())
		aRunningJobNumber = len(self.theSessionManager.getRunningJobList())
		aDispatchNumber = self.theSessionManager.getConcurrency() - aRunningJobNumber


		# When some jobs to be runned,
		if aDispatchNumber != 0:

			# initializes counter of the number of jobs to be runned
			aDispatchCount = 0

			for aSessionProxy in self.theSessionManager.getSessionProxy().values():

				# when the status is QUEUED
				if aSessionProxy.getStatus() == QUEUED:

					# calls run method
					aSessionProxy.run()

					# counts up
					aDispatchCount += 1

					# checks the number of jobs to be runned
					if aDispatchCount == aDispatchNumber:
						break


		#for aSessionProxy in self.theSessionManager.getSessionProxy().values():
		#	aJobID = aSessionProxy.getJobID()
		#	aStatus = aSessionProxy.getStatus()
		#	print " aJobID = %s aStatus = %s" %(aJobID,aStatus)


	# end of def update


	def setOptionList(self,optionlist):
		'''set option list to SGESessionProxy.
		optionlist(list of str) -- a list of options

		Return None
		'''

		# Check the argument type.
		if type(optionlist) != list:
			raise TypeError("A type of optionlist is \"%s\". The type of optionlist must be a list." %type(sessionmanager))


		# Check all elements in option list.
		for anOption in optionlist:

			# Check the type of element.
			if type(anOption) != str:
				raise TypeError("A type of element of optionlist is \"%s\". The type of element of optionlist must be \"str\"." %type(anOption))

		
			# Check "-q option".
			if len(anOption) == 4 and anOption[0:3] == "-q ":
				raise Exception("The [-q] option is set but no queue is specified.")


			# Check the value of "-q option".
			if len(anOption) > 4 and anOption[0:3] == "-q ":
				aQueueList = string.split(anOption[3:],",")
				for aQueue in aQueueList:
					if self.checkQueueExistence(aQueue) == False:
						raise Exception("The queue [%s] is not available." %aQueue)

		# Set the option list to SessionProxy.
		SessionProxy.setOptionList( optionlist )

	# end of def setOptionList(self,optionlist):


	def checkQueueExistence(self,queue):
		'''creates and returns new SGESessionProxy instance
		queue(str) -- a queue name

		Return True(exists)/False(does not exists)
		'''

		# Check the type of element.
		if type(queue) != str:
			raise TypeError("A type of queue is \"%s\". The type of queue must be \"str\"." %type(queue))

		# Get queue list.
		if self.__theQueueList == None:
			aCheckProcess = popen2.Popen3("%s -q | grep \".q \" | awk \'NR>3 {print $1}\'" %SGESessionProxy.QHOST)
			aCheckProcess.wait()
			self.__theQueueList = []
			for aQueue in aCheckProcess.fromchild.readlines():
				self.__theQueueList.append(aQueue[:-1])

		# Check the existence of the queue.
		if queue in self.__theQueueList:
			return True
		else:
			return False


	# end of def checkQueueExistence(self,queuelist):



# end of class SGESystemProxy(SystemProxy):




