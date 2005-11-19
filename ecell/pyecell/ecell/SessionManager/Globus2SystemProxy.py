#! /usr/bin/env python

'''
A module for session manager
 - for grid environment ( Globus 2.4 )
 - privides API depending on environment


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

Design: Koichi Takahashi <shafi@e-cell.org>
Programming: Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

 E-Cell Project, Lab. for Bioinformatics, Keio University.

'''

# imports standard modules
import sys
import string
import os
import time
import signal
import re
import popen2


# imports ecell modules
from Util import *
from Globus2SessionProxy import *
from SystemProxy import *

GRID_PROXY_INIT = 'grid-proxy-init'
GRID_INFO_SEARCH = 'grid-info-search'
GLOBUS_JOB_RUN='globus-job-run'

class Globus2SystemProxy(SystemProxy):
	'''Globus2SystemProxy
	'''

	def __init__(self,sessionmanager):
		'''Constructor
		sessionmanager -- the reference to SessionManager
		'''

		# calls superclass's constructor
		SystemProxy.__init__(self,sessionmanager)

		# get user parameter for qstat
		self.__theUser = os.getenv('USER')

		self.__thePassword = None
		self.__theIdentity = None
		self.__theHostList = []
		self.__theFreeCpuList = []

		# set the reference of this class to
		# SessionProxy
		Globus2SessionProxy.setSystemProxy(self)

		sys.exitfunc = self.__exit


	def __exit(self):
		'''When Globus2SystemProxy exits, this method id called.
		Remove tmp root directory on remote hosts.
		'''

		for aHost in self.__theHostList:

			if aHost != self.getLocalHost():
				aCommand = "%s %s %s -rf %s%s%s" \
				            %(GLOBUS_JOB_RUN, aHost, RM, os.getcwd(),\
				              os.sep, self.theSessionManager.getTmpRootDir() ) 
				#print "delete %s" %aCommand
				os.system(aCommand)


	def setPassword( self, password ):
		'''Set password for gate keeper
		This must be called before gridProxyInit
		password(str) -- passowrd of gate keeper
		Return None
		'''

		self.__thePassword = password


	def gridProxyInit( self ):
		'''initialize gate keeper
		Before calling this method, setPassword must be called.
		Return None
		'''

		if self.__thePassword == None:
			raise("Error: setPassword(password) must be called before this method.")


		# execute grid-proxy-init
		aStdout,aStdin,aStderr = \
		popen2.popen3("echo %s | %s -pwstdin" %(self.__thePassword, GRID_PROXY_INIT))


		# check standard error
		aStderrContents = string.join( aStderr.readlines() )
		if len(aStderrContents) > 0:
			raise("Error: %s failed. \n%s" %(GRID_PROXY_INIT,aStderrContents))

		# get identity 
		aStdinContentList = aStdout.readlines()
		self.__theIdentity = string.split(aStdinContentList[0])[2]


	def setHosts( self, hostlist ):
		'''Set host names on which the jobs will be conducted
		hostlist(list of str) -- list of host name
		Return None
		'''

		if type(hostlist) != list:
			raise TypeError("hostlist must be a list of host name.")

		self.__theHostList = hostlist

	def getDefaultConcurrency(self):
		'''returns default cpus
		Return int : the number of cpus
		'''

		return 1


	def createSessionProxy(self):
		'''creates and returns new Globus2SessionProxy instance
		Return Globus2SessionProxy
		'''

		# creates and returns new Globus2Session Proxy instance
		return Globus2SessionProxy()


	def updateFreeCpuUsingMDS(self):
		'''Update free cpu information using MDS.
		The free cpu is defined that the MdsCpu Mds-Cpu-Free-1minX100 > 60.
		Return None
		'''

		self.__theFreeCpuList = []

		for aHost in self.__theHostList:

			aCommand = "%s -h %s objectClass=MdsCpu Mds-Cpu-Free-1minX100" \
			           %(GRID_INFO_SEARCH,aHost)

			aStdout, aStdin, aStderr = popen2.popen3(aCommand)
			for aLine in aStdout.readlines():
				#print "(%s)" %aLine[:-1]
				if string.find(aLine,'Mds-Cpu-Free-1minX100') == 0:
					#print "%s -> (%s)" %(aHost, aLine[23:-1] )
					aFreeCpu = string.atoi( aLine[23:-1] )
					#print "free cpu = %s" %aFreeCpu
					if aFreeCpu > 60:
						self.__theFreeCpuList.append(aHost) 
					break

	def getFreeCpuList(self):
		'''Return free cpus
		Return list of str : cpu names
		'''

		return self.__theFreeCpuList



	def update(self):
		'''updates status
		Return None

		Updates status using the result of qstat as below.

		'''

		# update free cpu list
		self.updateFreeCpuUsingMDS()

		# updates all SessionProxy's status
		for aSessionProxy in self.theSessionManager.getSessionProxy().values():
			aSessionProxy.update()

		# calculates the number of jobs to be run
		aFinishedJobNumber = len(self.theSessionManager.getFinishedJobList())
		aRunningJobNumber = len(self.theSessionManager.getRunningJobList())
		aDispatchNumber = self.theSessionManager.getConcurrency() - aRunningJobNumber


		#print "finish %s, running %s, dispatch %s" \
		#       %(aFinishedJobNumber,aRunningJobNumber,aDispatchNumber)
		#print len(self.__theFreeCpuList) 


		# When some jobs to be runned,
		if aDispatchNumber != 0 and len(self.__theFreeCpuList) > 0:


			# initializes counter of the number of jobs to be runned
			aDispatchCount = 0

			for aSessionProxy in self.theSessionManager.getSessionProxy().values():

				# when the status is QUEUED
				if aSessionProxy.getStatus() == QUEUED:

					#print self.__theFreeCpuList

					aFreeCpu = self.__theFreeCpuList[0]

					#print "free cpu = %s " %aFreeCpu

					aSessionProxy.setCpu(aFreeCpu)

					# call run method
					aSessionProxy.run()

					# count up
					aDispatchCount += 1

					# check the number of jobs to be conducted.
					if aDispatchCount == aDispatchNumber:
						break

					# check the number of free cpus that can be used.
					if len(self.__theFreeCpuList) == 1:
						break
					else:
						self.__theFreeCpuList = self.__theFreeCpuList[1:]

	# end of def update




