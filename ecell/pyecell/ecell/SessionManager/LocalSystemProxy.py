#! /usr/bin/env python

'''
A module for session manager
 - for Local environment ( The number of CPU is only one. )
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

Design: Kouichi Takahashi <shafi@e-cell.org>
Programming: Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

 E-Cell Project, Lab. for Bioinformatics, Keio University.

'''

# imports standard modules
import sys
import string
import os
import time
import signal


# imports ecell modules
from Util import *
from LocalSessionProxy import *
from SystemProxy import *


class LocalSystemProxy(SystemProxy):
	'''LocalSystemProxy
	'''

	def __init__(self,sessionmanager):
		'''constructor
		sessionmanager -- the reference to SessionManager
		'''

		# calls superclass's constructor
		SystemProxy.__init__(self,sessionmanager)


	def getDefaultConcurrency(self):
		'''returns default cpus
		Return int : the number of cpus
		'''

		# returns the number of cpu
		return 1


	def createSessionProxy(self):
		'''creates and returns new LocalSessionProxy instance
		Return LocalSessionProxy
		'''

		# creates and returns new LocalSession Proxy instance
		return LocalSessionProxy()



	def update(self):
		'''
		updates status
		Return None
		'''

		# updates all SessionProxy's status
		for aSessionProxy in self.theSessionManager.getSessionProxy().values():

			aSessionProxy.update()


		# calculates the number of jobs to be run
		aFinishedJobNumber = len(self.theSessionManager.getFinishedJobList())
		anErrorJobNumber = len(self.theSessionManager.getErrorJobList())
		aRunningJobNumber = len(self.theSessionManager.getRunningJobList())
		aDispatchNumber = self.theSessionManager.getConcurrency() - aRunningJobNumber

		#print "cpu - finished = runnning " 
		print "cpu(%s) finished(%s) error(%s) runnning(%s) " %( 
		                                      self.theSessionManager.getConcurrency() ,
		                                      aFinishedJobNumber ,
		                                      anErrorJobNumber ,
		                                      aDispatchNumber )

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
					if aDispatchCount >= aDispatchNumber:
						break


		#for aSessionProxy in self.theSessionManager.getSessionProxy().values():
		#	aJobID = aSessionProxy.getJobID()
		#	aStatus = aSessionProxy.getStatus()
		#	print " aJobID = %s aStatus = %s%s" %(aJobID,aStatus,STATUS[aStatus])


			






