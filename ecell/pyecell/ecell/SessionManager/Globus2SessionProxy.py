#! /usr/bin/env python

'''
A module for session manager
 - for grid environment ( Globus 2.4 )
 - privides API of one process

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

import sys
import string
import os

from Util import *
from SessionProxy import *

GLOBUS_JOB_SUBMIT='globus-job-submit'
GLOBUS_JOB_GET_STATUS='globus-job-status'
GLOBUS_JOB_GET_OUTPUT='globus-job-get-output'
GLOBUS_JOB_CLEAN='globus-job-clean'

class Globus2SessionProxy(SessionProxy):
	'''Globus2SessionProxy class
	Target is grid environment.
	'''

	def __init__(self):
		'''constructor
		'''

		# call superclass's constructor
		SessionProxy.__init__(self)

		# initialize parameter
		self.__theContactString = None

		# initialize script name
		# if current shell is '/bin/tcsh', set 'script.tcsh' 
		self.__theTmpScriptFileName = "script." + os.path.basename( getCurrentShell() )

		#print self.getSystemProxy().__class__.__name__

		self.__theCpu = None

	def getContactString(self):
		'''return job id 
		'''
		return self.__theContactString 


	def retry(self):
		'''retry
		Return None
		'''

		# When the status is RUN, kill the sub process
		if self.getStatus() == RUN:
			#os.popen("qdel %s" %self.__theGlobus2JobID )
			os.popen("qdel %s" %self.__theGlobus2JobID )

		self.setStatus(QUEUED)
	

		# run again.
		self.run()



	def run(self):
		'''run process
		Return None
		'''

		self.setStatus(RUN)

		# --------------------------------
		# set up
		# --------------------------------
		# call super class's method
		SessionProxy.run(self)

		# check status
		if self.getStatus() == FINISHED or self.getStatus() == ERROR:
			return None


		# save current directory
		aCwd = os.getcwd()

		# change directory to job directory
		os.chdir( self.getJobDirectory() )

		# --------------------------------
		# execute script
		# --------------------------------
		# create context


		# --------------------------------------------------------------------------
		# When the interpreter is E-Cell session
		# --------------------------------------------------------------------------
		if self.getInterpreter() == ECELL3_SESSION:  # if(1)

			# create argument string
			anArgument = "%s -e %s --parameters=\"%s\"" \
			              %(ECELL3_SESSION,
			                self.getScriptFileName(),
			                str(self.getSessionArgument()))

			# create script context
			aScriptContext = "#!%s\n%s\n\n" \
				                  %(getCurrentShell(),
				                    anArgument)

			# write script file
			open( self.__theTmpScriptFileName, 'w' ).write( aScriptContext )


			# -----------------------------------
			# create a value of option -v of qsub
			# -----------------------------------
			aHyphenVOption = getEnvString()

			# when ECELL3_DM_PATH is set, append it to the -v option 
			if self.getDM_PATH() != "": 
				aHyphenVOption += ",ECELL3_DM_PATH=%s" %self.getDM_PATH()


			# create a context
			aContext = "%s %s -env PATH=$PATH -dir %s -s %s" %(
			             GLOBUS_JOB_SUBMIT,
			             self.__theCpu,
			             os.path.abspath( os.getcwd() ),
			             self.__theTmpScriptFileName)


		# --------------------------------------------------------------------------
		# When the interpreter is users' script.
		# --------------------------------------------------------------------------
		else: # if(1)
			pass

			'''
			# convert argument into string
			anArgument = str(self.getArgument())

			# create a context
			aContext = "%s -v %s -cwd -S %s -o %s -e %s %s %s" \
			          %(getEnvString(),
			            self.__theCpu,
			            self.getStdoutFileName(),
			            self.getStderrFileName(),
			            self.getScriptFileName(),
			            anArgument)
			'''

		# end of if(1)


		# --------------------------------------------------------------------------
		# execute the context
		# --------------------------------------------------------------------------

		#print aContext
		#sys.exit(0)
		self.__theContactString = os.popen(aContext).readline()[:-1]

		# --------------------------------------------------------------------------
		# get back previous directory
		# --------------------------------------------------------------------------
		os.chdir( aCwd )

	# end of run


	def setCpu( self, cpu ):

		self.__theCpu = cpu

	def getCpu( self ):

		return self.__theCpu

	def update( self ):
		'''update jobs's status
		Return None
		'''

		# call super class's method
		SessionProxy.update(self)

		#GLOBUS_JOB_GET_OUTPUT='globus-job-get-output'


		if STATUS[self.getStatus()] == 'QUEUED': 
			return None

		aCommand = "%s %s" %(GLOBUS_JOB_GET_STATUS,\
	   	                  self.__theContactString)

		#print aCommand
		aStatus = os.popen(aCommand).readline()[:-1]
		#print "(%s)" %aStatus

		if aStatus == 'DONE':
			self.setStatus(FINISHED) 

			# standard output
			aCommand = "%s %s" %(GLOBUS_JOB_GET_OUTPUT,\
			                     self.__theContactString)
			aStdout = os.popen(aCommand).readlines()
			aStdoutFile = self.getJobDirectory() + os.sep + \
			              self.getStdoutFileName()
			open(aStdoutFile,'w').write(string.join(aStdout,''))

			# standard error output
			aCommand = "%s -err %s" %(GLOBUS_JOB_GET_OUTPUT,\
			                     self.__theContactString)
			aStderr = os.popen(aCommand).readlines()
			aStderrFile = self.getJobDirectory() + os.sep + \
			              self.getStderrFileName()
			open(aStderrFile,'w').write(string.join(aStderr,''))


		#print STATUS[self.getStatus()]


	def stop(self):
		'''stop the job
		Return None
		'''

		# When this job is running, stop it.
		if self.getStatus() == RUN:

			aCommand = "%s %s" %(GLOBUS_JOB_CLEAN,\
			                     self.__theContactString)
			os.popen(aCommand)

		# set error status
		self.setStatus(ERROR) 


# end of class Globus2SessionProxy




