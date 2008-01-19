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

import sys
import string
import os
import popen2

from ecell.SessionManager.Util import *
from ecell.SessionManager.SessionProxy import *

class LocalSessionProxy(SessionProxy):
	'''LocalSessionProxy class
	Target environment is local PC that has only one cpu.
	'''


	def __init__(self):
		'''Constructor
		'''

		# call superclass's constructor
		SessionProxy.__init__(self)

		# initialize parameter
		self.__theSubProcess = None



	def retry(self):
		'''retry
		Return None
		'''


		# When the status is RUN, kill the sub process
		if self.getStatus() == RUN:
			aPid = self.__theSubProcess.pid
			os.kill(aPid,1)

		self.setStatus( QUEUED )

		# run again.
		self.run()



	def run(self):
		'''run this process
		Return None
		'''

		# --------------------------------
		# set up
		# --------------------------------
		# call super class's method
		SessionProxy.run(self)

		#print "run %s" %self.getJobID()

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
		#print str(self.getSessionArgument())
		if self.getInterpreter() == ECELL3_SESSION:
			
			if self.getDM_PATH != "":
				
				aContext = "ECELL3_DM_PATH=%s %s -e %s --parameters=\"%s\"" %(self.getDM_PATH(),
									     ECELL3_SESSION,
									     os.path.basename(self.getScriptFileName()),
									     str(self.getSessionArgument()))
			else:
				aContext = "%s -e %s --parameters=\"%s\""%(ECELL3_SESSION,
									   os.path.basename(self.getScriptFileName()),
									   str(self.getSessionArgument()))
			# print aContext
		else:
			aContext = "%s %s %s" %(self.getInterpreter(),
		                            os.path.basename(self.getScriptFileName()),
		                            self.getArgument())

		# execute the context
		self.__theSubProcess = popen2.Popen3( aContext, capturestderr=True )

		# get back previous directory
		os.chdir( aCwd )



	def update( self ):
		'''update jobs's status
		Return None
		'''

		#print "update %s" %self.getJobID()

		# call super class's method
		SessionProxy.update(self)


		# when this job is still waiting.
		if self.__theSubProcess == None:
			return None

		# check the running status 
		if self.__theSubProcess.poll() == -1:

			# when this job if running, do nothing
			return None
	
		# when this job is not running (finised or error)

		#print "finish or error %s" %self.getJobID()
		if self.theOutputCopyDoneStatus == True:
			return None
		self.theOutputCopyDoneStatus = True

		#print self.__theSubProcess.fromchild

		aStdout = self.__theSubProcess.fromchild
		aStderr = self.__theSubProcess.childerr

		# --------------------------------
		# write an output to file
		# --------------------------------

		# write a stdout file 
		aStdoutFile = "%s%s%s" %( self.getJobDirectory(),
		                          os.sep,
		                          self.getStdoutFileName() )
		shutil.copyfileobj(aStdout,open(aStdoutFile,'a'))


		# write a stderr file 
		aStderrFile = "%s%s%s" %( self.getJobDirectory(),
		                          os.sep,
		                          self.getStderrFileName() )
		shutil.copyfileobj(aStderr,open(aStderrFile,'a'))

		# --------------------------------
		# check status
		# --------------------------------
		
		# check the file size of standard error file
		# when the file size is 0, set status as FINISHED
		if os.stat(aStderrFile)[6] == 0L:

			# sets finished status
			self.setStatus(FINISHED) 
		
		# when the file size is 0, set status as ERROR
		else:

			# sets error status
			self.setStatus(ERROR) 

		# relase the memory of sub process
		self.__theSubProcess = None


	def stop(self):
		'''stop this job
		Return None
		'''

		if self.getStatus() == RUN:
			aPid = self.__theSubProcess.pid
			#print "stop %s " %aPid
			os.kill(aPid,1)

		# set error status
		self.setStatus(ERROR) 


# end of class LocalSessionProxy

