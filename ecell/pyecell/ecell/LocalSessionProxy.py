#! /usr/bin/env python

'''
A module for session manager
 - for Local environment ( The number of CPU is only one. )
 - privides API of one process

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
import popen2

from ecell.Util import *
from SessionProxy import *

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

		# call super class's method
		SessionProxy.update(self)


		if self.__theSubProcess == None:
			return None


		#print "jobid %s---->(%s)" %(self.getJobID(),self.__theSubProcess.poll())

		# check the running status 
		if self.__theSubProcess.poll() == -1:

			# when this job if running, do nothing
			return None
	
		# when this job is not running

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

