#! /usr/bin/env python

'''
A module for session manager
 - for cluster environment ( Sun Grid Engine )
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

from ecell.Util import *
from SessionProxy import *


QSUB = 'qsub'


class SGESessionProxy(SessionProxy):
	'''SGESessionProxy class
	Target environment is local PC that has only one cpu.
	'''


	def __init__(self):
		'''constructor
		'''

		# call superclass's constructor
		SessionProxy.__init__(self)

		# initialize parameter
		self.__theSGEJobID = None

		# initialize script name
		# for example, script.tcsh if $SHELL is '/bin/tcsh'
		self.__theTmpScriptFileName = "script." + os.path.basename( os.getenv('SHELL') )



	def getSGEJobID(self):
		'''return job id 
		'''
		return self.__theSGEJobID 


	def retry(self):
		'''retry
		Return None
		'''


		# When the status is RUN, kill the sub process
		if self.getStatus() == RUN:
			os.popen("qdel %s" %self.__theSGEJobID )

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

		# argument
		anArgment = ''
		if self.getInterpreter() == ECELL3_SESSION:

			# write temporary script
			anArgument = "--parameters=\"%s\"" %str(self.getSessionArgument())
			aScriptContext = "#!%s\nuname -a\n%s -e %s %s\n exit $?" \
			                  %(os.getenv('SHELL'),
			                    ECELL3_SESSION,
			                    self.getScriptFileName(),
			                    anArgument)
			open( self.__theTmpScriptFileName, 'w' ).write( aScriptContext )


			# create context to be thrown by qsub
			aContext = "%s -cwd -S %s -o %s -e %s %s" \
			                                                 %(QSUB,
			                                                 os.getenv('SHELL'),
					                                         self.getStdoutFileName(),
					                                         self.getStderrFileName(),
		                                                     self.__theTmpScriptFileName)



		else:
			anArgument = str(self.getArgument())
			aContext = "%s -cwd -S %s -o %s -e %s %s %s" %(QSUB,
		                                            self.getInterpreter(),
					                                self.getStdoutFileName(),
					                                self.getStderrFileName(),
		                                            self.getScriptFileName(),
		                                            anArgument)


		# execute the context
		self.__theSGEJobID = string.split(os.popen(aContext).readline())[2]
		

		# get back previous directory
		os.chdir( aCwd )



	def update( self ):
		'''update jobs's status
		Return None
		'''

		# call super class's method
		SessionProxy.update(self)



	def stop(self):
		'''stop the job
		Return None
		'''

		# When this job is running, stop it.
		if self.getStatus() == RUN:

			os.popen("qdel %s" %self.__theSGEJobID )

		# set error status
		self.setStatus(ERROR) 


# end of class SGESessionProxy




