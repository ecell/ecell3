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

from ecell.SessionManager.Util import *
from ecell.SessionManager.SessionProxy import *


class SGESessionProxy(SessionProxy):
	'''SGESessionProxy class
	Target environment is local PC that has only one cpu.
	'''

	QSUB = 'qsub'
	QSTAT = 'qstat'
	QHOST = 'qhost'

	def __init__(self):
		'''constructor
		'''

		# call superclass's constructor
		SessionProxy.__init__(self)

		# initialize parameter
		self.__theSGEJobID = None

		# initialize script name
		# if current shell is '/bin/tcsh', set 'script.tcsh' 
		self.__theTmpScriptFileName = "script." + os.path.basename( getCurrentShell() )


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

		if SessionProxy.getOptionList() == None or \
		   len(SessionProxy.getOptionList()) == 0:
			anOptionStr = ""
		else:
			anOptionMap = {}
			for anOption in SessionProxy.getOptionList():
				anOptionMap[anOption] = None
			if anOptionMap.has_key("-cwd"):
				del anOptionMap["-cwd"] 
			anOptionStr = string.join(anOptionMap.keys(),' ')


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
			aContext = "%s -v %s -cwd -S %s -o %s -e %s %s %s" \
			            %(SGESessionProxy.QSUB,
			             aHyphenVOption,
			             getCurrentShell(),
			             self.getStdoutFileName(),
			             self.getStderrFileName(),
			             anOptionStr,
			             self.__theTmpScriptFileName)


		# --------------------------------------------------------------------------
		# When the interpreter is users' script.
		# --------------------------------------------------------------------------
		else: # if(1)

			# convert argument into string
			anArgument = str(self.getArgument())

			# create a context
			aContext = "%s -v %s -cwd -S %s -o %s -e %s %s %s %s" \
			           %(SGESessionProxy.QSUB,
			            getEnvString(),
			            self.getInterpreter(),
			            self.getStdoutFileName(),
			            self.getStderrFileName(),
			            anOptionStr,
			            self.getScriptFileName(),
			            anArgument)

		# end of if(1)


		# --------------------------------------------------------------------------
		# execute the context
		# --------------------------------------------------------------------------
		self.__theSGEJobID = string.split(os.popen(aContext).readline())[2]
		

		# --------------------------------------------------------------------------
		# get back previous directory
		# --------------------------------------------------------------------------
		os.chdir( aCwd )

	# end of run



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




