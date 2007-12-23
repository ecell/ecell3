#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
import time
import popen2

import ecell.eml
import ecell.emc
import ecell.ecs


# constants

try:
	ECELL3_SESSION = os.path.abspath( os.popen('which ecell3-session').readline()[:-1] )
except IOError:
	ECELL3_SESSION = 'ecell3-session'

DEFAULT_STDOUT = 'stdout'
DEFAULT_STDERR = 'stderr'

BANNERSTRING =\
'''ecell3-session-manager [ E-Cell SE Version %s, on Python Version %d.%d.%d ]
Copyright (C) 2001-2007 Keio University.
Copyright (C) 2005-2007 The Molecular Sciences Institute.'''\
% ( ecell.ecs.getLibECSVersion(), sys.version_info[0], sys.version_info[1], sys.version_info[2] )


SYSTEM_PROXY = 'SystemProxy'
SESSION_PROXY = 'SessionProxy'

DEFAULT_TMP_DIRECTORY = 'tmp'
DEFAULT_ENVIRONMENT = 'Local'


# job status
QUEUED 		= 0
RUN		= 1
FINISHED 	= 2
ERROR		= 3

STATUS = { 0:'QUEUED',
           1:'RUN',
           2:'FINISHED',
           3:'ERROR',
          }


def createScriptContext( anInstance, parameters ):
	'''create script context
	'''

	# theSession == self in the script
	aContext = { 'theSession': anInstance,'self': anInstance }

	# flatten class methods and object properties so that
	# 'self.' isn't needed for each method calls in the script
	aKeyList = list ( anInstance.__dict__.keys() +\
                          anInstance.__class__.__dict__.keys() )
	aDict = {}
	for aKey in aKeyList:
		aDict[ aKey ] = getattr( anInstance, aKey )

		aContext.update( aDict )
		aContext.update( parameters )

		return aContext

def getCurrentShell():
	'''return the current shell
	Note: os.env('SHELL') returns not current shell but login shell.
	'''

	aShellName = string.split( os.popen('ps -p %s'%os.getppid()).readlines()[1] )[3]
	aCurrentShell = os.popen('which %s' %aShellName).read()[:-1]

	return aCurrentShell


def getEnvString():
	'''return the env string as below
	VARIABLE1=VALUE1,VARIABLE2=VALUE2,VARIABLE3=VALUE3, ...
	'''

	aString = "ECELL3_PREFIX=%s" %os.getenv('ECELL3_PREFIX')
	aString += ",LTDL_LIBRARY_PATH=%s" %os.getenv('LTDL_LIBRARY_PATH')
	aString += ",LD_LIBRARY_PATH=%s" %os.getenv('LD_LIBRARY_PATH')
	aString += ",PYTHONPATH=%s" %os.getenv('PYTHONPATH')
	aString += ",PATH=%s" %os.getenv('PATH')

	return aString


def checkCommandExistence(command):
	'''constructor
	command(str) -- a command name to be checked.
	Return True(exists)/False(does not exist)
	'''

	# Check the argument.
	if type(command) != str:
		raise TypeError("type of command is %s. But command must be str." %type(command))

	if len(command) == 0:
		raise Exception("The length of command must be > 0.")

	# Check the existence of command
	aCheckProcess = popen2.Popen3("which %s" %command)
	aCheckProcess.wait()
	if aCheckProcess.poll() == 0:
		return True
	else:
		return False

# end of def checkCommandExistence(command):


