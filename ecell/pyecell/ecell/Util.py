#! /usr/bin/env python

'''
A module for session manager
 - defines constants
 - provides general methods

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
Copyright (C) 1996-2003 Keio University.
Send feedback to Kouichi Takahashi <shafi@e-cell.org>'''\
% ( ecell.ecs.getLibECSVersion(), sys.version_info[0], sys.version_info[1], sys.version_info[2] )


SYSTEM_PROXY = 'SystemProxy'
SESSION_PROXY = 'SessionProxy'

DEFAULT_TMP_DIRECTORY = 'tmp'
DEFAULT_ENVIRONMENT = 'Local'



# job status
QUEUED 		= 0
RUN			= 1
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



