#! /usr/bin/env python

'''
A module for session manager
 - privides API depending on environment


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

# imports standard modules
import sys
import string
import os
import time


# imports ecell modules
from ecell.Util import *
from ecell.SessionProxy import *


class SystemProxy:
	'''SystemProxy abstract class
	'''

	def __init__(self, sessionmanager ):
		'''constructor
		sessionmanager -- a reference to SessionManager
		'''

		self.theSessionManager = sessionmanager


	def getDefaultConcurrency(self):
		'''ABSTRACT : This method must be overwrote in subclass
		returns default cpus
		Return int : the number of cpus
		raise NotImplementedError
        '''

		# When this method is not implemented in sub class,
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')


	def update(self):
		'''ABSTRACT : This method must be overwrote in subclass
		updates all SessionProxys' status
		Return None
		raise NotImplementedError
        '''
        
		# When this method is not implemented in sub class,
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



	def createSessionProxy(self):
		'''ABSTRACT : This method must be overwrote in subclass
		creates and returns new SessionProxy instance
		Return SessionProxy
		raise NotImplementedError
		'''

		# When this method is not implemented in sub class,
		# raise NotImplementedError
		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + ' must be implemented in subclass')



# end of class SystemProxy



