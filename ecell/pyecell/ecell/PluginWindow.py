#!/usr/bin/env python


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


from Window import *

import string
import sys
from ecell.ecssupport import *

# ---------------------------------------------------------------
# PluginWindow --> Window
#   - has some plugin functions
# ---------------------------------------------------------------
class PluginWindow( Window ):


	# ---------------------------------------------------------------
	# Constructor
	#   - sets glade file
	#   - sets root property
	#   - call openwindow method of this class
	#  
	# In the constructor sub class, you have to call openWindow().
	#
	# aDirName       : directory name
	# aPluginManager : reference to plugin manager
	# aRoot          : root property
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aDirname, aPluginManager, aRoot=None ):

		self.theRoot = aRoot
		self.theClassName = self.__class__.__name__
		aGladeFileName = os.path.join( aDirname , self.theClassName + ".glade" )

		self.theGladeFile = aGladeFileName
		self.thePluginManager = aPluginManager

	# end of __init__


	# ---------------------------------------------------------------
	# openWindow
	#   - call openwindow method of super class
	#  
	#
	# aDirName       : directory name
	# aPluginManager : reference to plugin manager
	# aRoot          : root property
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openWindow( self ):
		Window.openWindow( self )
		self.addHandlers( {'window_exit' : self.exit,} )

	# end of openWindow

	# ---------------------------------------------------------------
	# update (abstract method )
	#   - Sub class must override this method.
	# ---------------------------------------------------------------
	def update( self ):

		import inspect
		caller = inspect.getouterframes(inspect.currentframe())[0][3]
		raise NotImplementedError(caller + 'must be implemented in subclass')

	# end of update

	# ---------------------------------------------------------------
	# exit 
	#   - remove this window from PluginManager
	#
	# *objects : dammy elements of argument
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def exit( self, *objects ):

		self.thePluginManager.removeInstance( self )
	
	# end of exit


# end of PluginWindow

