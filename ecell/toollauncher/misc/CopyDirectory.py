#!/usr/bin/python
#
# CopyDirectory.py
#

import os
import ToolLauncher
import traceback
import re

from ToolLauncherOkCancel import *
from shutil import *
from fileFind import *

# ==========================================================================
# CopyDirectory Class
# Auther T.Itaba
# ==========================================================================
class CopyDirectory:

	# ==========================================================================
	# Constractor
	# ==========================================================================
	def __init__( self, aToolLauncher, newDir, symlinks=0 ):

		"""Constructor 
		"""
		self.theToolLauncher = aToolLauncher
		self.theNewDir = newDir
		self.theSymlinks = symlinks

	# end of __init__


	# ==========================================================================
	def copyDirectory( self ):

		if os.path.exists( self.theNewDir ):
			self.theToolLauncher.viewErrorMessage( 'Please enter the appropriate name.' )
		else:
			os.mkdir( self.theNewDir )
			msg = "A new model \""+self.theNewDir+"\" has been created."
			self.theToolLauncher.printMessage( msg )
			self.theToolLauncher.viewErrorMessage( msg )

	# end of copyDirectory

# end of CopyDirectory


