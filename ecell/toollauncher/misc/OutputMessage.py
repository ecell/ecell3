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
#!/usr/bin/python
#
# OutputMessage.py

from ToolLauncherOkCancel import *

import os
import os.path
import string
import re


# ==========================================================================
# CopyDirectory Class
# Auther T.Itaba
# ==========================================================================
class OutputMessage:

	# ==========================================================================
	# Constractor
	# ==========================================================================
	def __init__( self, aToolLauncher, fileName, msg ):
		"""Constructor 
		"""

		self.theToolLauncher = aToolLauncher
		self.theFileName = fileName
		self.theMessage  = msg

	# end of __init__


	# ==========================================================================
	def putMessage( self ):

		( filePath, fileName ) = os.path.split( self.theFileName )

		if fileName == "":
			( modelPath, modelName ) = os.path.split( filePath )
			self.theFileName = filePath+os.sep+modelName+".log"

		if os.path.isfile( self.theFileName ) == True:
			self.viewDialog()
		else:
			self.createFile()

	# end of putMessage


	# ==========================================================================
	def viewDialog( self ):

		self.thePrefWindow = ToolLauncherOkCancel( self )
		errorMsg = self.theFileName+' is exist.\n'
		errorMsg += 'Does it overwrite?'
		self.thePrefWindow.openWindow( errorMsg )

	# end of viewDialog

	# ==========================================================================
	def onOk( self, *arg ):
		self.createFile()
	# end of onOk

	# ==========================================================================
#	def onCancel( self, *arg ):
	# end of onCancel


	# ==========================================================================
	def createFile( self ):
		file = open( self.theFileName, 'w' )
		file.write( self.theMessage )
		file.close()

	# end of putMessage

# end of CopyDirectory