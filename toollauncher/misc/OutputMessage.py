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