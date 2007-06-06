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
# ZipManager.py  - to start up E-Cell3 Tool Launcher
#

import os
import string
import traceback
import sys
import re
import tempfile
from Preferences import *

ERROR_MSG1 = "\" already exists. Please enter the appropriate name of the file."
ERROR_MSG2 = "\" already exists. Please enter the appropriate name of the directory."
ERROR_MSG3 = "\" does not exist. Please enter the appropriate name of the or file."
EXTENSION_EM = ".em"

class Eml:

	# ==========================================================================
	# Constractor
	# ==========================================================================
	def __init__( self, aToolLauncher ):
		"""Constructor 
		"""
		self.theToolLauncher = aToolLauncher
		self.theFileSetFlag  = False
	# end of __init__


	# ==========================================================================
	def compileEM( self, emFile, emlFile ):

		if self.checkExpension( emFile, "em" ) == False:
			msg = "Please enter the appropriate EM file name."
			self.printMessage( msg, msg )
		else:
			emFile  = self.getCompileFile( emFile )
			if emFile != -1:
				emlFile = self.getNewFileEML( emFile, emlFile )
				if emlFile != -1:
					self.compile( emFile, emlFile, "ecell3-em2eml" )
					if self.theToolLauncher.thePref['save_em'] == '0':
						self.deleteFile( emFile )

	# end of compileEM


	# ==========================================================================
	def compileEML( self, emlFile, emFile ):

		if self.checkExpension( emlFile, "eml" ) == False:
			msg = "Please enter the appropriate EML file name."
			self.printMessage( msg, msg )
		else:
			emlFile = self.getCompileFile( emlFile )
			if emlFile != -1:
				emFile  = self.getNewFileEM( emFile, emlFile )
				if emFile != -1:
					self.compile( emlFile, emFile,  "ecell3-eml2em" )
					if self.theToolLauncher.thePref['save_eml'] == '0':
						self.deleteFile( emlFile )

	# end of compileEM


	# ==========================================================================
	def compile( self, file1, file2, programFile ):

		errorMsg = ""

		try:
			
			cmdstr = programFile+' -o \"'+file2+'\" \"'+file1+'\"'
			TMPDIR = tempfile.gettempdir()
			ret = self.execute( cmdstr+" > " + TMPDIR + os.sep + "eml.log 2> " + TMPDIR + os.sep + "emlError.log" )
			self.checkCompile( ret, file1, file2)
		except:
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage(anErrorMessage, 
			                  "Conversion from \""+file1+"\" to\n\""+file2+"\" has failed.")
		
		return errorMsg

	# end of compile

	# ==========================================================================
	def execute( self, cmdstr ):

		return os.system( cmdstr )

	# end of execute

	# ==========================================================================
	def errorMsgNotExistFile( self, fileName ):

		return fileName+" does not exist. Please enter the appropriate name of the file."

	# end of errorMsgNotExistFile


	# ==========================================================================
	def errorMsgNotExistDir( self, dirName ):

		return dirName+" does not exist. Please enter the appropriate name of the directory."

	# end of errorMsgNotExistFile


	# ==========================================================================
	def deleteFile( self, fileName ):

		if os.path.isfile( fileName ):
			os.remove( fileName )

	# end of deleteFile


	# ==========================================================================
	def checkCompile( self, ret, file1, file2 ):

                TMPDIR = tempfile.gettempdir()
		logfile = open( TMPDIR + os.sep + 'eml.log','r' )
		logdata = logfile.read()
		logfile.close()
		errorfile = open( TMPDIR + os.sep + 'emlError.log','r' )
		errordata = errorfile.read()
		errorfile.close()
		os.remove( TMPDIR + os.sep + 'eml.log' )
		os.remove( TMPDIR + os.sep + 'emlError.log' )
		errorMsg = re.search( 'error', logdata )
		if ret == 0:
			msg = "Conversion from \""+ file1+"\" to\n\""+file2+"\" was successful."
			self.theToolLauncher.printMessage( msg )
			self.theToolLauncher.viewErrorMessage( msg )
		else:
			self.theToolLauncher.printMessage( logdata )
			self.theToolLauncher.printMessage( errordata )
			msg = "Conversion from \""+ file1+"\" to\n\""+file2+"\" has failed."
			self.theToolLauncher.printMessage( msg )
			self.theToolLauncher.viewErrorMessage( msg )

	# end of checkCompile


	# ==========================================================================
	def printMessage( self, messageWindowMsg, dialogMsg ):

		if messageWindowMsg != "":
			self.theToolLauncher.printMessage( messageWindowMsg )

		if dialogMsg != "":
			self.theToolLauncher.viewErrorMessage( dialogMsg )

	# end of printMessage


	# ==========================================================================
	def checkExpension( self, fileName, expansion ):

		( path, name ) = os.path.split( fileName )
		if len(re.split( '[\W]+',name, 1)) == 2:
			fileName = name

		if len(re.split( '[\W]+',fileName, 1)) == 2:
			if re.split( '[\W]+',fileName, 1)[1] == expansion:
				return True
			else:
				return False
		else:
			return False

	# end of checkExpansion


	# ==========================================================================
	def getCompileFile( self, compileFile ):

		if os.path.exists( compileFile ) == True:
			self.theFileSetFlag = False
			return compileFile
		else:
			if os.path.exists(os.path.join( self.getCurrentModelPath(), compileFile ) ):
				compileFile = os.path.join( self.getCurrentModelPath(), compileFile )
			else:
				msg = "\""+compileFile+"\" does not exist. Please enter the appropriate name of the file."
				self.printMessage( msg, msg )
				return -1

		return compileFile

	# end of getCompileFileEM


	# ==========================================================================
	def getNewFileEML( self, emFile, emlFile ):

		( emPath, emName ) = os.path.split( emFile )
		newFileName = -1
		if emlFile == "":
			newFileName = os.path.join( self.getCurrentModelPath(), emName+"l" )
		else:
			if os.path.exists( emlFile ):
				if emlFile[-1] == os.sep:
					if os.path.isdir( emlFile ):
						newFileName = emlFile+emName+"l"
					else:
						self.printMessage( "\""+emlFile+ERROR_MSG1, "\""+emlFile+ERROR_MSG1 )
						newFileName = -1
				else:
					if os.path.isfile( emlFile ):
						newFileName = emlFile
					else:
						self.printMessage( "\""+emlFile+ERROR_MSG2, "\""+emlFile+ERROR_MSG2 )
						newFileName = -1
			else:
				if os.path.exists( os.path.join( self.getCurrentModelPath(), emlFile ) ):
					if emlFile[-1] == os.sep:
						newFileName = os.path.join( self.getCurrentModelPath(), emlFile )+emName+"l"
					else:
						newFileName = os.path.join( self.getCurrentModelPath(), emlFile )
				else:
					( emlPath, emlName ) = os.path.split( emlFile )
					if os.path.exists( os.path.join( self.getCurrentModelPath(), emlPath ) ):
						if emlFile[-1] == os.sep:
							newFileName = os.path.join( self.getCurrentModelPath(), emlFile )+emName+"l"
						else:
							newFileName = os.path.join( self.getCurrentModelPath(), emlFile )
					else:
						msg = "\""+emlFile+ERROR_MSG3
						self.printMessage( msg, msg )
						newFileName = -1

		return newFileName

	# end of getNewFileEML


	# ==========================================================================
	def getNewFileEM( self, emFile, emlFile ):

		( emlPath, emlName ) = os.path.split( emlFile )
		newName = re.split( '[\W]+',emlName, 1)[0]
		newFileName = -1
		if emFile == "":
			newFileName = os.path.join( self.getCurrentModelPath(), newName+EXTENSION_EM )
		else:
			if os.path.exists( emFile ):
				if emFile[-1] == os.sep:
					if os.path.isdir( emFile ):
						newFileName = emFile+newName+EXTENSION_EM
					else:
						self.printMessage( "\""+emFile+ERROR_MSG1, "\""+emFile+ERROR_MSG1 )
						newFileName = -1
				else:
					if os.path.isfile( emFile ):
						newFileName = emFile
					else:
						self.printMessage( "\""+emFile+ERROR_MSG2, "\""+emFile+ERROR_MSG2 )
						newFileName = -1
			else:
				if os.path.exists( os.path.join( self.getCurrentModelPath(), emFile ) ):
					if emFile[-1] == os.sep:
						newFileName = os.path.join( self.getCurrentModelPath(), emFile )+newName+EXTENSION_EM
					else:
						newFileName = os.path.join( self.getCurrentModelPath(), emFile )
				else:
					( emPath, emName ) = os.path.split( emFile )
					if os.path.exists( os.path.join( self.getCurrentModelPath(), emPath ) ):
						if emFile[-1] == os.sep:
							newFileName = os.path.join( self.getCurrentModelPath(), emFile )+newName+EXTENSION_EM
						else:
							newFileName = os.path.join( self.getCurrentModelPath(), emFile )
					else:
						msg = "\""+emFile+ERROR_MSG3
						self.printMessage( msg, msg )
						newFileName = -1
		return newFileName


	# ==========================================================================
	def getCurrentModelPath( self ):

		pref = Preferences( self.theToolLauncher )
		return pref.getCurrentModelPath()

	# end of getCurrentModelPath


	# ==========================================================================
	def printMessage( self, windowMessage, boxMessage ):

		if windowMessage != "":
			self.theToolLauncher.printMessage( windowMessage )

		if boxMessage != "":
			self.theToolLauncher.viewErrorMessage( boxMessage )

	# end of printMessage

# end of Eml
