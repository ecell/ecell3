#!/usr/bin/python
#
# ZipManager.py  - to start up E-Cell3 Tool Launcher
#

import os
import string
import traceback
import sys
import re
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

		if self.theToolLauncher.thePref['programs_path'] == "":
			msg = "Please specify the program path on the preference window."
			self.printMessage( msg, msg )
		elif self.checkExpension( emFile, "em" ) == False:
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

		if self.theToolLauncher.thePref['programs_path'] == "":
			msg = "Please specify the program path on the preference window."
			self.printMessage( msg, msg )
		elif self.checkExpension( emlFile, "eml" ) == False:
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

		compileFlg = 0
		errorMsg = ""

		flg = self.execute( file2, file1, programFile )
		if flg == "0":
			self.checkCompile( file1, file2 )

		return errorMsg

	# end of compile


	# ==========================================================================
	def execute( self, file1, file2, programFile ):

		try:
			programPath = self.getProgramPath( programFile )
			if programPath == None:
				msg = "\""+self.theToolLauncher.thePref['programs_path']+"\" does not exist.\nPlease enter the appropriate name of the directory."
				self.printMessage( msg, msg )
			else:
				cmdstr = 'python '+'\"'+programPath+'\" -o \"'+file1+'\" \"'+file2+'\"'
				self.theToolLauncher.execute( cmdstr+" > eml.log 2> emlError.log" )
				return "0"

		# catch exceptions
		except:
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage(anErrorMessage, 
			                  "Conversion from \""+file2+"\" to\n\""+file1+"\" has failed.")

	# end of execute


	# ==========================================================================
	def getProgramPath( self, programFile ):

		if os.path.exists( self.theToolLauncher.thePref['programs_path'] ):
			return self.theToolLauncher.thePref['programs_path']+os.sep+programFile
		else:
			if os.path.exists( self.theToolLauncher.thePref['ecell3_path']+os.sep+"bin" ):
				return self.theToolLauncher.thePref['ecell3_path']+os.sep+"bin"+os.sep+programFile
			else:
				return None

	# end of getProgramPath


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
	def checkCompile( self, file1, file2 ):

		logfile = open( 'eml.log','r' )
		logdata = logfile.read()
		logfile.close()
		os.remove( 'eml.log' )
		os.remove( 'emlError.log' )
		errorMsg = re.search( 'error', logdata )
		if errorMsg == None:
			msg = "Conversion from \""+ file1+"\" to\n\""+file2+"\" was successful."
			self.theToolLauncher.printMessage( msg )
			self.theToolLauncher.viewErrorMessage( msg )
		else:
			self.theToolLauncher.printMessage( logdata )
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
			if os.path.exists(os.path.join( self.getModelPath(), compileFile ) ):
				compileFile = os.path.join( self.getModelPath(), compileFile )
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
			newFileName = os.path.join( self.getModelPath(), emName+"l" )
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
				if os.path.exists( os.path.join( self.getModelPath(), emlFile ) ):
					if emlFile[-1] == os.sep:
						newFileName = os.path.join( self.getModelPath(), emlFile )+emName+"l"
					else:
						newFileName = os.path.join( self.getModelPath(), emlFile )
				else:
					( emlPath, emlName ) = os.path.split( emlFile )
					if os.path.exists( os.path.join( self.getModelPath(), emlPath ) ):
						if emlFile[-1] == os.sep:
							newFileName = os.path.join( self.getModelPath(), emlFile )+emName+"l"
						else:
							newFileName = os.path.join( self.getModelPath(), emlFile )
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
			newFileName = os.path.join( self.getModelPath(), newName+EXTENSION_EM )
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
				if os.path.exists( os.path.join( self.getModelPath(), emFile ) ):
					if emFile[-1] == os.sep:
						newFileName = os.path.join( self.getModelPath(), emFile )+newName+EXTENSION_EM
					else:
						newFileName = os.path.join( self.getModelPath(), emFile )
				else:
					( emPath, emName ) = os.path.split( emFile )
					if os.path.exists( os.path.join( self.getModelPath(), emPath ) ):
						if emFile[-1] == os.sep:
							newFileName = os.path.join( self.getModelPath(), emFile )+newName+EXTENSION_EM
						else:
							newFileName = os.path.join( self.getModelPath(), emFile )
					else:
						msg = "\""+emFile+ERROR_MSG3
						self.printMessage( msg, msg )
						newFileName = -1
		return newFileName


	# ==========================================================================
	def getModelPath( self ):

		pref = Preferences( self.theToolLauncher )
		return pref.getModelPath()

	# end of getModelPath


	# ==========================================================================
	def printMessage( self, windowMessage, boxMessage ):

		if windowMessage != "":
			self.theToolLauncher.printMessage( windowMessage )

		if boxMessage != "":
			self.theToolLauncher.viewErrorMessage( boxMessage )

	# end of printMessage

# end of Eml
