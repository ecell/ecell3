#!/usr/bin/python
#
# CompileCpp.py


from ToolLauncherOkCancel import *

import os
import os.path
import string
import re
import tempfile
from Preferences import *

MSG_NO_SUCH_CPPFILE         = "Process file does not exist."

# ==========================================================================
# CopyDirectory Class
# Author T.Itaba
# ==========================================================================
class CompileCpp:

	# ==========================================================================
	# Constructor
	# ==========================================================================
	def __init__( self, aToolLauncher ):
		"""Constructor 
		"""
		self.theToolLauncher = aToolLauncher

	# end of __init__


	# ==========================================================================
	def compile( self, src ):

                errorMsg = MSG_NO_SUCH_CPPFILE
                programPath = self.getProgramPath()
                if src == "":
                        src = os.path.join(
                                self.theToolLauncher.thePref['models_path'], self.theToolLauncher.thePref['current_model'] )
                if os.path.exists(src):
                        if os.path.isdir(src):
                                srcList = os.listdir( src )
                                for srcFile in srcList:
                                        ( cpppath, cppfilename ) = os.path.split( srcFile )
                                        if len(re.split( '[\W]+',cppfilename, 1)) == 2:
                                                if re.split( '[\W]+',cppfilename, 1)[1] == 'cpp':
                                                        errorMsg = ""
                                                        msg = self.compileCpp( os.path.join( src, cppfilename ) )
                                                        if msg != "":
                                                                errorMsg = msg
                        else:
                                errorMsg = self.compileCpp( src )

                        if errorMsg == "":
                                msg = "Compilation of \""+src+"\" was successful."
                        elif errorMsg == MSG_NO_SUCH_CPPFILE:
                                msg = MSG_NO_SUCH_CPPFILE
                        else:
                                msg = "Compilation of \""+src+"\" has failed."

                        self.printMessage( "", msg )

                else:
                        msg = "\""+src+ "\" does not exist.\nPlease enter the appropriate name of the file."
                        self.printMessage( msg, msg )

	# end of putMessage


	# ==========================================================================
	def compileCpp( self, cppfilename ):

		programpath = self.getProgramPath()
		cmd = programpath+' \"'+cppfilename+'\"'
		defaultpath = os.getcwd()

		if self.theToolLauncher.thePref['models_path'] == "":
			os.chdir( self.theToolLauncher.thePref['ecell3_path']+os.sep+'share' )
		else:
			os.chdir( os.path.join(
				self.theToolLauncher.thePref['models_path'], self.theToolLauncher.thePref['current_model'] ) )

		TMPDIR = tempfile.gettempdir()
		ret = os.system( cmd+" 2> \"" + TMPDIR + os.sep + "error.log\" > \"" + TMPDIR + os.sep + "cpp.log\"" )
		errorMsg = self.checkLog( )

		if not ret == 0:
			self.printMessage( errorMsg, "" )
			msg = "Compilation of \""+cppfilename+"\" has failed."
			
			self.printMessage( msg, "" )
		else:
			msg = "Compilation of \""+cppfilename+"\" was successful."
			self.printMessage( msg, "" )

		os.chdir( defaultpath )

		return errorMsg

	# end of compile


	# ==========================================================================
	def checkLog( self ):

		TMPDIR = tempfile.gettempdir()
		logdata = self.readSystemOut( TMPDIR + os.sep + "cpp.log" )
		errorMsg = re.search( 'Error', logdata )

		self.printMessage( logdata, "")
		logdata = self.readSystemOut( TMPDIR + os.sep + "error.log" )
		self.printMessage( logdata, "")

		if errorMsg == None:
			return ""
		else:
			return logdata

	# end of compile


	# ==========================================================================
	def readSystemOut( self, fileName ):

		outputFile = open( fileName,'r' )
		fileData = outputFile.read()
		outputFile.close()
		os.remove( fileName )
		return fileData

	# end of compile


	# ==========================================================================
	def getProgramPath( self ):
		# don't use the full path of ecell3-dmc because E-Cell BINDIR is
		# already in the $PATH. Full path with whitespaces will cause 
		# problems on MS-Windows
		programPath = 'ecell3-dmc'
		return programPath
	# end of getProgramPath


	# ==========================================================================
	def printMessage( self, windowMessage, boxMessage ):

		if windowMessage != "":
			self.theToolLauncher.printMessage( windowMessage )

		if boxMessage != "":
			self.theToolLauncher.viewErrorMessage( boxMessage )

	# end of printMessage

# end of CompileCpp
