#!/usr/bin/python
# 
# Preferences.py
#

import os

if os.name == 'nt':
	EDITOR_PATH = os.environ['ProgramFiles'] + \
			'\Windows NT\Accessories\wordpad.exe'
else:
	EDITOR_PATH = 'gedit'
PROGRAMS_PATH = os.environ['ECELL3_PREFIX'] + os.sep + 'bin'
SAVE_EM = "1"
SAVE_EML = "1"
MODEL_PATH = os.environ['ECELL3_PREFIX'] + os.sep + 'share' + os.sep + 'doc' + \
        os.sep + 'ecell-' + os.environ['VERSION']
MODEL_HOME = os.environ['ECELL3_PREFIX'] + os.sep + 'share' + os.sep + 'doc' + \
		os.sep + 'ecell-' + os.environ['VERSION'] 
AUTO_LOAD_PREF = "1"
TRANSLATE_EM = "0"

class Preferences:

	# ==========================================================================
	# Constractor
	# ==========================================================================
	def __init__( self, aToolLauncher ):
		"""Constructor 
		"""
		self.theToolLauncher = aToolLauncher
	# end of __init__


	# ==========================================================================
	def sevePrefernces( self ):

		self.theToolLauncher.thePref['save_em']        = SAVE_EM
		self.theToolLauncher.thePref['save_eml']       = SAVE_EML
		self.theToolLauncher.thePref['translate_em']   = TRANSLATE_EM
		self.theToolLauncher.thePref['auto_load_pref'] = AUTO_LOAD_PREF
		self.theToolLauncher.thePref['editor_path']    = EDITOR_PATH
		self.theToolLauncher.thePref['model_home']     = MODEL_HOME
		self.theToolLauncher.thePref['model_path']     = MODEL_PATH
		self.theToolLauncher.thePref['programs_path']  = PROGRAMS_PATH
		self.theToolLauncher.savePreferences()

	# end of sevePrefernces


	# ==========================================================================
	def checkPreferencePash( self ):

		errorFlg = self.checkProgramPath()
		if errorFlg == 0:
			errorFlg = self.checkModelPath()

		return errorFlg

	# end of checkModelHome


	# ==========================================================================
	def checkProgramPath( self ):

		errorMsg = "Please enter the appropriate name of the program path."
		flg = self.checkFilePath( self.theToolLauncher.thePref['programs_path'], "" )
		if flg == 1:
			path = self.theToolLauncher.thePref['ecell3_path']+os.sep+self.theToolLauncher.thePref['programs_path']
			flg = self.checkFilePath( path, "" )

		if flg == 1:
			self.printMessage( errorMsg, errorMsg )

		return flg

	# end of checkProgramPath


	# ==========================================================================
	def checkModelPath( self ):

		if self.theToolLauncher.thePref['model_path'] == "":
			return 0
		else:
			errorMsg = "Please enter the appropriate name of the model path."
			flg = self.checkFilePath( self.theToolLauncher.thePref['model_path'], errorMsg )
			return flg

	# end of checkModelHome


	# ==========================================================================
	def checkFilePath( self, filePath, errorMsg ):

		if os.path.exists( filePath ) == 0:
			self.printMessage( errorMsg, errorMsg )
			return 1
		else:
			return 0

	# end of checkFilePath


	# ==========================================================================
	def printMessage( self, messageBox, errorView ):

		if messageBox != "":
			self.theToolLauncher.printMessage( messageBox )

		if errorView != "":
			self.theToolLauncher.viewErrorMessage( errorView )

	# end of printMessage


	# ==========================================================================
	def getModelHome( self ):
		return self.theToolLauncher.thePref['model_home']

	# end of getModelHome


	# ==========================================================================
	def getModelPath( self ):

		if self.theToolLauncher.thePref['model_path'] == "":
			return  self.theToolLauncher.thePref['ecell3_path']+MODEL_HOME
		elif os.path.exists( self.theToolLauncher.thePref['model_path'] ) == 0:
			return  self.theToolLauncher.thePref['ecell3_path']+os.sep+"work"
		else:
			return self.theToolLauncher.thePref['model_path']

	# end of printMessage


	# ==========================================================================
	def getProgramPath( self ):

		programPath = self.theToolLauncher.thePref['programs_path']
		if os.path.exists( programPath ) == 0:
			programPath = self.theToolLauncher.thePref['ecell3_path']+self.theToolLauncher.thePref['programs_path']
			if os.path.exists( programPath ) == 0:
				return -1

		return programPath

	# end of getProgramPath

# end of Preferences
