#!/usr/bin/python
# 
# Preferences.py
#

import os

SAVE_EM = "1"
SAVE_EML = "1"
AUTO_LOAD_PREF = "1"
TRANSLATE_EM = "0"
if os.name == 'nt':
	EDITOR_PATH = os.path.join(
			os.environ['SystemRoot'], 'system32', 'notepad.exe' )
else:
	EDITOR_PATH = os.path.join( os.sep, 'usr', 'bin', 'gedit' )
# models_path is the base directory of model instances
# models_path must exist before the Toollauncher can function
MODELS_PATH = os.path.join( os.environ['ECELL3_PREFIX'], 'share' )
# current_model is the name the current working model
CURRENT_MODEL = 'ecell'


class Preferences:

	# ==========================================================================
	# Constructor
	# ==========================================================================
	def __init__( self, aToolLauncher ):
		"""Constructor 
		"""
		self.theToolLauncher = aToolLauncher
	# end of __init__


	# ==========================================================================
	def savePreferences( self ):

		self.theToolLauncher.thePref['save_em']        = SAVE_EM
		self.theToolLauncher.thePref['save_eml']       = SAVE_EML
		self.theToolLauncher.thePref['translate_em']   = TRANSLATE_EM
		self.theToolLauncher.thePref['auto_load_pref'] = AUTO_LOAD_PREF
		self.theToolLauncher.thePref['editor_path']    = EDITOR_PATH
		self.theToolLauncher.thePref['models_path']    = MODELS_PATH
		self.theToolLauncher.thePref['current_model']  = CURRENT_MODEL
		self.theToolLauncher.savePreferences()

	# end of savePreferences


	# ==========================================================================
	def isPrefOK( self ):

                if not self.isEditorPathOK():
                        return False
                elif not self.isModelsPathOK():
                        return False
                elif not self.isCurrentModelOK():
                        return False
                else:
                        return True

	# end of isPrefOK


	# ==========================================================================
	def isEditorPathOK( self ):

                if not os.path.isfile( self.theToolLauncher.thePref['editor_path'] ):
                        errorMsg = "Please enter a valid path for your favorite text editor."
			self.printMessage( errorMsg, errorMsg )
                        return False 
                else:
                        return True

	# end of isEditorPathOK


	# ==========================================================================
	def isModelsPathOK( self ):

                if self.theToolLauncher.thePref['models_path'] == '' or not os.path.isdir( self.theToolLauncher.thePref['models_path'] ):
			errorMsg = "Please enter a valid base directory of the models that you will create."
			self.printMessage( errorMsg, errorMsg )
                        return False
			
		else:
			return True

	# end of isModelsPathOK


	# ==========================================================================
	def isCurrentModelOK( self ):

                currentModelPath = os.path.join(
                                self.theToolLauncher.thePref['models_path'],
                                self.theToolLauncher.thePref['current_model'] )

                if self.theToolLauncher.thePref['current_model'] == '':
                        errorMsg = "Please enter a valid current model name."
			self.printMessage( errorMsg, errorMsg )
                        return False			
                elif not os.path.isdir( currentModelPath ):
                        errorMsg = "The directory " + currentModelPath + \
                            " for current model does not exist.\n Please create it using the Folder... button."
			self.printMessage( '', errorMsg )
                        return False			
		else:
			return True

	# end of isCurrentModelOK

        
	# ==========================================================================
	def printMessage( self, messageBox, errorView ):

		if messageBox != "":
			self.theToolLauncher.printMessage( messageBox )

		if errorView != "":
			self.theToolLauncher.viewErrorMessage( errorView )

	# end of printMessage


	# ==========================================================================
	def getModelsPath( self ):
		if self.theToolLauncher.thePref['models_path'] == "" or not os.path.isdir( self.theToolLauncher.thePref['models_path'] ) :
			return  MODELS_PATH
		else:
			return self.theToolLauncher.thePref['models_path']

	# end of getCurrentModelPath

	# ==========================================================================

        
	def getCurrentModelPath( self ):
		if self.theToolLauncher.thePref['models_path'] == "" or not os.path.isdir( self.theToolLauncher.thePref['models_path'] ) :
			return  MODELS_PATH
		elif not os.path.isdir( os.path.join(
		    self.theToolLauncher.thePref['models_path'], self.theToolLauncher.thePref['current_model']) ):
			return  MODELS_PATH
		else:
			return os.path.join( self.theToolLauncher.thePref['models_path'],
				self.theToolLauncher.thePref['current_model']  )

	# end of getCurrentModelPath


# end of Preferences
