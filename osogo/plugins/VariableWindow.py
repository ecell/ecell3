#!/usr/bin/env python

import string
import operator

from OsogoPluginWindow import *
from ecell.ecssupport import *

class VariableWindow( OsogoPluginWindow ):

	# ------------------------------------------------------
	# Constructor
	#  [1] Checks this entity have Value, Concentration, 
	#      Fixed Property. 
	#  [2] Checks type of Value and Concentration
	#  [3] Only when this entity has all of them, 
	#      creates this window.
	#
	# aDirName(str)   : directory name that includes glade file
	# aData           : RawFullPN
	# aPluginManager  : the reference to pluginmanager
	# return -> None
	# ------------------------------------------------------
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):
        
		# calls constructor of superclass
		OsogoPluginWindow.__init__( self, aDirName, aData, aPluginManager, aRoot )

		# creates EntityStub
		self.theSession = aPluginManager.theSession
		self.theFullIDString = createFullIDString( self.theFullID() )
		self.theStub = self.theSession.createEntityStub( self.theFullIDString )

		# flags 
		aValueFlag = FALSE
		aConcentrationFlag = FALSE
		aFixedFlag = FALSE

		# --------------------------------------------------------------------
		# [1] Checks this entity have Value, Concentration, Fixed property.
		# --------------------------------------------------------------------
		
		for aProperty in self.theStub.getPropertyList(): # for(1)
			if aProperty == 'Value':
				aValueFlag = TRUE
			elif aProperty == 'Concentration':
				aConcentrationFlag = TRUE
			elif aProperty == 'Fixed':
				aFixedFlag = TRUE
		# end of for(1)

		# If this entity does not have 'Value', does not create instance 
		if aValueFlag == FALSE:
			aMessage = "Error: %s does not have \"Value\" property" %self.theFullIDString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			return None

		# If this entity does not have 'Concentration', does not create instance 
		if aConcentrationFlag == FALSE:
			aMessage = "Error: %s does not have \"Concentration\" property" %self.theFullIDString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			return None

		# If this entity does not have 'Fixed', does not create instance 
		if aFixedFlag == FALSE:
			aMessage = "Error: %s does not have \"Fixed\" property" %self.theFullIDString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			return None


		# --------------------------------------------------------------------
		#  [2] Checks Value and Concentration is Number
		# --------------------------------------------------------------------

		# If Value is not Number
		if operator.isNumberType( self.theStub.getProperty('Value') ):
			pass
		else:
			aMessage = "Error: \"Value\" property is not number" 
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			return None

		# If Concentration is not Number
		if operator.isNumberType( self.theStub.getProperty('Concentration') ):
			pass
		else:
			aMessage = "Error: \"Concentration\" property is not number" 
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			return None

		# --------------------------------------------------------------------
		#  [3] Creates this instance
		# --------------------------------------------------------------------
		# registers this instance to plugin manager
		self.thePluginManager.appendInstance( self )    
		# calls openWindow of superclass
		self.openWindow()

		# adds handers
		self.addHandlers( { \
		                    'button_toggled': self.changeFixFlag,
							'on_value_spinbutton_changed' : self.changeValue,
							'input_value': self.changeValue,
							#'input_concentration': self.inputConcentration,
							'window_exit' : self.exit } )

		# sets FULLID to label 
		self["id_label"].set_text( self.theFullIDString )
		self.update()

	# end of __init__


	# ------------------------------------------------------
	# update
	#  - update all widget
	#
	# return -> None
	# ------------------------------------------------------
	def update( self ):

		self['checkbox'].set_active( self.theStub.getProperty('Fixed') )
		self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )
		self['concentration_entry'].set_text( str(self.theStub.getProperty('Concentration')) )

	# end of update


	# ------------------------------------------------------
	# changeFixFlag
	#  - change fix flag
	#
	# *obj             : dammy objects
	# return -> None
	# ------------------------------------------------------
	def changeFixFlag( self, *obj ):

		self.theStub.setProperty( 'Fixed', self['checkbox'].get_active() )
		self.thePluginManager.updateAllWindows()    

	# end of changeFixFlag


	# ------------------------------------------------------
	# changeValue
	#  - get text form GtkEntryField
	#  - convert text to float
	#  - set the float value to session
	#
	# *obj             : dammy objects
	# return -> None
	# ------------------------------------------------------
	def changeValue( self, obj ):

		# gets text
		aText = string.strip( obj.get_text() )

		# if nothing is written in GtkEntryField, do nothing.
		if aText == '':
			return None

		# if something is written, convert it.
		try:
			aNumber = getFloatFromString( aText )
		except:
			ConfirmWindow(OK_MODE,"Value must be float.","Error!")
			return None

		# if aNumber is None may be
		if aNumber == None:
			pass
		else:
			self.theStub.setProperty( 'Value', aNumber )
			self.thePluginManager.updateAllWindows()    

	# end of changeValue


	# ----------
	# for debug
	# ----------
	def mainLoop():
		# FIXME: should be a custom function
		gtk.mainloop()

