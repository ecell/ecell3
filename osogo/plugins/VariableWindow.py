#!/usr/bin/env python

import string
import operator
import os

from OsogoPluginWindow import *
from ecell.ecssupport import *

class VariableWindow( OsogoPluginWindow ):
	"""VariableWindow
	- displays Value, Concentration and Fixed Property.
	- Value and Fixed Property can be changed.
	"""

	# ===============================================================================
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):
		"""Constructor
		[Note]:When the entity has not Value, MolarConc and Fixed Property,
		       throws exception (TypeError).
		"""
        
		# calls constructor of superclass
		OsogoPluginWindow.__init__( self, aDirName, aData, aPluginManager, aRoot )

		# creates EntityStub
		self.theSession = aPluginManager.theSession
		self.theFullIDString = createFullIDString( self.theFullID() )
		self.theStub = self.theSession.createEntityStub( self.theFullIDString )

		# initializes flags for validation of Property
		aValueFlag = FALSE
		aMolarConcFlag = FALSE
		aFixedFlag = FALSE

		# --------------------------------------------------------------------
		# [1] Checks this entity have Value, MolarConc, Fixed property.
		# --------------------------------------------------------------------
		
		for aProperty in self.theStub.getPropertyList(): # for(1)
		
			if aProperty == 'Value':
				aValueFlag = TRUE
			elif aProperty == 'MolarConc':
				aMolarConcFlag = TRUE
			elif aProperty == 'Fixed':
				aFixedFlag = TRUE
		# end of for(1)

		# If this entity does not have 'Value', does not create instance 
		if aValueFlag == FALSE:
			aMessage = "Error: %s does not have \"Value\" property" %self.theFullIDString
			#self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			raise TypeError( aMessage )

		# If this entity does not have 'MolarConc', does not create instance 
		if aMolarConcFlag == FALSE:
			aMessage = "Error: %s does not have \"MolarConc\" property" %self.theFullIDString
			#self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			raise TypeError( aMessage )

		# If this entity does not have 'Fixed', does not create instance 
		if aFixedFlag == FALSE:
			aMessage = "Error: %s does not have \"Fixed\" property" %self.theFullIDString
			#self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			raise TypeError( aMessage )


		# --------------------------------------------------------------------
		#  [2] Checks Value and MolarConc is Number
		# --------------------------------------------------------------------

		# If Value is not Number
		if operator.isNumberType( self.theStub.getProperty('Value') ):
			pass
		else:
			aMessage = "Error: \"Value\" property is not number" 
			#self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			raise TypeError( aMessage )

		# If MolarConc is not Number
		if operator.isNumberType( self.theStub.getProperty('MolarConc') ):
			pass
		else:
			aMessage = "Error: \"MolarConc\" property is not number" 
			#self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow(OK_MODE, aMessage, 'Error!')
			raise TypeError( aMessage )

		# --------------------------------------------------------------------
		#  [3] Creates this instance
		# --------------------------------------------------------------------
		# registers this instance to plugin manager
		self.thePluginManager.appendInstance( self )    


	# ===============================================================================
	def openWindow(self):
		"""overwriets superclass's method
		"""

		# calls superclass's method
		OsogoPluginWindow.openWindow(self)

		# adds handers
		self.addHandlers( { \
		     'on_fix_checkbox_toggled'             : self.changeFixFlag,
		     'on_value_spinbutton_activate'        : self.changeValue,
		     'on_value_spinbutton_focus_out_event' : self.changeValue,
		     'on_value_spinbutton_changed'         : self.changeValueByButton,
		                    })

		# sets FULLID to label 
		self["id_label"].set_text( self.theFullIDString )
                self.setIconList(
			os.environ['OSOGOPATH'] + os.sep + "ecell.png",
			os.environ['OSOGOPATH'] + os.sep + "ecell32.png")
		# sets value to each entry and fix_checkbox
		self.update()


	# ===============================================================================
	def update( self ):
		"""overwrites super class's method
		Returns None
		"""

		# updates 'Value Fxed' check box
		self['fix_checkbox'].set_active( self.theStub.getProperty('Fixed') )

		# updates value entry
		self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )

		# updates concentration entry
		self['concentration_entry'].set_text( str(self.theStub.getProperty('MolarConc')) )


	# ===============================================================================
	def changeFixFlag( self, *arg ):
		"""when 'Value Fixed' checkbox is toggled, this method is called.
		Returns None
		"""

		# set 'Fixed' property with the status of toggel button
		self.theStub.setProperty( 'Fixed', self['fix_checkbox'].get_active() )

		# updates plugin manager
		self.thePluginManager.updateAllWindows()    


	# ===============================================================================
	def changeValue( self, *arg ):
		"""When enter is pressed on value entry, this method is called.
		Return None
		"""

		# gets text
		aText = string.strip( self['value_spinbutton'].get_text() )

		# The following 2 lines are needed for initialize this window.
		# When openWindow is called, this method must be called but value_spinbutton
		# is empty. The validation of it should be ignored.
		if aText == '':
			return None

		# Convert inputted text to number
		try:
			aNumber = string.atof(aText)

		# When it is not number
		except:
			self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )

			# displays confirm window
			ConfirmWindow(OK_MODE,"Input number.","Value Error!")

		# When it is number
		else:

			# sets value
			self.theStub.setProperty( 'Value', aNumber )

			# updates all window
			# not only plugin managers, but also EntityWindow
			self.thePluginManager.updateAllWindows()    


	# ===============================================================================
	def changeValueByButton( self, *arg ):
		"""When a button of value spinbutton is pressed, this method is called.
		Return None
		"""

		# converts inputted text into number.
		# When it is number, sets it to EntityStub.
		try:
			aText = string.strip( self['value_spinbutton'].get_text() )

			# The following 2 lines are needed for initialize this window.
			# When openWindow is called, this method must be called but value_spinbutton
			# is empty. The validation of it should be ignored.
			if len(aText) == 0:
				return None

			aNumber = string.atof(aText)

			self.theStub.setProperty( 'Value', aNumber )
			self.thePluginManager.updateAllWindows()    

		# When it is not number, sets previous value to value spinbutton.
		except:
			self['value_spinbutton'].set_text( str(self.theStub.getProperty('Value')) )


# end of class VariableWindow


