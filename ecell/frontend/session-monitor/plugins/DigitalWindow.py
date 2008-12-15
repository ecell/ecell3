#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

import os
import operator

from ecell.ecssupport import *

import ecell.ui.osogo.ConfirmWindow
import ecell.ui.osogo.config as config
from ecell.ui.osogo.OsogoPluginWindow import *

# ------------------------------------------------------
# DigitalWindow -> OsogoPluginWindow
#   - show one numerical property 
# ------------------------------------------------------
class DigitalWindow( OsogoPluginWindow ):

	# ------------------------------------------------------
	# Constructor
	# 
	# aDirName(str)   : directory name that includes glade file
	# data            : RawFullPN
	# aPluginManager  : the reference to pluginmanager 
	# return -> None
	# ------------------------------------------------------
	def __init__( self, aDirName, aData, aPluginManager, aRoot=None ):

		# call constructor of superclass
		OsogoPluginWindow.__init__( self, aDirName, aData, \
		                            aPluginManager, aRoot )
        
		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		if operator.isNumberType( aValue ) == FALSE:
			aMessage = "Error: (%s) is not numerical data" %aFullPNString
			self.thePluginManager.printMessage( aMessage )
			aDialog = ConfirmWindow.ConfirmWindow(0,aMessage,'Error!')
			raise TypeError( aMessage )

	# end of __init__

	def openWindow(self):

		OsogoPluginWindow.openWindow(self)

		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		anAttribute = self.theSession.theSimulator.getEntityPropertyAttributes( \
		              aFullPNString )

		self.thePluginManager.appendInstance( self )

		self.addHandlers( { 
		            'on_value_frame_activate'        :self.inputValue,
	  	            'on_increase_button_clicked'     :self.increaseValue,
	  	            'on_decrease_button_clicked'     :self.decreaseValue,
		            } )

		aString = str( self.theFullPN()[ID] )
		aString += ':\n' + str( self.theFullPN()[PROPERTY] )
		self["id_label"].set_text( aString )

		# If this property is not settable, sets unsensitive TextEntry and Buttons.
		if anAttribute[SETTABLE] == FALSE:
			self["value_frame"].set_editable( FALSE )
			self["increase_button"].set_sensitive( FALSE )
			self["decrease_button"].set_sensitive( FALSE )
                self.setIconList(
			os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
			os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )
		self.update()

	# ------------------------------------------------------
	# changeFullPN
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def changeFullPN( self, anObject ):

		OsogoPluginWindow.changeFullPN( self, anObject )

		aString = str( self.theFullPN()[ID] )
		aString += ':\n' + str( self.theFullPN()[PROPERTY] )
		self["id_label"].set_text( aString )

	# end of changeFullPN


	# ------------------------------------------------------
	# update
	# 
	# return -> None
	# ------------------------------------------------------
	def update( self ):

		aFullPNString = createFullPNString( self.theFullPN() )
		aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
		self["value_frame"].set_text( str( aValue ) )

	# end of update


	# ------------------------------------------------------
	# inputValue
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def inputValue( self, *arg ):

		# gets text from text field.
		aText = self['value_frame'].get_text().split()
		if type(aText) == type([]):
			if len(aText) > 0:
				aText = aText[0]
			else:
				return None
		else:
			return None

		# Only when the length of text > 0,
		# checks type of text and set it.
		if len(aText)>0:
			# Only when the value is numeric, 
			# the value will be set to value_frame.
			try:
				aValue = float( aText )
				self.setValue( self.theFullPN(), aValue )
			except:
				ConfirmWindow.ConfirmWindow(0,'Input numerical value.')
				aValue = self.getValue( self.theFullPN() )
				self["value_frame"].set_text( str( aValue ) )
			return None
		else:
			return None

	# end of inputValue


	# ------------------------------------------------------
	# increaseValue
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def increaseValue( self, obj ):

		if self.getValue( self.theFullPN() ):
			self.setValue( self.theFullPN(), self.getValue( self.theFullPN() ) * 2.0 )
		else:
			self.setValue( self.theFullPN(), 1.0 )

	# end of increaseValue
		
        
	# ------------------------------------------------------
	# decreaseValue
	# 
	# anObject(any)   : a dammy object
	# return -> None
	# ------------------------------------------------------
	def decreaseValue( self, obj ):

		self.setValue( self.theFullPN(), self.getValue( self.theFullPN() ) * 0.5 )

	# end of decreaseValue
			
# end of DigitalWindow


### test code

if __name__ == "__main__":

    class simulator:

        dic={('Variable', '/CELL/CYTOPLASM', 'ATP','Value') : (1950,),}

        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]

        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Variable','/CELL/CYTOPLASM','ATP','Value')

    def mainQuit( obj, aData ):
        gtk.main_quit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.main()

    def main():
        aDigitalWindow = DigitalWindow( 'plugins', simulator(), [fpn,] )
        #aDigitalWindow.addHandler( 'gtk_main_quit', mainQuit )
        aDigitalWindow.addHandler( { 'gtk_main_quit' : mainQuit } )
        aDigitalWindow.update()

        mainLoop()

    main()









