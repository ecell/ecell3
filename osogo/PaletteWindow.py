#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


from config import *

from main import *
from gtk import *
import os
import re
import string


# ---------------------------------------------------------------
# PaletteWindow -> GtkWindow
#   - manages PaletteWindow
# ---------------------------------------------------------------
class PaletteWindow(GtkWindow):

	# ---------------------------------------------------------------
	# Constructor
	#   - creates MessageWindow
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self ):

		GtkWindow.__init__( self, WINDOW_TOPLEVEL )

		self.theToolbar = GtkToolbar( ORIENTATION_VERTICAL, TOOLBAR_BOTH )
		self.add( self.theToolbar )
		self.set_data('toolbar', self.theToolbar)
		self.connect('destroy', self.destroy)
		self.theExist = 1

	# end of __init__

	# ---------------------------------------------------------------
	# setPluginList
	#
	# pluginList : plugin list
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setPluginList( self, pluginlist ):

		aPluginNameList = []
		aIndicator = 0
        
		if not pluginlist:
			return
        
		for aModule in pluginlist.values():

			aModuleName = aModule.theName
			aButtonName = string.replace( aModuleName, 'Window', '' )
			aPluginNameList.append( aModuleName )

			aPixMap = GtkPixmap( self, os.path.join( aModule.theDirectoryName,\
			                                                    aModuleName ) + '.xpm' )

			if aIndicator == 0:
				aIndicator = 1
				aFirstButtonObj = GtkRadioButton()
				aFirstButton = \
				      self.theToolbar.append_element( TOOLBAR_CHILD_RADIOBUTTON,
				                                      aFirstButtonObj, aButtonName,
				 	                                  '', '', aPixMap, None )
				self.set_data( aModuleName, aFirstButton )
			else :
				aButtonObj = GtkRadioButton( aFirstButtonObj )
				aButton = \
				           self.theToolbar.append_element( TOOLBAR_CHILD_RADIOBUTTON,
						                                  aButtonObj, aButtonName,
							                              '', '', aPixMap, None )
				self.set_data( aModuleName, aButton )

			self.set_data( 'plugin_list' , aPluginNameList )
			aFirstButton.set_active( 1 )

	# end of setPluginList

	# ---------------------------------------------------------------
	# getSelectedPluginName
	#
	# return -> selected plugin name (string)
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def getSelectedPluginName( self ):

		aPluginList = self.get_data( 'plugin_list' )
		for aPluginName in aPluginList :
			aButton = self.get_data( aPluginName )
			if aButton.get_active() :
				aSelectedPluginName = aPluginName
		return aSelectedPluginName

	# end of getSelectedPluginName

	# ---------------------------------------------------------------
	# destroy
	#
	# *objects : dammy objets
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def destroy( self, *objects ):
		print 'destroy'
		self.theExist = 0

	def getExist( self ):
		return self.theExist


if __name__ == "__main__":

	def mainLoop():
		aPaletteWindow = PaletteWindow()
		gtk.mainloop()

	def main():
		mainLoop()

	main()
