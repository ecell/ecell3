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
import gtk
import os
import re
import string

from OsogoWindow import *

# ---------------------------------------------------------------
# PaletteWindow -> gtk.Window
#   - manages PaletteWindow
# ---------------------------------------------------------------
class PaletteWindow(gtk.Window,OsogoWindow):

	# ---------------------------------------------------------------
	# Constructor
	#   - creates MessageWindow
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aMainWindow ):

		OsogoWindow.__init__( self, aMainWindow )
		gtk.Window.__init__( self, gtk.WINDOW_TOPLEVEL )
		self.theMainWindow = aMainWindow
		self.theToolbar = gtk.Toolbar()# gtk.ORIENTATION_VERTICAL, gtk.TOOLBAR_BOTH )
		self.add( self.theToolbar )
		self.set_data('toolbar', self.theToolbar)
		self.connect('destroy', self.destroy)
		self.theExist = 1
                self.connect( 'destroy', self.destroyWindow )


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

			aPixMap = gtk.Image()
			aPixMap.set_from_file( os.path.join( aModule.theDirectoryName,\
			                                                    aModuleName ) + '.xpm' )

			if aIndicator == 0:
				aIndicator = 1
				aFirstButtonObj = gtk.RadioButton()
				aFirstButton = self.theToolbar.append_element(\
                                               gtk.TOOLBAR_CHILD_RADIOBUTTON,\
					       aFirstButtonObj, aButtonName,\
					       '', '', aPixMap, None, None )
				
				self.set_data( aModuleName, aFirstButton )
			else :
				aButtonObj = gtk.RadioButton( aFirstButtonObj )
				aButton = self.theToolbar.append_element(\
					gtk.TOOLBAR_CHILD_RADIOBUTTON,\
					aButtonObj, aButtonName,\
					'', '', aPixMap, None, None )
				
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
	#def destroy( self, *objects ):
	#	self.theExist = 0
	#	self.theMainWindow.updateBasicWindows()

	#def getExist( self ):
	#	return self.theExist


if __name__ == "__main__":

	def mainLoop():
		aPaletteWindow = PaletteWindow()
		gtk.mainloop()

	def main():
		mainLoop()

	main()
