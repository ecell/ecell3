#!/usr/bin/python
#
# ToolLauncherVersion.py  - E-Cell3 Tool Launcher Version Window
#


from ParentWindow import *
import sys

try:
	import gtk
	import os
except KeyboardInterrupt:
	sys.exit(1)

class ToolLauncherVersion( ParentWindow ):

	# ==========================================================================
	# Constractor
	# ==========================================================================
	def __init__( self, aToolLauncher ):
		"""Constructor 
		- calls parent class's constructor
		- calls openWindow
		"""
		ParentWindow.__init__( self, 'ToolLauncherVersion.glade' )
		self.theToolLauncher = aToolLauncher
		self.thePathSelectorDlg = gtk.FileSelection( 'Select Path' )
		self.thePathSelectorDlg.ok_button.connect('clicked', self.onOK)
	# end of __init__

	# ==========================================================================
	# 
	# ==========================================================================
	def openWindow( self ):
		"""overwrite parent class' method
		Returns None
		"""
		ParentWindow.openWindow(self)
		self.theHandlerMap =  {
			'on_ok_button_clicked'              : self.onOK ,
		}
		self.addHandlers( self.theHandlerMap )
	# end of openWindow

	# ==========================================================================
	# 
	# ==========================================================================
	def onOK( self, *arg ):
		"""when OK button is clicked
		"""
		self.close()
		return gtk.TRUE
	# end of onOK
