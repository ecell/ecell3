#!/usr/bin/python
#
# ToolLauncherVersion.py  - E-Cell3 Tool Launcher ErrorMessage Window
#


from ParentWindow import *
import sys

try:
	import gtk
	import os
except KeyboardInterrupt:
	sys.exit(1)


class ToolLauncherErrorMessage( ParentWindow ):

	def __init__( self, aToolLauncher ):
		"""Constructor 
		- calls parent class's constructor
		- calls openWindow
		"""
		ParentWindow.__init__( self, 'ToolLauncherErrorMessage.glade' )
		self.theToolLauncher = aToolLauncher
		self.thePathSelectorDlg = gtk.FileSelection( 'Select Path' )
		self.thePathSelectorDlg.ok_button.connect('clicked', self.onOK)

	def openWindow( self, msg ):
		"""overwrite parent class' method
		Returns None
		"""
		ParentWindow.openWindow(self)
		self.theHandlerMap =  {
			'on_ok_button_clicked'              : self.onOK ,
		}
		self.addHandlers( self.theHandlerMap )
                self.setIconList(
			os.environ['TLPATH'] + os.sep + "toollauncher.png",
			os.environ['TLPATH'] + os.sep + "toollauncher32.png")

		self.__update( msg )

	def __update( self, msg):
	    	"""update the checkboxes and entry boxes with the preferences from ToolLauncher
		"""
		self['errormsg'].set_text( msg )
		self.update()

	def onOK( self, *arg ):
		"""when OK button is clicked
		"""
		self.close()
		return True

