#!/usr/bin/python
#
# ToolLauncherOkCancel.py  - E-Cell3 Tool Launcher OkCancel Window
#

from ParentWindow import *
import sys

try:
	import gtk
	import os
except KeyboardInterrupt:
	sys.exit(1)

class ToolLauncherOkCancel( ParentWindow ):

	def __init__( self, obj ):
		"""Constructor 
		- calls parent class's constructor
		- calls openWindow
		"""
		ParentWindow.__init__( self, 'ToolLauncherOkCancel.glade' )
		self.theObj = obj
	# end of __init__


	def openWindow( self, msg ):
		"""overwrite parent class' method
		Returns None
		"""
		ParentWindow.openWindow( self )
		self.theHandlerMap =  {
			'on_ok_button_clicked'                  : self.onOk ,
			'on_cancel_button_clicked'              : self.onCancel ,
		}
		self.addHandlers( self.theHandlerMap )
		self.__update( msg )
	# end of openWindow


	def __update( self, msg ):
	    	"""update the checkboxes and entry boxes with the preferences from ToolLauncher
		"""
		self['errormsg'].set_text( msg )
		self.update()
	# end of __update


	def onOk( self, *arg ):
		"""when OK button is clicked
		"""
		self.theObj.onOk()
		self.close()
		return gtk.TRUE
	# end of onOk


	def onCancel( self, *arg ):
		"""when Cancel button is clicked
		"""
#		self.theObj.onCancel()
		self.close()
#		return gtk.FALSE
	# end of onCancel
# end of ToolLauncherOkCancel

