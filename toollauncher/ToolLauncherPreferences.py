#!/usr/bin/python
#
# ToolLauncherPreferences.py  - E-Cell3 Tool Launcher Preferences Window
#


from ParentWindow import *
from Preferences import *
import sys

try:
	import gtk
	import os
except KeyboardInterrupt:
	sys.exit(1)


class ToolLauncherPreferences( ParentWindow ):

	def __init__( self, aToolLauncher ):
		"""Constructor 
		- calls parent class's constructor
		- calls openWindow
		"""
		ParentWindow.__init__( self, 'ToolLauncherPreferences.glade' )
		self.theOpenFlag = "0"
		self.theToolLauncher = aToolLauncher
		self.thePathSelectorDlg = gtk.FileSelection( 'Select Path' )
		self.thePathSelectorDlg.ok_button.connect('clicked', self.updatePathSelection)
		self.thePathSelectorDlg.cancel_button.connect('clicked', self.hidePathSelectorDlg)
		self.thePathSelectorDlg.connect('delete_event', self.__deleteFileSelection)


	def openWindow( self ):
		"""overwrite parent class' method
		Returns None
		"""
		ParentWindow.openWindow(self)
		self.theHandlerMap =  {
			'on_ok_button_clicked'              : self.onOK ,
			'on_cancel_button_clicked'          : self.onCancel ,
			'on_editor_path_button_clicked'	    : self.setPath ,
			'on_model_path_button_clicked'	    : self.setPath ,
			'on_programs_path_button_clicked'   : self.setPath ,
			'on_revert_button_clicked'          : self.sevePref
		}
		self.addHandlers( self.theHandlerMap )
		self.theOpenFlag = "1"
		self.__update()

	def getOpenFlg( self ):
		return self.theOpenFlag


	# ==========================================================================
	def sevePref( self, *arg ):
		defaultPref = Preferences( self.theToolLauncher )
		defaultPref.sevePrefernces()
		self.openWindow()
	# end of sevePref


	def updatePathSelection( self, *arg ):
		"""update the selected path into the respective entry box
		when the OK button of Path Selector dialog is clicked
		"""

		if self.thePathSelectorDlg.get_title() == 'Select Editor Path':
			self['editor_path'].set_text( self.thePathSelectorDlg.get_filename() )
		elif self.thePathSelectorDlg.get_title() == 'Select Model Home Directory':
			directoryName = os.path.dirname( self.thePathSelectorDlg.get_filename() )
			self['model_path'].set_text( directoryName )
		elif self.thePathSelectorDlg.get_title() == 'Select Conversion Programs Directory':
			directoryName = os.path.dirname( self.thePathSelectorDlg.get_filename() )
			self['programs_path'].set_text( directoryName )
		self.thePathSelectorDlg.hide()


	def hidePathSelectorDlg( self, *arg):
		"""hide the Path Selector dialog when the Cancel button is clicked
		"""
		self.thePathSelectorDlg.hide()


	def setPath( self, *arg ):
		"""when any of the path selection buttons is clicked
		"""
		if len( arg ) < 1:
			return None

		if self.thePathSelectorDlg == None:
			self.thePathSelectorDlg = gtk.FileSelection( 'Select Path' )
			self.thePathSelectorDlg.ok_button.connect('clicked', self.updatePathSelection)
			self.thePathSelectorDlg.cancel_button.connect('clicked', self.hidePathSelectorDlg)
			self.thePathSelectorDlg.connect('delete_event', self.__deleteFileSelection)

		self.thePathSelectorDlg.set_filename( self.theToolLauncher.thePref['ecell3_path']+os.sep )

		if arg[0] == self['editor_path_button']:
			self.thePathSelectorDlg.set_title('Select Editor Path')
		elif arg[0] == self['model_path_button']:
			self.thePathSelectorDlg.set_title('Select Model Home Directory')
		elif arg[0] == self['programs_path_button']:
			self.thePathSelectorDlg.set_title('Select Conversion Programs Directory')

		self.thePathSelectorDlg.set_modal(True)
		self.thePathSelectorDlg.activate()
		self.thePathSelectorDlg.show_all()


	def onOK( self, *arg ):
	        """when ok button is clicked
	        """
		if self['save_em_checkbox'].get_active():
		    	self.theToolLauncher.thePref['save_em'] = '1'
		else:
		    	self.theToolLauncher.thePref['save_em'] = '0' 

		if self['save_eml_checkbox'].get_active():
		    	self.theToolLauncher.thePref['save_eml'] = '1'
		else:
		    	self.theToolLauncher.thePref['save_eml'] = '0' 

		self.theToolLauncher.thePref['editor_path'] = self['editor_path'].get_text()
		self.theToolLauncher.thePref['model_home'] = self['model_path'].get_text()
		self.theToolLauncher.thePref['programs_path'] = self['programs_path'].get_text()
		self.theToolLauncher.savePreferences()
		flg = self.theToolLauncher.checkPref()
		if flg == 0:
			return self.__deleted( *arg )


	def onCancel( self, *arg ):
		"""when cancel button is clicked
		"""
		return self.__deleted( *arg )
	
	def __update( self, *arg):
	    	"""update the checkboxes and entry boxes with the preferences from ToolLauncher
		"""
		if self.theToolLauncher.thePref['save_em'] == '1' :
		    self['save_em_checkbox'].set_active( gtk.TRUE )
		else:
		    self['save_em_checkbox'].set_active( gtk.FALSE )

		if self.theToolLauncher.thePref['save_eml'] == '1' :
		    self['save_eml_checkbox'].set_active( gtk.TRUE )
		else:
		    self['save_eml_checkbox'].set_active( gtk.FALSE )


		self['editor_path'].set_text( self.theToolLauncher.thePref['editor_path'] )
		self['model_path'].set_text( self.theToolLauncher.thePref['model_home'] )
		self['programs_path'].set_text( self.theToolLauncher.thePref['programs_path'] )
		self.update()


	def __deleted( self, *arg ):
		"""close the window, 
		arg[0] ---  self['exit_menu']
		Return True
		"""
		del self.thePathSelectorDlg 
		self.theToolLauncher.thePref['window_open'] = 0
		self.close()
		return gtk.TRUE


	def __deleteFileSelection( self, *arg ):
		"""deletes FileSelection
		Return None
		"""

		# deletes the reference to FileSelection
		if self.thePathSelectorDlg != None:
			self.thePathSelectorDlg.destroy()
			self.thePathSelectorDlg = None

