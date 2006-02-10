#!/usr/bin/python
#
# ToolLauncherPreferences.py  - E-Cell3 Tool Launcher Preferences Window
#


from ParentWindow import *
from Preferences import *
import sys
import gtk.gdk

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
                aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                           os.environ['TLPATH'] + os.sep + "toollauncher.png")
                aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                           os.environ['TLPATH'] + os.sep + "toollauncher32.png")
                self.thePathSelectorDlg.set_icon_list(aPixbuf16, aPixbuf32)

		self.defaultPref = Preferences( self.theToolLauncher )


	def openWindow( self ):
		"""overwrite parent class' method
		Returns None
		"""
		ParentWindow.openWindow(self)
		self.theHandlerMap =  {
			'on_ok_button_clicked'              : self.onOK ,
			'on_cancel_button_clicked'          : self.onCancel ,
			'on_editor_path_button_clicked'	    : self.setPath ,
			'on_models_path_button_clicked'	    : self.setPath ,
			'on_current_model_button_clicked'   : self.setPath ,
			'on_revert_button_clicked'          : self.savePref
		}
		self.addHandlers( self.theHandlerMap )
		self.theOpenFlag = "1"
                self.setIconList(
			os.environ['TLPATH'] + os.sep + "toollauncher.png",
			os.environ['TLPATH'] + os.sep + "toollauncher32.png")
                self.__update()

	def getOpenFlg( self ):
		return self.theOpenFlag


	# ==========================================================================
	def savePref( self, *arg ):
		self.defaultPref.savePreferences()
                self.__update()

	# end of savePref


	def updatePathSelection( self, *arg ):
		"""update the selected path into the respective entry box
		when the OK button of Path Selector dialog is clicked
		"""

		if self.thePathSelectorDlg.get_title() == 'Select Editor Path':
                        fileName = self.thePathSelectorDlg.get_filename()
                        if os.path.isfile( fileName ):
                                self['editor_path'].set_text( fileName )
			
		elif self.thePathSelectorDlg.get_title() == 'Select Models Home Directory':
                        dirName = self.thePathSelectorDlg.get_filename()
                        if os.path.isdir( dirName ):
                                self['models_path'].set_text( dirName )
                                self['current_model'].set_text( '' )			
			
		elif self.thePathSelectorDlg.get_title() == 'Select Current Model Directory':
                        dirName = self.thePathSelectorDlg.get_filename()
                        if os.path.isdir( dirName ):
                                modelBaseDir, modelName = os.path.split( 
                                        self.thePathSelectorDlg.get_filename() )
                                self['models_path'].set_text( modelBaseDir )
                                self['current_model'].set_text( modelName )			
			
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
                        aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                           os.environ['TLPATH'] + os.sep + "toollauncher.png")
                        aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                           os.environ['TLPATH'] + os.sep + "toollauncher32.png")
                        self.thePathSelectorDlg.set_icon_list(
                                        aPixbuf16, aPixbuf32)


		if arg[0] == self['editor_path_button']:
                        if os.path.isfile( self['editor_path'].get_text() ):
                                self.thePathSelectorDlg.set_filename( os.path.dirname(
                                        self['editor_path'].get_text()+os.sep ) )
                        else:
                                self.thePathSelectorDlg.set_filename(
                                                self.theToolLauncher.thePref['ecell3_path']+os.sep )
			self.thePathSelectorDlg.set_title('Select Editor Path')
		elif arg[0] == self['models_path_button']:
                        if os.path.isdir( self['models_path'].get_text() ):
                                self.thePathSelectorDlg.set_filename(
                                                self['models_path'].get_text()+os.sep )
                        else:
                                self.thePathSelectorDlg.set_filename(
                                                self.theToolLauncher.thePref['ecell3_path']+os.sep )
			self.thePathSelectorDlg.set_title('Select Models Home Directory')
		elif arg[0] == self['current_model_button']:
                        if os.path.isdir( self['models_path'].get_text() ):
                                self.thePathSelectorDlg.set_filename(
                                                self['models_path'].get_text()+os.sep )
                        else:
                                self.thePathSelectorDlg.set_filename(
                                                self.theToolLauncher.thePref['ecell3_path']+os.sep )                        
			self.thePathSelectorDlg.set_title('Select Current Model Directory')

		self.thePathSelectorDlg.set_modal(True)
		self.thePathSelectorDlg.activate()
		self.thePathSelectorDlg.show_all()
 

	def onOK( self, *arg ):
	        """when ok button is clicked
	        """
                if self.isPrefOK():
                        self.setToolLauncherPref()
                        return self.__deleted( *arg )
                        
        def setToolLauncherPref( self, *arg ):
      		if self['save_em_checkbox'].get_active():
		    	self.theToolLauncher.thePref['save_em'] = '1'
		else:
		    	self.theToolLauncher.thePref['save_em'] = '0' 

		if self['save_eml_checkbox'].get_active():
		    	self.theToolLauncher.thePref['save_eml'] = '1'
		else:
		    	self.theToolLauncher.thePref['save_eml'] = '0' 

		self.theToolLauncher.thePref['editor_path'] = self['editor_path'].get_text()
		self.theToolLauncher.thePref['models_path'] = self['models_path'].get_text()
		self.theToolLauncher.thePref['current_model'] = self['current_model'].get_text()
		self.theToolLauncher.savePreferences()

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

                if not os.path.isfile( self['editor_path'].get_text() ):
                        errorMsg = "Please enter a valid path for your favorite text editor."
			self.theToolLauncher.viewErrorMessage( errorMsg )
                        return False 
                else:
                        return True

	# end of isEditorPathOK


	# ==========================================================================
	def isModelsPathOK( self ):
                
                if self['models_path'].get_text() == '' or not os.path.isdir( self['models_path'].get_text() ):
			errorMsg = "Please enter a valid base directory of the models that you will create."
			self.theToolLauncher.viewErrorMessage( errorMsg )
                        return False
			
		else:
			return True

	# end of isModelsPathOK


	# ==========================================================================
	def isCurrentModelOK( self ):

                currentModelPath = os.path.join(
                                self['models_path'].get_text(),
                                self['current_model'].get_text() )
                
                if self['current_model'].get_text() == '':
                        errorMsg = "Please enter a valid current model name."
			self.theToolLauncher.viewErrorMessage( errorMsg )
                        return False			
                elif not os.path.isdir( currentModelPath ):
                        errorMsg = "The directory " + currentModelPath + \
                            " for current model does not exist.\n Please create it using the Folder... button."
			self.theToolLauncher.viewErrorMessage( errorMsg )
                        return False			
		else:
			return True

	# end of isCurrentModelOK


	def onCancel( self, *arg ):
		"""when cancel button is clicked
		"""
		return self.__deleted( *arg )
	
	def __update( self, *arg):
	    	"""update the checkboxes and entry boxes with the preferences from ToolLauncher
		"""
		if self.theToolLauncher.thePref['save_em'] == '1' :
		    self['save_em_checkbox'].set_active( True )
		else:
		    self['save_em_checkbox'].set_active( False )

		if self.theToolLauncher.thePref['save_eml'] == '1' :
		    self['save_eml_checkbox'].set_active( True )
		else:
		    self['save_eml_checkbox'].set_active( False )


		self['editor_path'].set_text( self.theToolLauncher.thePref['editor_path'] )
		self['models_path'].set_text( self.theToolLauncher.thePref['models_path'] )
		self['current_model'].set_text( self.theToolLauncher.thePref['current_model'] )
		self.update()


	def __deleted( self, *arg ):
		"""close the window, 
		arg[0] ---  self['exit_menu']
		Return True
		"""
		del self.thePathSelectorDlg 
		self.theToolLauncher.thePref['window_open'] = 0
		self.close()
		return True


	def __deleteFileSelection( self, *arg ):
		"""deletes FileSelection
		Return None
		"""

		# deletes the reference to FileSelection
		if self.thePathSelectorDlg != None:
			self.thePathSelectorDlg.destroy()
			self.thePathSelectorDlg = None

