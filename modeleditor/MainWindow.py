#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#		This file is part of E-CELL Model Editor package
#
#				Copyright (C) 1996-2003 Keio University
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
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#
import gtk

import ModelEditor
from ListWindow import *
import os
import os.path
import string
from Constants import *
from LayoutCommand import *


class MainWindow( ListWindow ):


	def __init__( self, theModelEditor ):
		
		"""
		in: ModelEditor theModelEditor
		returns nothing
		"""

		# init superclass
		ListWindow.__init__( self, theModelEditor )


	def openWindow( self ):
		
		"""
		in: nothing
		returns nothing
		"""

		# superclass openwindow
		ListWindow.openWindow( self )

		# add signal handlers
		
		self.addHandlers({ 
			'on_new_menu_activate' : self.__new_menu_clicked,\
			'load_model_menu_activate' : self.__load_menu_clicked,\
			'save_model_menu_activate' : self.__save_menu_clicked,\
			'on_save_model_as_activate' : self.__save_as_menu_clicked,\
			'on_validate_model_activate' : self.__validate_menu_clicked,\
			'on_import_menu_activate' : self.__import_menu_clicked,\
			'exit_menu_activate' : self.__quit_menu_clicked,\
			'on_undo_activate' : self.__undo_activated,\
			'on_redo_activate' : self.__redo_activated,\
			'stepper_window_activate' : self.__stepper_window_activated,\
			'entity_window_activate' : self.__entity_window_activated,\
			'on_pathway_editor_activate' : self.__pathway_editor_activated,\
			'about_menu_activate' : self.__about_menu_clicked,\
			'on_new_button_clicked' : self.__new_menu_clicked,\
			'on_open_button_clicked' : self.__load_menu_clicked,\
			'on_save_button_clicked' : self.__save_menu_clicked,\
			'on_quit_button_clicked' : self.__quit_menu_clicked,\
			'on_undo_button_clicked' :  self.__undo_activated,\
			'on_redo_button_clicked' : self.__redo_activated,\
			'on_StepperButton_clicked' : self.__stepper_window_activated,\
			'on_EntityButton_clicked' : self.__entity_window_activated,\
			'on_PathwayButton_clicked' : self.__pathway_editor_activated,\
			'on_LayoutButton_clicked' : self.__layout_window_activated,\
			'on_About_button_clicked' : self.__about_menu_clicked,\
			'on_scrolledwindow1_expose_event' : self.__scrollwindow_exposed
			})


		# set up message textbox buffer
		self.theMessageBuffer = gtk.TextBuffer(None)
		self['textview1'].set_buffer(self.theMessageBuffer)
		endIter=self.theMessageBuffer.get_end_iter()
		self.endMark=self.theMessageBuffer.create_mark('EM',endIter,gtk.FALSE)
                self.setIconList( os.environ['MEPATH'] + os.sep + "glade" +
                     os.sep + "modeleditor.png", os.environ['MEPATH'] + os.sep
                     + "glade" + os.sep + "modeleditor32.png" )	
		# update 
		self.update()


	def openModel ( self ):
		"""
		in: nothing
		returns nothing
		"""
		aFileName = self.__getFileSelection( self.theModelEditor.loadDirName )
		if aFileName == None:
			return
		# call modeleditor loadModel
		self.theModelEditor.loadModel( aFileName )


	def openRecentModel( self, aNumber ):
		"""
		in: nothing
		returns nothing
		"""

		# get filename and dirs
		recentList = self.theModelEditor.getRecentFileList()

		# call modeleditor loadModel
		self.theModelEditor.loadModel( recentList[ aNumber ] )


	def saveModel ( self, saveAsFlag = False ):
		"""
		in: bool saveAsFlag
		returns nothing 
		"""
		if saveAsFlag or not self.theModelEditor.modelHasName:
			aFileName = self.__getFileSelection( self.theModelEditor.saveDirName )
		else:
			aFileName = self.theModelEditor.theModelFileName

		if aFileName == None:
			return

		# call modeleditor savemodel
		self.theModelEditor.saveModel( aFileName )


	def importModel ( self ):
		"""
		in: nothing
		returns nothing
		"""
		self.theModelEditor.printMessage("Sorry, not implemented !", ME_ERROR )


	def displayMessage ( self, aMessage ):
		"""
		in: string or list of string aMessage
		returns nothing
		"""
		iter = self.theMessageBuffer.get_iter_at_mark( self.endMark )
		# writes string or list of strings to end of buffer
		if type(aMessage) == list:  
			
			# If first string is not '\n', add it.
			if len(aMessage)>0:
				if string.find(aMessage[0],'\n') != 0:
					aMessage[0] = '\n' + aMessage[0]

			# print message list
			for aLine in aMessage:
				aString = str( aLine )
				self.theMessageBuffer.insert( aString, len(aString), iter )
		else: 
			aString = str( aMessage )
			if string.find(aString,'\n') != 0:
				aString = '\n' + aString
			self.theMessageBuffer.insert(  iter, aString , len(aString) )

		# scrolls textwindow to end of buffer
		self['textview1'].scroll_to_mark(self.endMark,0)


	def update ( self ):
		"""
		in: nothing
		returns nothing
		"""

		# update View menu and buttons
		if not self.exists():
			return

		# update recent files menus
		recentList = self.theModelEditor.getRecentFileList()

		if len( recentList ) > 0:
			i = 1
			subMenu = gtk.Menu()
			for aFile in recentList:
		
				# get filename
				aFileName = os.path.split( aFile )[1]

				# add number
				aString = str(i) + ': ' + aFileName
		
				# create and add menuitem
				aMenuItem = gtk.MenuItem( aString )
				aMenuItem.connect ( 'activate', self.__recent_menu_clicked )
				aMenuItem.set_data( 'Number', i )

				subMenu.append( aMenuItem )
				i+=1

			# attach menu to menuitem
			subMenu.show_all()
			self['load_recent'].set_submenu( subMenu )

		# update title
		if self.theModelEditor.theModelName != '':
			aTitle = os.path.split ( self.theModelEditor.theModelName ) [1]
			self['MainWindow'].set_title( aTitle )

		else:
			self['MainWindow'].set_title( 'ModelEditor')

		# update undo, redo buttons, menus
		if self.theModelEditor.canUndo():
			undoFlag = gtk.TRUE
		else:
			undoFlag = gtk.FALSE

		self['button1'].set_sensitive( undoFlag )		
		self['undo1'].set_sensitive( undoFlag )

		if self.theModelEditor.canRedo():
			redoFlag = gtk.TRUE
		else:
			redoFlag = gtk.FALSE

		self['button2'].set_sensitive( redoFlag )		
		self['redo1'].set_sensitive( redoFlag )

			

		# update copy, cut, paste buttons, menus
#		adcpFlags = self.theModelEditor.getADCPFlags()

#		if adcpFlags[ME_COPY_FLAG]:
#			copyFlag = gtk.TRUE
#		else:
#			copyFlag = gtk.FALSE

#		self['copy1'].set_sensitive( copyFlag )
#		self['button4'].set_sensitive( copyFlag )

#		if adcpFlags[ME_COPY_FLAG] and adcpFlags[ME_DELETE_FLAG]:
#			cutFlag = gtk.TRUE
#		else:
#			cutFlag = gtk.FALSE

#		self['cut1'].set_sensitive( cutFlag )
#		self['button3'].set_sensitive( cutFlag )

#		if adcpFlags[ME_PASTE_FLAG]:
#			pasteFlag = gtk.TRUE
#		else:
#			pasteFlag = gtk.FALSE

#		self['paste1'].set_sensitive( pasteFlag )
#		self['button5'].set_sensitive( pasteFlag )





	def showAbout ( self ):
		# show about information
		self.theModelEditor.createAboutModelEditor()
		#self.theModelEditor.printMessage("Sorry, not implemented !", ME_ERROR )


	
	def __getFileSelection ( self, aDirname, aFileName = '' ):
		"""
		in: str aDirname, str aFileName
		returns None if dir is selected, or cancel is pressed
		sets self.searchDirName if not cancel is pressed
		"""
		defaultName = aDirname + os.sep + aFileName

		# create file selection dialog
		aDialog = gtk.FileSelection()

		# set init path for dialog
		aDialog.set_filename( defaultName )
                aPixbuf16 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                        os.sep + "glade" + os.sep + "modeleditor.png")
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                        os.sep + "glade" + os.sep + "modeleditor32.png")
		aDialog.set_icon_list(aPixbuf16, aPixbuf32)
		aDialog.set_title("Select a file name")
		aDialog.show_fileop_buttons( )

		# make dialog modal
		aDialog.set_modal( gtk.TRUE )

		# present dialog
		retVal = aDialog.run()

		aFileName = aDialog.get_filename()

		aDialog.hide()
		aDialog.destroy()

		# if cancel pressed return none
		if retVal == gtk.RESPONSE_CANCEL or retVal == gtk.RESPONSE_DELETE_EVENT:
			return None

		return aFileName
		

	#############################
	#      SIGNAL HANDLERS	    #
	#############################

	def deleted( self, *arg ):
		self.theModelEditor.quitApplication()
		return gtk.TRUE


	def __new_menu_clicked( self, *args ):
		self.theModelEditor.createNewModel()

	
	def __load_menu_clicked( self, *args ):
		self.openModel()
	

	def __save_menu_clicked( self, *args ):
		self.saveModel()

	
	def __save_as_menu_clicked( self, *args ):
		self.saveModel( True )


	def __validate_menu_clicked( self, *args ):
		self.theModelEditor.validateModel()


	def __import_menu_clicked( self, *args ):
		self.importModel()


	def __recent_menu_clicked( self, *args ):
		if type( args[0] ) == gtk.MenuItem:
			aNumber = args[0].get_data( 'Number' )
			self.openRecentModel( aNumber -1 )


	def __quit_menu_clicked( self, *args ):

		self.theModelEditor.quitApplication()


	def __undo_activated(self, *args):

		self.theModelEditor.undoCommandList()


	def __redo_activated(self, *args):

		self.theModelEditor.redoCommandList()



	def __stepper_window_activated( self, *args ):

		self.theModelEditor.createStepperWindow()


	def __entity_window_activated( self, *args ):

		self.theModelEditor.createEntityWindow()


	def __pathway_editor_activated( self, *args ):
		layoutManager = self.theModelEditor.theLayoutManager
		layoutName = layoutManager.getUniqueLayoutName()
		aCommand = CreateLayout( layoutManager, layoutName, True )
		self.theModelEditor.doCommandList( [ aCommand ] )

	
	def __layout_window_activated( self, *args ):

		self.theModelEditor.createLayoutWindow()


	def __about_menu_clicked( self, *args ):

		self.showAbout()



	def __scrollwindow_exposed( self, *args ):

		pass
