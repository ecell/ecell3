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


from OsogoWindow import *

from main import *

if GNOME_INSTALLED =='yes':
    import gnome.ui


import gtk

import MessageWindow

import string
import sys
import traceback

#
#import pyecell module
#
from ecell.GtkSessionMonitor import *
from ecell.ecssupport import *
from ecell.ECS import *
from ConfirmWindow import *

NAME        = 'gecell (Osogo)'
VERSION     = '0.0'
COPYRIGHT   = '(C) 2001-2002 Keio University'
AUTHORLIST  =  [
    'Design: Kenta Hashimoto <kem@e-cell.org>',
    'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
    'Programming: Yuki Fujita',
    'Yoshiya Matsubara',
    'Yuusuke Saito',
    'Masahiro Sugimoto <sugi@e-cell.org>',
    'Gabor Bereczki <gabor.bereczki@talk21.com>'
    ]
    
DESCRIPTION = 'Osogo is a simulation session monitoring module for E-CELL SE Version 3'


class MainWindow(OsogoWindow):
	"""MainWindow
	"""

	# ==========================================================================
	def __init__( self, aSession ):
		"""Constructor 
		- calls super class's constructor
		- calls openWindow
		"""
	
		# calls super class's constructor
		OsogoWindow.__init__( self, self, 'MainWindow.glade' )

		# -------------------------------------
		# stores pointer to Session
		# -------------------------------------
		self.theSession = aSession

	# end of init

	# ==========================================================================
	def openWindow( self ):
		"""override superclass's method
		Returns None
		"""

		# calls superclass's method
		OsogoWindow.openWindow(self)


		
		# -------------------------------------
		# creates MessageWindow 
		# -------------------------------------
		self.__theMessageWindow = MessageWindow.MessageWindow( self['textview1'] ) 
		self.__theMessageWindow.openWindow()
		self.__MessageWindow_attached = TRUE
		self.theSession.setMessageMethod( self.__printMessage )

		#get and setresizewindowminimumsize
		self.__expose(None,None)
		self.__MWminimalSize=self.__MWactualSize[:]
		self['scrolledwindow1'].set_size_request(\
		    self.__MWminimalSize[0],self.__MWminimalSize[1])
		self['handlebox24'].connect('child-attached',self.__MWChildAttached)
		self['handlebox24'].connect('child-detached',self.__MWChildDetached)
		


		# -------------------------------------
		# appends signal handlers
		# -------------------------------------
		self.theHandlerMap =  { 
		    # menu
			'load_model_menu_activate'             : self.__openFileSelection ,
			'load_script_menu_activate'            : self.__openFileSelection ,
			'save_model_menu_activate'             : self.__openFileSelection ,
			'exit_menu_activate'                   : self.__deleted ,
			'message_window_menu_activate'         : self.__toggleMessageWindow ,
			'interface_window_menu_activate'       : self.__displayWindow ,
			'create_new_entity_list_menu_activate' : self.__createEntityListWindow ,
			'logger_window_menu_activate'          : self.__displayWindow ,
			'stepper_window_menu_activate'         : self.__displayWindow ,
			'board_window_menu_activate'           : self.__displayWindow ,
			'preferences_menu_activate'            : self.openPreferences ,
			'about_menu_activate'                  : self.openAbout ,

			# toolbars
			'start_button_clicked'                 : self.__startSimulation ,
			'stop_button_clicked'                  : self.__stopSimulation ,
			'step_button_clicked'                  : self.__stepSimulation ,

			'on_sec_step_entry_activate'           : self.__setStepSizeOrSec ,

			'on_entitylist_button_clicked'         : self.__createEntityListWindow ,
			'on_logger_button_clicked'             : self.__displayWindow,
			'on_message_togglebutton_toggled'      : self.__toggleMessageWindow ,
			'on_stepper_button_clicked'            : self.__displayWindow,
			'on_interface_button_clicked'          : self.__displayWindow,
			'on_board_button_clicked'              : self.__displayWindow,
			'logo_button_clicked'                  : self.openAbout,
			'on_scrolledwindow1_expose_event'	: self.__expose
		}
		self.addHandlers( self.theHandlerMap )



		self['ecell_logo_toolbar'].set_style( gtk.TOOLBAR_ICONS )
		self.__setMenuAndButtonsStatus( FALSE )
		#self.theSession.updateFundamentalWindows()


		# toggles message window menu and button
		# At first message window is expanded, so the toggle button and menu are active.
		self['message_togglebutton'].set_active(TRUE)
		self['message_window_menu'].set_active(TRUE)

		# display MainWindow
		self[self.__class__.__name__].show_all()
		self.present()

		self.theStepSizeOrSec = 1

		# initializes FileSelection reference
		self.theFileSelection = None

		# initializes AboutDialog reference
		self.theAboutDialog = None
		self.update()
		
	# ==========================================================================
	def __expose( self, *arg ):
		"""expose
		Return None
		"""

	    # gets actual dimensions of scrolledwindow1 if it is displayed
	    # and attached

		if self.__MessageWindow_attached and self.__theMessageWindow.isShown:
			alloc_rect=self['scrolledwindow1'].get_allocation()
			self.__MWactualSize=[alloc_rect[2],alloc_rect[3]]
		

	# ==========================================================================
	def __resizeVertically( self, msg_heigth ): #gets messagebox heigth
		"""__resizeVertically
		Return None
		"""

		# gets fix components heigth
		menu_heigth=self['handlebox22'].get_child_requisition()[1]
		toolbar_heigth=max(self['handlebox19'].get_child_requisition()[1],\
		self['vbox73'].get_child_requisition()[1],\
		self['logo_button'].get_child_requisition()[1])
		statusbar_heigth=self['statusbar'].get_child_requisition()[1]

		# gets window_width
		window_width=self['MainWindow'].get_size()[0]

		# resizes
		window_heigth=menu_heigth+toolbar_heigth+statusbar_heigth+msg_heigth
		self['MainWindow'].resize(window_width,window_heigth)
	

	# ==========================================================================
	def __setMenuAndButtonsStatus( self, aDataLoadedStatus ):
		"""sets initial widgets status
		aDataLoadedStatus  -- the status of loading data
		         (TRUE:Model or Script is loaded / FALSE:Not loaded)
		Returns None
		"""

		# toolbar
		self['start_button'].set_sensitive(aDataLoadedStatus)
		self['stop_button'].set_sensitive(aDataLoadedStatus)
		self['step_button'].set_sensitive(aDataLoadedStatus)
		self['entitylist_button'].set_sensitive(aDataLoadedStatus)
		self['logger_button'].set_sensitive(aDataLoadedStatus)
		self['stepper_button'].set_sensitive(aDataLoadedStatus)
		self['interface_button'].set_sensitive(aDataLoadedStatus)
		self['board_button'].set_sensitive(aDataLoadedStatus)

		# file menu
		self['load_model_menu'].set_sensitive(not aDataLoadedStatus)
		self['load_script_menu'].set_sensitive(not aDataLoadedStatus)
		self['save_model_menu'].set_sensitive(aDataLoadedStatus)

		# view menu
		self['logger_window_menu'].set_sensitive(aDataLoadedStatus)
		self['stepper_window_menu'].set_sensitive(aDataLoadedStatus)
		self['interface_window_menu'].set_sensitive(aDataLoadedStatus)
		self['board_window_menu'].set_sensitive(aDataLoadedStatus)
		self['create_new_entity_list_menu'].set_sensitive(aDataLoadedStatus)
		self['save_model_menu'].set_sensitive(aDataLoadedStatus)





	# ==========================================================================
	def __openFileSelection( self, *arg ) :
		"""checks argument and calls self.openFileSelection() method.
		arg[0]  ---  self['load_model_menu'] / self['load_script_menu'] / self['save_model_menu']
		Return None
		[Note]:When the FileSelection is already displayed, moves it to the top of desctop.
		"""

		# checks the length of argument, but this is verbose
		if len( arg ) < 1:
			return None

		# when 'Load Model' is selected
		if arg[0] == self['load_model_menu']:
			self.openFileSelection('Load','Model')

		# when 'Load Script' is selected
		elif arg[0] == self['load_script_menu']:
			self.openFileSelection('Load','Script')

		# when 'Save Model' is selected
		elif arg[0] == self['save_model_menu']:
			self.openFileSelection('Save','Model')


	# ==========================================================================
	def openFileSelection( self, aType, aTarget ) :
		"""displays FileSelection 
		aType     ---  'Load'/'Save' (str)
		aTarget   ---  'Model'/'Script' (str)
		Return None
		[Note]:When FileSelection is already created, moves it to the top of desktop,
		       like singleton pattern.
		"""
		
		# When the FileSelection is already displayed, moves it to the top of desktop.
		if self.theFileSelection != None:
			self.theFileSelection.present()

		# When the FileSelection is not created yet, creates and displays it.
		else:

			# creates FileSelection
			self.theFileSelection = gtk.FileSelection()
			self.theFileSelection.connect('delete_event', self.__deleteFileSelection )
			self.theFileSelection.cancel_button.connect('clicked', self.__deleteFileSelection)

			# when 'Load Model' is selected
			if aType == 'Load' and aTarget == 'Model':
				self.theFileSelection.ok_button.connect('clicked', self.__loadData, aTarget)
				self.theFileSelection.complete( '*.'+ MODEL_FILE_EXTENSION )
				self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )

			# when 'Load Script' is selected
			elif aType == 'Load' and aTarget == 'Script':
				self.theFileSelection.ok_button.connect('clicked', self.__loadData, aTarget)
				self.theFileSelection.complete( '*.'+ SCRIPT_FILE_EXTENSION )
				self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,SCRIPT_FILE_EXTENSION) )

			# when 'Save Model' is selected
			elif aType == 'Save' and aTarget == 'Model':
				self.theFileSelection.ok_button.connect('clicked', self.__saveModel)
				self.theFileSelection.complete( '*.'+ MODEL_FILE_EXTENSION )
				self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )
			else:
				raise "(%s,%s) does not match." %(aType,aTarget)

			# displays the created FileSelection
			self.theFileSelection.show_all()


	# ==========================================================================
	def __deleteFileSelection( self, *arg ):
		"""deletes FileSelection
		Return None
		"""

		# deletes the reference to FileSelection
		if self.theFileSelection != None:
			self.theFileSelection.destroy()
			self.theFileSelection = None


	# ==========================================================================
	def __loadData( self, *arg ) :
		"""loads model or script file
		arg[0]    ---   ok button of FileSelection
		arg[1]    ---   'Model'/'Script' (str)
		Return None
		"""

		# checks the length of argument, but this is verbose
		if len( arg ) < 2:
			return None

		aFileType = arg[1]

		aFileName = self.theFileSelection.get_filename()
		if os.path.isfile( aFileName ):
			pass
		else:
			aMessage = ' Error ! No such file. \n[%s]' %aFileName
			self.theSession.message(aMessage)
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			self.theFileSelection.present()
			return None

		self.__deleteFileSelection()
		self.theSession.message('Loading %s file %s\n' %(aFileType, aFileName) )

		try:

			if aFileType == 'Model':
				self.theSession.loadModel( aFileName )

			elif aFileType == 'Script':
				self.theSession.loadScript( aFileName )

			self.theSession.theSimulator.initialize()

		except:

			# expants message window, when it is folded.
			if self.exists():
				if self['message_togglebutton'].get_active() == FALSE:
					self['message_togglebutton'].set_active(TRUE)
					self.__toggleMessageWindow( self['message_togglebutton'] ) 

			# displays confirm window
			aMessage = 'Can\'t load [%s]\nSee MessageWindow for details.' %aFileName
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			# displays message on MessageWindow
			aMessage = 'Can\'t load [%s]' %aFileName
			self.theSession.message(aMessage)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value, \
					sys.	exc_traceback), '\n' )
			self.theSession.message(anErrorMessage)


		self.update()
		self.theSession.updateFundamentalWindows()

	# end of loadModel

	# ==========================================================================
	def __saveModel( self, *arg ) :

		# gets file name
		aFileName = self.theFileSelection.get_filename()

		# when the file already exists
		if os.path.isfile( aFileName ):

			# displays confirm window
			aMessage = 'Would you like to replace the existing file? \n[%s]' %aFileName
			aDialog = ConfirmWindow(OKCANCEL_MODE,aMessage,'Confirm File Overwrite')

			# when canceled, does nothing 
			if aDialog.return_result() != OK_PRESSED:
				# does nothing
				return None

		# when ok is pressed, overwrites it.
		# deletes FileSelection
		self.__deleteFileSelection()

		try:

			# displays save message
			self.theSession.message('Save model file %s\n' % aFileName)
			# saves Model
			self.theSession.saveModel( aFileName )

		except:

			# expants message window, when it is folded.
			if self['message_togglebutton'].get_active() == FALSE:
				self['message_togglebutton'].set_active(TRUE)
				self.__toggleMessageWindow( self['message_togglebutton'] ) 

			# displays confirm window
			aMessage = 'Can\'t save [%s]\nSee MessageWindow for details.' %aFileName
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			# displays error message of MessageWindow
			self.theSession.message('Can\'t save [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.theSession.message(anErrorMessage)

		# updates
		self.update()
		self.updateButtons()
		self.theSession.updateFundamentalWindows()

	# end of loadModel


	# ==========================================================================
	def __deleted( self, *arg ):
		"""When delete_event happens or exit menu is selected, 
		this method is called.
		"""
		running_flag = False
		if self.theSession.theRunningFlag:
		# stop simulation temporarily
			self.theSession.stop()
			running_flag = True

		# If there is no logger data, exit this program.
		if len(self.theSession.getLoggerList()) != 0:
			aMessage = 'Are you sure you want to quit?'
			aTitle = 'Question'
			# Popup confirm window, and check user request
			aDialog = ConfirmWindow(1,aMessage,aTitle)

			# ok is pressed
		
			if aDialog.return_result() != OK_PRESSED:
				if running_flag:
					self.theSession.run()
				return gtk.TRUE		


		
		self.close()
		self.theSession.QuitGUI()
		return gtk.TRUE

	# ==========================================================================
	def close( self ):
		""" restores message method and closes window """

		self.theSession.restoreMessageMethod()
		OsogoWindow.close( self )


	# ==========================================================================
	def __startSimulation( self, *arg ) :
		"""starts simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		"""
		self.theSession.run()

	# ==========================================================================
	def startSimulation( self ) :
		""" starts simulation """
		self.__startSimulation( self, None )


	# ==========================================================================
	def __stopSimulation( self, *arg  ) :
		"""stops simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		"""
		self.theSession.stop()

	# ==========================================================================
	def stopSimulation( self ) :
		""" stops simulation """
		self.__stopSimulation( self, None )

	# ==========================================================================
	def __stepSimulation( self, *arg  ) : 
		"""steps simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		if step measure is second, then Session.run()
		if step measure is step than Session.step ()
		"""
		if self.getStepType():
			self.theSession.run( self.getStepSize() )
		else:
			self.theSession.step( self.getStepSize() )

	# ==========================================================================
	def stepSimulation( self ) :
		""" steps simulation """
		self.__stepSimulation( self, None )


	# ==========================================================================
	def getStepType( self ):
		""" returns state of sec radiobutton
			gtk.TRUE: seconds
			gtk.FALSE: steps
		"""
		return self['sec_radiobutton'].get_active()

	# ==========================================================================
	def setStepType( self, aState ):
		""" sets Step Type radiobutton state 
			values for aState
			True : seconds
			False : step			
			"""
		if aState == True :
			self['sec_radiobutton'].set_active( gtk.TRUE )
		if aState == False:
			self['sec_radiobutton'].set_active( gtk.FALSE )


	# ==========================================================================
	def getStepSize( self ):
		""" returns user or script specifid step size """
		return self.theStepSizeOrSec

	# ==========================================================================
	def setStepSize( self, num ):
		""" sets Stepsize entry box to num """
		self['sec_step_entry'].set_text( str (num) )
		self.__setStepSizeOrSec( self, None )

	# ==========================================================================
	def __setStepSizeOrSec( self, *arg ):

		# gets the inputerd characters from the GtkEntry. 
		aNewValue = string.strip( self['sec_step_entry'].get_text() )

		try:
			# converts string to float
			aNewValue = string.atof(aNewValue)

		# when string can't be converted to float
		except ValueError:

			# displays a Confirm Window.
			aMessage = "\"%s\" is not numerical value." %aNewValue
			aMessage += "\nInput numerical character" 
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			# sets the previous value 
			self['sec_step_entry'].set_text(str(self.theStepSizeOrSec))

		# when string can be converted to float
		else:

			#check for numerical constraints
			aMessage = ""

			if aNewValue <= 0:
				aMessage += "Input positive number.\n"
			    			
			# when 'step' is selected.
			if self['step_radiobutton'].get_active() == TRUE:

				# step must be integer 
				if int(aNewValue) != aNewValue:
					aMessage += "Input integer.\n"

				# convets float to int
				aNewValue =  int(aNewValue)
			
			if len(aMessage) > 0:

				#aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
				aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			else:

				self.theStepSizeOrSec=aNewValue
			

	# ==========================================================================
	def update( self ):
		"""updates this window 
		Returns None
		"""
		if not self.exists():
			return None
		# updates time
		aTime = self.theSession.theSimulator.getCurrentTime()
		self.theCurrentTime = aTime
		self['time_entry'].set_text( str( self.theCurrentTime ) )
		self['sec_step_entry'].set_text( str( self.theStepSizeOrSec ) )
		# when Model is already loaded.
		if len(self.theSession.theModelName) > 0:
			# updates status of menu and button 
			self.__setMenuAndButtonsStatus( TRUE )

			self.updateButtons()
        

	# ==========================================================================
	def updateButtons( self ):
		""" updates Buttons and menus with 
		latest FundamentalWindow status
		"""
		pass
		
		#if self.exists():

		#MessageWindow
		#	if self.__theMessageWindow.exists():
		#		flag = gtk.TRUE
		#	else:
		#		flag = gtk.FALSE
		#		self['message_window_menu'].activate()
		#
		#	self['message_togglebutton'].set_active( flag )


	# ==========================================================================
	def __displayWindow( self, *arg ):
		"""This method is called, when the menu or buttons on MainWindow is pressed.
		arg[0]   ---  menu or button
		Returns None
		"""

		# checks the length of argument, but this is verbose
		if len( arg ) < 1:
			return None

		# --------------------------------------
		# LoggerWindow
		# --------------------------------------
		# When LoggerWindow is selected
		if arg[0] == self['logger_button'] or \
		   arg[0] == self['logger_window_menu']:
			self.theSession.displayWindow('LoggerWindow')

		# When StepperWindow is selected
		elif arg[0] == self['stepper_button'] or \
		     arg[0] == self['stepper_window_menu']:
			self.theSession.displayWindow('StepperWindow')

		# When InterfaceWindow is selected
		elif arg[0] == self['interface_button'] or \
		     arg[0] == self['interface_window_menu']:
			self.theSession.displayWindow('InterfaceWindow')

		# When BoardWindow is selected
		elif arg[0] == self['board_button'] or \
		     arg[0] == self['board_window_menu']:
			self.theSession.displayWindow('BoardWindow')



	# ==========================================================================
	def hideMessageWindow( self ):
		self['message_togglebutton'].set_active(gtk.FALSE)
		self.__toggleMessageWindow( self['message_togglebutton'] ) 


	# ==========================================================================
	def showMessageWindow( self ):
		self['message_togglebutton'].set_active(gtk.TRUE)
		self.__toggleMessageWindow( self['message_togglebutton'] ) 

	# ==========================================================================
	def __toggleMessageWindow( self, *arg ) :
		"""expands or folds MessageWindow
		arg[0]   ---  self['message_togglebutton'] or self['message_window_menu']
		Returns None
		"""

		# checks the length of argument, but this is verbose
		if len(arg) < 1 :
			return None

		# When Message is required to be expanded.
		if arg[0].get_active() == TRUE:

			self['handlebox24'].show()
			if self.__MessageWindow_attached:
				self.__resizeVertically(self.__MWactualSize[1])
			else:
				self.__resizeVertically(0)

		# When Message is required to be folded.
		else:

			# hide handlebox, resize window
			self['handlebox24'].hide()
			self.__resizeVertically(0)
			self['message_togglebutton'].set_active(FALSE)


	# ==========================================================================
	def __MWChildAttached(self,obj,obj2):
		"""__MWChildAttached
		called when MessageBox is reatached to MainWindow
		must resize msgbox scrolledwindow to minimal size
		and the Mainwindow to extended size
		"""

		self['scrolledwindow1'].set_size_request(\
		                        self.__MWminimalSize[0], self.__MWminimalSize[1])
		self.__resizeVertically(self.__MWactualSize[1])
		self.__MessageWindow_attached=TRUE

	    
	# ==========================================================================
	def __MWChildDetached(self,obj,obj2):
		"""__MWChildDetached
		called when MessageBox is detached from MainWindow
		must resize msgbox scrolledwindow to actual size
		and the Mainwindow to minimalsize
		"""

		self['scrolledwindow1'].set_size_request(self.__MWactualSize[0], self.__MWactualSize[1])
		self.__resizeVertically(0)
		self.__MessageWindow_attached=FALSE
	        

	# ==========================================================================
	def openAbout( self, button_obj ):
		"""display the About window
		arg[0]   ---  self['about_menu']
		Return None
		"""

		# when AboutDialog is not created yet
		if self.theAboutDialog == None:
			if GNOME_INSTALLED=='yes':
			    self.theAboutDialog = gnome.ui.About( NAME, VERSION, COPYRIGHT, \
		                                          DESCRIPTION, AUTHORLIST)
			else:
			    self.theAboutDialog = OsogoAboutWindow( NAME, VERSION, COPYRIGHT, \
		                                          DESCRIPTION, AUTHORLIST)
			self.theAboutDialog.set_title( 'about osogo' )
			self.theAboutDialog.show_all()
			self.theAboutDialog.connect('destroy', self.__deleteAboutDialog )

		# when AboutDialog is already created 
		else:

			# moves it to the top of desktop
			self.theAboutDialog.present()

	# ==========================================================================
	def __deleteAboutDialog( self, *arg ):
		"""deletes AboutDialog
		Return None
		"""

		# deletes the reference to AboutDialog
		if self.theAboutDialog != None:
			self.theAboutDialog.destroy()
			self.theAboutDialog = None


	# ==========================================================================
	def openPreferences( self, *arg ):
		"""display the preference window
		arg[0]   ---  self['preferences_menu']
		Return None
		"""

		# Preference Window is not implemented yet.
		# So, displays Warning Dialog
		aMessage = ' Sorry ! Not implemented... [%s]\n' %'03/April/2003'
		aDialog = ConfirmWindow(OK_MODE,aMessage,'Sorry!')


	# ==========================================================================
	def __printMessage( self, aMessage ):
		"""prints message on MessageWindow
		aMessage   ---  a message (str or list)
		Return None
		"""

		# prints message on MessageWindow
		self.__theMessageWindow.printMessage( aMessage )

	# ==========================================================================
	def __createEntityListWindow( self, *arg ):
		self.theSession.createEntityListWindow( )

	# ========================================================================
	def deleted( self, *arg ):
		""" When 'delete_event' signal is chatcked( for example, [X] button is clicked ),
		delete this window.
		Returns TRUE
		"""
		return self.__deleted( *arg )



# end of MainWindow

# osogo about dialof class

class OsogoAboutWindow(gtk.Dialog):
	    """ popup window to display info about e-cell when gnome support is not
	    present
	    """
	    def __init__(self, _name, _version, _copyright, _description,
			    _authorlist):
		    #init dialog
		    gtk.Dialog.__init__(self)
		    
		    #add button and connect destroy signal
		    OK_Button=gtk.Button("OK")
		    OK_Button.connect("clicked",self.clicked)
		    self.add_action_widget(OK_Button,1)
		    #add name description, copyright, version as labels
		    self.vbox.add(gtk.Label(_name))
		    self.vbox.add(gtk.Label(_version))
		    self.vbox.add(gtk.Label(_copyright))
		    self.vbox.add(gtk.Label(_description))
		    #add authorlist as a scrolled window
		    
		    Auth_Vbox=gtk.VBox()
		    Auth_Vbox.add(gtk.Label("AUTHORS:"))
		    for _author in _authorlist:
			Auth_Vbox.add(gtk.Label(_author))
		    Auth_List=gtk.ScrolledWindow()
		    Auth_List.set_size_request(200,200)
		    Auth_List.add_with_viewport(Auth_Vbox)
		    self.vbox.add(Auth_List)
		    
	    def destroy(self):
		gtk.Dialog.destroy(self)

	    def clicked(self,button_obj):
		self.destroy()
		
# end of osogo about dialof class
