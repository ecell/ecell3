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
from OsogoPluginManager import *

import gnome.ui
import gtk

import MessageWindow
import EntityListWindow
import LoggerWindow
import InterfaceWindow 
import StepperWindow 
import BoardWindow 
import ConfigParser

import string
import sys
import traceback

#
#import pyecell module
#
import ecell.ecs
import ecell.Session
from ecell.ecssupport import *
from ecell.ECS import *


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
    'Gabor Bereczki <gabor.bereczki@axelero.hu>'
    ]
    
DESCRIPTION = 'Osogo is a simulation session monitoring module for E-CELL SE Version 3'


class MainWindow(OsogoWindow):
	"""MainWindow
	"""

	# ==========================================================================
	def __init__( self ):
		"""Constructor 
		- calls super class's constructor
		- calls openWindow
		"""
	
		# calls super class's constructor
		OsogoWindow.__init__( self, self, 'MainWindow.glade' )

		# calls openWindow
		self.openWindow()

	# end of init

	# ==========================================================================
	def openWindow( self ):
		"""override superclass's method
		Returns None
		"""

		# calls superclass's method
		OsogoWindow.openWindow(self)

		# -------------------------------------
		# reads defaults from osogo.ini 
		# -------------------------------------
		self.theConfigDB=ConfigParser.ConfigParser()
    		self.theConfigDB.read(OSOGO_PATH+os.sep+'osogo.ini')
		
		
		# -------------------------------------
		# creates MessageWindow 
		# -------------------------------------
		self.theMessageWindow = MessageWindow.MessageWindow( self['textview1'] ) 
		self.theMessageWindow.openWindow()
		self.MessageWindow_attached=TRUE
		#get and setresizewindowminimumsize
		self.expose(None,None)
		self.MW_minimal_size=self.MW_actual_size[:]
		self['scrolledwindow1'].set_size_request(\
		    self.MW_minimal_size[0],self.MW_minimal_size[1])
		self['handlebox24'].connect('child-attached',self.MW_child_attached)
		self['handlebox24'].connect('child-detached',self.MW_child_detached)
		
		# -------------------------------------
		# creates Session
		# -------------------------------------
		self.theSession = ecell.Session( ecell.ecs.Simulator() )
		self.theSession.theMainWindow = self
		self.theSession.setMessageMethod( self.theMessageWindow.printMessage )

		# -------------------------------------
		# creates PluginManager
		# -------------------------------------
		self.thePluginManager = OsogoPluginManager( self )
		self.thePluginManager.loadAll()

		# -------------------------------------
		# creates FundamentalWindow
		# -------------------------------------
		
		# key:window name(str) value:window instance
		self.__theFundamentalWindows = {}

		# creates fundamental windows
		aLoggerWindow     = LoggerWindow.LoggerWindow( self.theSession , self )
		anInterfaceWindow = InterfaceWindow.InterfaceWindow( self )
		aStepperWindow    = StepperWindow.StepperWindow( self.theSession , self )
		aBoardWindow      = BoardWindow.BoardWindow( self.theSession, self )

		# saves them to map
		self.__theFundamentalWindows['LoggerWindow'] = aLoggerWindow
		self.__theFundamentalWindows['InterfaceWindow'] = anInterfaceWindow
		self.__theFundamentalWindows['StepperWindow'] = aStepperWindow
		self.__theFundamentalWindows['BoardWindow'] = aBoardWindow

		# key:EntityListWindow instance value:None
		# In deleteEntityListWindow method, an instance of EntityListWindow is
		# accessed directory. The sequence information of EntityListWindow does
		# not need. So the references to EntityListWindow instances should be 
		# held dict's key. Values of dict are not also imported.
		self.theEntityListInstanceMap = {}  

		self.theUpdateInterval = 150
		self.theStepSizeOrSec = 1.0
		self.theRunningFlag = 0

		# -------------------------------------
		# appends signal handlers
		# -------------------------------------
		self.theHandlerMap =  { 
		    # menu
			'load_model_menu_activate'             : self.__openFileSelection ,
			'load_script_menu_activate'            : self.__openFileSelection ,
			'save_model_menu_activate'             : self.__openFileSelection ,
			'exit_menu_activate'                   : self.deleted ,
			'message_window_menu_activate'         : self.__toggleMessageWindow ,
			'interface_window_menu_activate'       : self.__displayWindow ,
			'create_new_entity_list_menu_activate' : self.createEntityListWindow ,
			'logger_window_menu_activate'          : self.__displayWindow ,
			'stepper_window_menu_activate'         : self.__displayWindow ,
			'board_window_menu_activate'           : self.__displayWindow ,
			'preferences_menu_activate'            : self.openPreferences ,
			'about_menu_activate'                  : self.openAbout ,

			# toolbars
			'start_button_clicked'                 : self.startSimulation ,
			'stop_button_clicked'                  : self.stopSimulation ,
			'step_button_clicked'                  : self.stepSimulation ,

			'on_sec_step_entry_activate'           : self.setStepSizeOrSec ,

			'on_entitylist_button_clicked'         : self.createEntityListWindow ,
			'on_logger_button_clicked'             : self.__displayWindow,
			'on_message_togglebutton_toggled'      : self.__toggleMessageWindow ,
			'on_stepper_button_clicked'            : self.__displayWindow,
			'on_interface_button_clicked'          : self.__displayWindow,
			'on_board_button_clicked'              : self.__displayWindow,
			'logo_button_clicked'                  : self.openAbout,
		}
		self.addHandlers( self.theHandlerMap )


		# -------------------------------------
		# initializes for run method 
		# -------------------------------------
		self.theSession.theSimulator.setEventChecker( gtk.events_pending )
		self.theSession.theSimulator.setEventHandler( gtk.mainiteration  )
		self['ecell_logo_toolbar'].set_style( gtk.TOOLBAR_ICONS )
		self.setMenuAndButtonsStatus( FALSE )
		self.updateFundamentalWindows()


		# toggles message window menu and button
		# At first message window is expanded, so the toggle button and menu are active.
		self['message_togglebutton'].set_active(TRUE)
		self['message_window_menu'].set_active(TRUE)

		# display MainWindow
		self[self.__class__.__name__].show_all()
		self.present()


		# initializes FileSelection reference
		self.theFileSelection = None

		# initializes AboutDialog reference
		self.theAboutDialog = None


	# ==========================================================================
	def expose(self,obj,obj2):
		"""expose
		Return None
		"""

	    # gets actual dimensions of scrolledwindow1 if it is displayed
	    # and attached
		if self.MessageWindow_attached and self.theMessageWindow.isShown:
			alloc_rect=self['scrolledwindow1'].get_allocation()
			self.MW_actual_size=[alloc_rect[2],alloc_rect[3]]
		

	# ==========================================================================
	def resize_vertically(self,msg_heigth): #gets messagebox heigth
		"""resize_vertically
		Return None
		"""

		# gets fix components heigth
		menu_heigth=self['handlebox22'].get_child_requisition()[1]
		toolbar_heigth=max(self['handlebox19'].get_child_requisition()[1],\
		self['vbox73'].get_child_requisition()[1],\
		self['logo_button'].get_child_requisition()[1])
		statusbar_heigth=self['statusbar'].get_child_requisition()[1]

		# gets window_width
		window_width=self['MainWindow'].get_size()[1]

		# resizes
		window_heigth=menu_heigth+toolbar_heigth+statusbar_heigth+msg_heigth
		self['MainWindow'].resize(window_width,window_heigth)
	

	# ==========================================================================
	def setMenuAndButtonsStatus( self, aDataLoadedStatus ):
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
	def read_ini(self,aPath):
		"""read osogo.ini file
		an osogo.ini file may be in the given path
		that have an osogo section or others but no default
		argument may be a filename as well
		"""

	    # first delete every section apart from default
		for aSection in self.theConfigDB.sections():
			self.theConfigDB.remove(aSection)

		# gets pathname
		if not os.path.isdir( aPath ):
			aPath=os.path.dirname( aPath )

		# checks whether file exists
		aFilename=aPath+os.sep+'osogo.ini'
		if not os.path.isfile( aFilename ):
			self.printMessage('There is no osogo.ini file in this directory.\n Falling back to system defauls.\n')
			return None

	    # tries to read file

		try:
			self.printMessage('Reading osogo.ini file from directory [%s]' %aPath)
			self.theConfigDB.read( aFilename )

		# catch exceptions
		except:
			self.printMessage(' error while executing ini file [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage(anErrorMessage)


	# ==========================================================================
	def get_parameter(self, aParameter):
		"""tries to get a parameter from ConfigDB
		if the param is not present in either osogo or default section
		raises exception and quits
		"""

		# first try to get it from osogo section
		if self.theConfigDB.has_section('osogo'):
			if self.theConfigDB.has_option('osogo',aParameter):
				return self.theConfigDB.get('osogo',aParameter)

		# gets it from default
		return self.theConfigDB.get('DEFAULT',aParameter)
		    

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
			self.printMessage(aMessage)
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			self.theFileSelection.present()
			return None

		self.__deleteFileSelection()
		self.read_ini(aFileName)
		self.theSession.message('Loading %s file %s\n' %(aFileType, aFileName) )

		try:

			if aFileType == 'Model':
				self.theSession.loadModel( aFileName )

			elif aFileType == 'Script':
				self.theSession.loadScript( aFileName )

			self.theSession.theSimulator.initialize()

		except:

			# expants message window, when it is folded.
			if self['message_togglebutton'].get_active() == FALSE:
				self['message_togglebutton'].set_active(TRUE)
				self.__toggleMessageWindow( self['message_togglebutton'] ) 

			# displays confirm window
			aMessage = 'Can\'t load [%s]\nSee MessageWindow for details.' %aFileName
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			# displays message on MessageWindow
			aMessage = 'Can\'t load [%s]' %aFileName
			self.printMessage(aMessage)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage(anErrorMessage)


		self.update()
		self.updateFundamentalWindows()

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
			self.printMessage('Can\'t save [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage(anErrorMessage)

		# updates
		self.update()
		self.updateFundamentalWindows()

	# end of loadModel


	# ==========================================================================
	def deleted( self, *arg ):
		"""When delete_event happens or exit menu is selected, 
		this method is called.
		"""

		aMessage = 'Are you sure you want to quit?'
		aTitle = 'Question'

		# If simulation is running
		if self.theRunningFlag == TRUE:

			# stop simulation temporarily
			self.stopSimulation()

			# If there is no logger data, exit this program.
			if len(self.theSession.getLoggerList())==FALSE:
				mainQuit()

			else:

				# Popup confirm window, and check user request
				aDialog = ConfirmWindow(1,aMessage,aTitle)

				# ok is pressed
				if aDialog.return_result() == FALSE:
					mainQuit()
				# cancel is pressed
				else:
					self.startSimulation('')
					return TRUE

		# If simulation is not running
		else:

			# If there is no logger data, exit this program.
			if len(self.theSession.getLoggerList()) == FALSE:
				mainQuit()
			else:
				# Popup confirm window, and check user request
				aDialog = ConfirmWindow(OKCANCEL_MODE,aMessage,aTitle)

				# ok is pressed
				if aDialog.return_result() == FALSE:
					mainQuit()
				else:
				# cancel is pressed
					pass

		return TRUE


	# ==========================================================================
	def startSimulation( self, arg ) :
		"""starts simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		"""

		if self.theRunningFlag == TRUE:
			return

		try:
			self.theRunningFlag = TRUE
			# this can fail if the simulator is not ready
			self.theSession.theSimulator.initialize()

			aCurrentTime = self.theSession.getCurrentTime()
			self.theSession.message("%15s"%aCurrentTime + ":Start\n" )
			self.theTimer = gtk.timeout_add(self.theUpdateInterval, self.updateByTimeOut, FALSE)
			self.theSession.run()
			self.theRunningFlag = FALSE
			self.removeTimeOut()

		except:
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(anErrorMessage)
			self.theRunningFlag = 0


	# ==========================================================================
	def stopSimulation( self, *arg ) :
		"""stos simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		"""

		try:
			if self.theRunningFlag == TRUE:
				self.theSession.stop()

				aCurrentTime = self.theSession.getCurrentTime()
				self.theSession.message( ("%15s"%aCurrentTime + ":Stop\n" ))
				self.removeTimeOut()
				self.update()
				self.updateFundamentalWindows()
				self.theRunningFlag = FALSE

		except:
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(anErrorMessage)

		self.updateFundamentalWindows()


	# ==========================================================================
	def stepSimulation( self, arg ) : 
		"""steps simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		"""

		if self.theRunningFlag == 1:
			return

		try:
			if self.theStepSizeOrSec==0:
			    self.theStepSizeOrSec=1
			    self.theSession.message( "Zero step value overridden to 1\n" )
			self.theRunningFlag = 1
			# this can fail if the simulator is not ready
			self.theSession.theSimulator.initialize()

			self.theSession.message( "Step\n" )
			self['sec_step_entry'].set_text( str( self.theStepSizeOrSec ) )

			self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )
			#if self.theStepType == 0:
			# When 'sec' is selected.
			if self['sec_radiobutton'].get_active() == TRUE:
				self.theSession.run( float( self.theStepSizeOrSec ) )

			# When 'step' is selected.
			else:
				self.theSession.step( int( self.theStepSizeOrSec ) )

			self.theRunningFlag = 0
			self.removeTimeOut()
			self.update()
			self.updateFundamentalWindows()

		except:
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage( anErrorMessage )
			self.theRunningFlag = 0

            

	# ==========================================================================
	def setStepSizeOrSec( self, *arg ):

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
	def updateByTimeOut( self, arg ):
		"""when time out, calls updates method()
		Returns None
		"""

		self.update()
		self.updateFundamentalWindows()

		self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )


	# ==========================================================================
	def removeTimeOut( self ):
		"""removes time out
		Returns None
		"""

		gtk.timeout_remove( self.theTimer )


	# ==========================================================================
	def update( self ):
		"""updates this window 
		Returns None
		"""

		# updates time
		aTime = self.theSession.theSimulator.getCurrentTime()
		self.theCurrentTime = aTime
		self['time_entry'].set_text( str( self.theCurrentTime ) )

		# updates all plugin windows
		self.thePluginManager.updateAllPluginWindow()
        

	# ==========================================================================
	def createEntityListWindow( self, *arg ):
		"""creaets an EntityListWindow
		"""

		# when Model is already loaded.
		if len(self.theSession.theModelName) > 0:

			# creates new EntityListWindow instance
			anEntityListWindow = EntityListWindow.EntityListWindow( self )
			anEntityListWindow.openWindow()

			# saves the instance into map
			self.theEntityListInstanceMap[ anEntityListWindow ] = None
		
		# updates all fundamental windows
		self.updateFundamentalWindows()


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
			self.displayWindow('LoggerWindow')

		# When StepperWindow is selected
		elif arg[0] == self['stepper_button'] or \
		     arg[0] == self['stepper_window_menu']:
			self.displayWindow('StepperWindow')

		# When InterfaceWindow is selected
		elif arg[0] == self['interface_button'] or \
		     arg[0] == self['interface_window_menu']:
			self.displayWindow('InterfaceWindow')

		# When BoardWindow is selected
		elif arg[0] == self['board_button'] or \
		     arg[0] == self['board_window_menu']:
			self.displayWindow('BoardWindow')


	# ==========================================================================
	def displayWindow( self, aWindowName ):
		"""When the Window is not created, calls its openWidow() method.
		When already created, move it to the top of desktop.
		aWindowName   ---  window name(str)
		Return None
		[None]:When the WindowName does not matched, creates nothing.
		"""

		# When the WindowName does not matched, creates nothing.
		if self.__theFundamentalWindows.has_key( aWindowName ) == FALSE:
			print "No such WindowType (%s) " %aWindowName
			return None

		# When the Window is already created, move it to the top of desktop
		if self.__theFundamentalWindows[aWindowName].exists():
			self.__theFundamentalWindows[aWindowName].present()

		# Whend the Window is not created yet, create it.
		else:
			self.__theFundamentalWindows[aWindowName].openWindow()
			self.__theFundamentalWindows[aWindowName].update()


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
			if self.MessageWindow_attached:
				self.resize_vertically(self.MW_actual_size[1])
			else:
				self.resize_vertically(0)

		# When Message is required to be folded.
		else:

			# hide handlebox, resize window
			self['handlebox24'].hide()
			self.resize_vertically(0)
			self['message_togglebutton'].set_active(FALSE)


	# ==========================================================================
	def MW_child_attached(self,obj,obj2):
		"""MW_child_attached
		called when MessageBox is reatached to MainWindow
		must resize msgbox scrolledwindow to minimal size
		and the Mainwindow to extended size
		"""

		self['scrolledwindow1'].set_size_request(\
		                        self.MW_minimal_size[0], self.MW_minimal_size[1])
		self.resize_vertically(self.MW_actual_size[1])
		self.MessageWindow_attached=TRUE

	    
	# ==========================================================================
	def MW_child_detached(self,obj,obj2):
		"""MW_child_detached
		called when MessageBox is detached from MainWindow
		must resize msgbox scrolledwindow to actual size
		and the Mainwindow to minimalsize
		"""

		self['scrolledwindow1'].set_size_request(self.MW_actual_size[0], self.MW_actual_size[1])
		self.resize_vertically(0)
		self.MessageWindow_attached=FALSE
	        

	# ==========================================================================
	def openAbout( self, button_obj ):
		"""display the About window
		arg[0]   ---  self['about_menu']
		Return None
		"""

		# when AboutDialog is not created yet
		if self.theAboutDialog == None:
			self.theAboutDialog = gnome.ui.About( NAME, VERSION, COPYRIGHT, \
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
	def printMessage( self, aMessage ):
		"""prints message on MessageWindow
		aMessage   ---  a message (str or list)
		Return None
		"""

		# prints message on MessageWindow
		self.theMessageWindow.printMessage( aMessage )


	# ==========================================================================
	def updateFundamentalWindows( self ):
		"""updates fundamental windows
		Return None
		"""

		# updates all fundamental windows
		for aFundamentalWindow in self.__theFundamentalWindows.values():
			aFundamentalWindow.update()

		# updates all EntityListWindow
		for anEntityListWindow in self.theEntityListInstanceMap.keys():
			anEntityListWindow.update()

		# when Model is already loaded.
		if len(self.theSession.theModelName) > 0:

			# updates status of menu and button 
			self.setMenuAndButtonsStatus( TRUE )


	def getWindow( self, aWindowName ):
		"""
		aWindowName   ---  Window name (str)
		Returns Window or EntityListWindow list
		"""

		if self.__theFundamentalWindows.has_key(aWindowName):
			return self.__theFundamentalWindows[aWindowName]

		elif aWindowName == 'EntityListWindow':
			return self.theEntityListInstanceMap.keys()

		else:
			raise "(%s) does not match." %aWindowName

	# ==========================================================================
	def deleteEntityListWindow( self, anEntityListWindow ):
		"""deletes the reference to the instance of EntityListWindow
		anEntityListWindow   ---  an instance of EntityListWindow(EntityListWindow)
		Return None
		[Note]: When the argument is not anEntityListWindow, throws exception.
		        When this has not the reference to the argument, does nothing.
		"""

		# When the argument is not anEntityListWindow, throws exception.
		if anEntityListWindow.__class__.__name__ != 'EntityListWindow':
			raise "(%s) must be EntityListWindow" %anEntityListWindow

		# deletes the reference to the PropertyWindow instance on the EntityListWindow
		self.thePluginManager.deletePropertyWindowOnEntityListWinsow( anEntityListWindow.thePropertyWindow )

		# deletes the reference to the EntityListWindow instance
		if self.theEntityListInstanceMap.has_key( anEntityListWindow ):
			del self.theEntityListInstanceMap[ anEntityListWindow ]
	

# end of MainWindow




