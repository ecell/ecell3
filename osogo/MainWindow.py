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


#from Window import *
from OsogoWindow import *

from main import *
#from Plugin import *
from OsogoPluginManager import *

import gnome.ui
import gtk
#import GTK

import MessageWindow
#import PaletteWindow 
import EntityListWindow
import LoggerWindow
import InterfaceWindow 
import StepperWindow 
import ConfigParser
import string

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


# ---------------------------------------------------------------
# MainWindow -> OsogoWindow
#   - manages MainWindow
# ---------------------------------------------------------------
class MainWindow(OsogoWindow):

	# ---------------------------------------------------------------
	# Constructor
	#   - creates MessageWindow 
	#   - creates LoggerWindow 
	#   - creates InterfaceWindow 
	#   - creates PalleteWindow 
	#   - creates PluginManager 
	#   - creates Session 
	#   - adds signal to handers
	#   - creates ScriptFileSelection
	#   - creates ModelFileSelection
	#   - creates CellStateFileSelection
	#   - initialize Session
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self ):
	
		OsogoWindow.__init__( self, self, 'MainWindow.glade' )
		self.openWindow()

	# end of init

	def openWindow( self ):

		OsogoWindow.openWindow(self)

		# reads defaults from osogo.ini 
		# -------------------------------------
		self.theConfigDB=ConfigParser.ConfigParser()
    		self.theConfigDB.read(OSOGO_PATH+os.sep+'osogo.ini')
		
		
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
		self.theSession.setMessageMethod( self.theMessageWindow.printMessage )
		self.theSession.theMainWindow = self


		# -------------------------------------
		# creates LoggerWindow
		# -------------------------------------
		self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )
		#self.theLoggerWindow.openWindow()

		# -------------------------------------
		# creates InterfaceWindow
		# -------------------------------------
		self.theInterfaceWindow = InterfaceWindow.InterfaceWindow( self )
		#self.theInterfaceWindow.openWindow()

		# -------------------------------------
		# creates PluginManager
		# -------------------------------------
		self.thePluginManager = OsogoPluginManager( self )
		self.thePluginManager.loadAll()

		# -------------------------------------
		# creates PaletteManager
		# -------------------------------------
		#self.thePaletteWindow = PaletteWindow.PaletteWindow( self )
		#self.thePaletteWindow.setPluginList( self.thePluginManager.thePluginMap )

		self.theEntityChecker = 0
		self.theStepperChecker = 0

		self.theUpdateInterval = 150
		self.theStepSize = 1.0
		self.theStepType = 0
		self.theRunningFlag = 0

		# -------------------------------------
		# creates StepperWindow
		# -------------------------------------
		self.theStepperWindow = StepperWindow.StepperWindow( self.theSession , self )
#		self.theStepperWindow.openWindow()


		self.theEntityListWindowList = []
		self['MainWindow'].connect('expose-event',self.expose)

		self.theHandlerMap = \
		  { 
		# menu
		    'load_rule_menu_activate'              : self.openLoadModelFileSelection ,
			'load_script_menu_activate'            : self.openLoadScriptFileSelection ,
			'save_model_menu_activate'             : self.openSaveModelFileSelection ,
			'exit_menu_activate'                   : self.exit ,
			'message_window_menu_activate'         : self.toggleMessageWindowByMenu ,
			'interface_window_menu_activate'       : self.toggleInterfaceWindowByMenu ,
			#'palette_window_menu_activate'         : self.togglePaletteWindowByMenu ,
			'create_new_entity_list_menu_activate' : self.clickEntityListWindow ,
			'logger_window_menu_activate'          : self.toggleLoggerWindowByMenu ,
			'stepper_window_menu_activate'         : self.toggleStepperWindowByMenu ,
			'preferences_menu_activate'            : self.openPreferences ,
			'about_menu_activate'                  : self.openAbout ,
		# button
			'start_button_clicked'                 : self.startSimulation ,
			'stop_button_clicked'                  : self.stopSimulation ,
			'step_button_clicked'                  : self.stepSimulation ,
			'input_step_size'                      : self.setStepSize ,
			'step_sec_toggled'                     : self.changeStepType ,
			'entitylist_clicked'                   : self.clickEntityListWindow ,
			'logger_togglebutton_toggled'          : self.toggleLoggerWindowByButton ,
			#'palette_togglebutton_toggled'         : self.togglePaletteWindowByButton ,
			'message_togglebutton_toggled'         : self.toggleMessageWindowByButton ,
			'interface_togglebutton_toggled'       : self.toggleInterfaceWindowByButton ,
			'stepper_togglebutton_toggled'         : self.toggleStepperWindowByButton ,
			'logo_button_clicked'                  : self.openAbout,
			'delete_event'                         : self.exit
		}
		self.addHandlers( self.theHandlerMap )

		#override on_delete_event
		#aWin = self['MainWindow']
		#aWin.connect("delete_event",self.exit)

		# -------------------------------------
		# initialize for run method 
		# -------------------------------------
		self.theSession.theSimulator.setEventChecker( gtk.events_pending )
		self.theSession.theSimulator.setEventHandler( gtk.mainiteration  )
		self['ecell_logo_toolbar'].set_style( gtk.TOOLBAR_ICONS )
		self.setInitialWidgetStatus()
		self.updateFundamentalWindows()

		self[self.__class__.__name__].hide_all()
		self[self.__class__.__name__].show_all()

	# end of __init__

	def expose(self,obj,obj2):
	    #get actual dimensions of scrolledwindow1 if it is displayed
	    #and attached
	    if self.MessageWindow_attached and self.theMessageWindow.isShown:
		alloc_rect=self['scrolledwindow1'].get_allocation()
		self.MW_actual_size=[alloc_rect[2],alloc_rect[3]]
		
	def resize_vertically(self,msg_heigth): #gets messagebox heigth
	    #get fix components heigth
	    menu_heigth=self['handlebox22'].get_child_requisition()[1]
	    toolbar_heigth=max(self['handlebox19'].get_child_requisition()[1],\
		self['handlebox23'].get_child_requisition()[1],\
		self['logo_button'].get_child_requisition()[1])
	    statusbar_heigth=self['statusbar'].get_child_requisition()[1]
	    #get window_width
	    window_width=self['MainWindow'].get_size()[1]
	    #resize
	    window_heigth=menu_heigth+toolbar_heigth+statusbar_heigth+msg_heigth
	    self['MainWindow'].resize(window_width,window_heigth)
	
	def setUnSensitiveMenu( self ):
		#self['palette_window_menu'].set_sensitive(FALSE)
		self['create_new_entity_list_menu'].set_sensitive(FALSE)

	# end of setUnSensitiveMenu


	# ---------------------------------------------------------------
	# setInitialWidgetStatus
	#   - set initial status to all of the widgets on this window
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def setInitialWidgetStatus( self ):
		self['start_button'].set_sensitive(FALSE)
		self['stop_button'].set_sensitive(FALSE)
		self['step_button'].set_sensitive(FALSE)
		self['entitylist'].set_sensitive(FALSE)
		#pself['palette_togglebutton'].set_sensitive(FALSE)
		#self['palette_window_menu'].set_sensitive(FALSE)
		self['create_new_entity_list_menu'].set_sensitive(FALSE)
		self['save_model_menu'].set_sensitive(FALSE)

		self.setUnSensitiveMenu()

	# end of setInitialWidgetStatus

	# ---------------------------------------------------------------
	# read osogo.ini file
	# an osogo.ini file may be in the given path
	# that have an osogo section or others but no default
	# argument may be a filename as well
	# ---------------------------------------------------------------
	def read_ini(self,aPath):
	    #first delete every section apart from default
	    for aSection in self.theConfigDB.sections():
		self.theConfigDB.remove(aSection)
	    #gets pathname
	    if not os.path.isdir( aPath ):
		aPath=os.path.dirname( aPath )
	    #check whether file exists
	    aFilename=aPath+os.sep+'osogo.ini'
	    if not os.path.isfile( aFilename ):
		self.printMessage('There is no osogo.ini file in this directory.\n Falling back to system defauls.\n')
		return 0
	    #try to read file
	    try:
		self.printMessage('Reading osogo.ini file from directory [%s]' %aPath)
		self.theConfigDB.read( aFilename )
	    #catch exceptions
	    except:
		import sys
		import traceback
		self.printMessage(' error while executing ESS [%s]' %aFileName)
		anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
		self.printMessage("-----------")
		self.printMessage(anErrorMessage)
		self.printMessage("-----------")
	# ---------------------------------------------------------------
	# tries to get a parameter from ConfigDB
	# if the param is not present in either osogo or default section
	# raises exception and quits
	# ---------------------------------------------------------------

	def get_parameter(self, aParameter):
	    #first try to get it from osogo section
	    if self.theConfigDB.has_section('osogo'):
		if self.theConfigDB.has_option('osogo',aParameter):
		    return self.theConfigDB.get('osogo',aParameter)
	    #get it from default
	    return self.theConfigDB.get('DEFAULT',aParameter)
		    
	# ---------------------------------------------------------------
	# closeParentWindow
	#
	# anObject: a reference to widget 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def closeParentWindow( self, anObject ):

		aParentWindow = anObject.get_parent_window()
		aParentWindow.hide()

	# end of closeParentWindow


	# ---------------------------------------------------------------
	# openLoadModelFileSelection
	#
	# anObject: a reference to widget
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openLoadModelFileSelection( self, anObject ) :

		self.theModelFileSelection = gtk.FileSelection( 'Select Model File' )
		self.theModelFileSelection.ok_button.connect('clicked', self.__loadModel)
		self.theModelFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theModelFileSelection.complete( '*.' + MODEL_FILE_EXTENSION )
		self.theModelFileSelection.show_all()

	# end of openLoadModelFileSelection


	# ---------------------------------------------------------------
	# __loadModel
	#
	# button_obj: reference to button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __loadModel( self, button_obj ) :

		try:
			aFileName = self.theModelFileSelection.get_filename()

			if os.path.isfile( aFileName ):
				pass
			else:
				aMessage = ' Error ! No such file. \n[%s]' %aFileName
				self.printMessage(aMessage)
				aDialog = ConfirmWindow(0,aMessage,'Error!')
				return None

			self.theModelFileSelection.hide()
			self.read_ini(aFileName)
			self.theSession.message( 'loading rule file %s\n' % aFileName)
			aModelFile = open( aFileName, 'r' )
			self.theSession.loadModel( aModelFile )
			aModelFile.close()
			self.theSession.theSimulator.initialize()
			self.update()
			self.updateFundamentalWindows()

		except:
			self.printMessage(' can\'t load [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage("-----------")
			self.printMessage(anErrorMessage)
			self.printMessage("-----------")

	# end of loadModel


	# ---------------------------------------------------------------
	# openLoadScriptFileSelection
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openLoadScriptFileSelection( self, anObject ) :

		self.theLoadScriptFileSelection = gtk.FileSelection( 'Select Script File' )
		#self.theScriptFileSelection.set_modal(TRUE)
		self.theLoadScriptFileSelection.ok_button.connect('clicked', self.loadScript)
		self.theLoadScriptFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theLoadScriptFileSelection.complete( '*.' + SCRIPT_FILE_EXTENSION )
		self.theLoadScriptFileSelection.show_all()
        
	# end of openLoadScriptFileSelection


	# ---------------------------------------------------------------
	# loadScript
	#
	# anObject:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def loadScript( self, anObject ):

		aFileName = self.theLoadScriptFileSelection.get_filename()
		self.theLoadScriptFileSelection.hide()

		if not os.access( aFileName, os.R_OK ):
			self.printMessage('Error: loadScript: can\'t load [%s]' % aFileName)
			return

		self.theSession.message( 'loading script file %s\n' % aFileName )

		try:
			self.theSession.loadScript( aFileName )
		except:
			import sys
			import traceback
			self.printMessage(' error while executing ESS [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage("-----------")
			self.printMessage(anErrorMessage)
			self.printMessage("-----------")
		else:
			self.read_ini( aFileName )
			self.update()
			self.updateFundamentalWindows()

	# end of loadScript


	# ---------------------------------------------------------------
	# openSaveModelFileSelection
	#
	# anObject: a reference to widget
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openSaveModelFileSelection( self, anObject ) :

		self.theSaveModelFileSelection = gtk.FileSelection( 'Select Model File' )
		self.theSaveModelFileSelection.ok_button.connect('clicked', self.__saveModel)
		self.theSaveModelFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
		self.theSaveModelFileSelection.complete( '*.' + MODEL_FILE_EXTENSION )
		self.theSaveModelFileSelection.show_all()

	# end of openSaveModelFileSelection


	# ---------------------------------------------------------------
	# __saveModel
	#
	# button_obj: reference to button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __saveModel( self, button_obj ) :

		try:
			aFileName = self.theSaveModelFileSelection.get_filename()

			if os.path.isfile( aFileName ):
				aMessage = ' Would you like overwrite? \n[%s]' %aFileName
				self.printMessage(aMessage)
				aDialog = ConfirmWindow(OKCANCEL_MODE,aMessage,'Overwrite?')
				if aDialog == -1:
					return None
			else:
				pass

			self.theSaveModelFileSelection.hide()
			self.theSession.message( 'save model file %s\n' % aFileName)

			aModelFile = open( aFileName, 'w')
			self.theSession.saveModel( aModelFile )
			aModelFile.close()
			self.update()
			self.updateFundamentalWindows()

		except:
			self.printMessage(' can\'t save [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.printMessage("-----------")
			self.printMessage(anErrorMessage)
			self.printMessage("-----------")
			#self.setUnSensitiveMenu()

	# end of loadModel


	# ---------------------------------------------------------------
	# exit
	#
	# *obj:  dammy object
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def exit( self, *obj ):

		aMessage = 'Are you sure you want to quit?'
		aTitle = 'Question'

		# If simulation is running
		if self.theRunningFlag == TRUE:

			# stop simulation temporarily
			self.stopSimulation()

			# If there is no logger data, exit this program.
			if len(self.theSession.getLoggerList())==0:
				mainQuit()

			else:

				# Popup confirm window, and check user request
				aDialog = ConfirmWindow(1,aMessage,aTitle)

				# ok is pressed
				if aDialog.return_result() == 0:
					mainQuit()
				# cancel is pressed
				else:
					self.startSimulation('')
					return TRUE

		# If simulation is not running
		else:

			# If there is no logger data, exit this program.
			if len(self.theSession.getLoggerList())==0:
				mainQuit()
			else:
				# Popup confirm window, and check user request
				aDialog = ConfirmWindow(1,aMessage,aTitle)

				# ok is pressed
				if aDialog.return_result() == 0:
					mainQuit()
				else:
				# cancel is pressed
					pass

		return TRUE
        
	# end of exit


	# ---------------------------------------------------------------
	# startSimulation
	#
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def startSimulation( self, obj ) :

		if self.theRunningFlag == 1:
			return

		try:
			self.theRunningFlag = 1
			# this can fail if the simulator is not ready
			self.theSession.theSimulator.initialize()

			aCurrentTime = self.theSession.getCurrentTime()
			self.theSession.message("%15s"%aCurrentTime + ":Start\n" )
			self.theTimer = gtk.timeout_add(self.theUpdateInterval, self.updateByTimeOut, 0)
			self.theLoggerWindow.update()
			self.theSession.run()
			self.theRunningFlag = 0
			self.removeTimeOut()

		except:
			import sys
			import traceback
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(anErrorMessage)
			self.theRunningFlag = 0

	# end of startSimulation
            
	# ---------------------------------------------------------------
	# stopSimulation
	#
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def stopSimulation( self, obj=None ) :

		try:
			if self.theRunningFlag:
				self.theSession.stop()

				aCurrentTime = self.theSession.getCurrentTime()
				self.theSession.message( ("%15s"%aCurrentTime + ":Stop\n" ))
				self.removeTimeOut()
				self.update()
				self.updateFundamentalWindows()
				self.theLoggerWindow.update()
				self.theRunningFlag = 0

		except:
			import sys
			import traceback
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage(anErrorMessage)

		self.updateFundamentalWindows()

	# end of stopSimulation


	# ---------------------------------------------------------------
	# stepSimulation
	#
	# obj:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def stepSimulation( self, obj ) : 

		if self.theRunningFlag == 1:
			return

		try:

			self.theRunningFlag = 1
			# this can fail if the simulator is not ready
			self.theSession.theSimulator.initialize()

			self.theSession.message( "Step\n" )
			self['step_combo_entry'].set_text( str( self.theStepSize ) )
			self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )
			if self.theStepType == 0:
				self.theSession.run( float( self.theStepSize ) )
			else:
				self.theSession.step( int( self.theStepSize ) )
			self.theRunningFlag = 0
			self.removeTimeOut()
			self.update()
			self.updateFundamentalWindows()
			self.theLoggerWindow.update()


		except:
			import sys
			import traceback
			anErrorMessage = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
			self.theMessageWindow.printMessage( anErrorMessage )
			self.theRunningFlag = 0

	# end of stepSimulation

            
	# ---------------------------------------------------------------
	# changeStepType
	#
	# anObject:  dammy object
	#
	# return -> None
	# ---------------------------------------------------------------
	def changeStepType ( self, anObject ):

		self.theStepType = 1 - self.theStepType

	# end of changeStepType


	# ---------------------------------------------------------------
	# setStepSize
	#
	# anObject:  GtkEntry( step_combo_entry )
	#
	# return -> None
	# Notice: In this method, 'step size' is not set to the Sesstion.
	#         Only will be saved the value to self.theStepSize, and
	#         when Step button is pressed, this saved value is used.
	# ---------------------------------------------------------------
	def setStepSize( self, anObject ):

		# If the inputed charactor is not numerical charactors,
		# displays a confirm window and set 1.0 to the GtkEntry.
		try:

			# gets the inputerd characters from the GtkEntry. 
			aNumberString = string.strip( anObject.get_text() )

			if len( aNumberString ) == 0:
				# When user delete all character on the GtkEntry,
				# does nothing and keep previous value.
				pass
			else:

				# considers the case that character 'e' is included.

				# for example, '4e'
				if string.find(aNumberString, 'e') == len(aNumberString)-1:
					return None

				# for expample, '3e-' or '5e+'
				if len(aNumberString) >= 2 and \
				   ( string.find(aNumberString, 'e-') == len(aNumberString)-2 or \
				   string.find(aNumberString, 'e+') == len(aNumberString)-2 ):
					return None

				# for expample, '3e-2'
				if string.find(aNumberString, 'e-') != -1:
					anIndexNumber = aNumberString[string.find(aNumberString,'e-')+2:]
					anIndexNumber = string.atof( anIndexNumber )
					baseNumber =  aNumberString[:string.find(aNumberString,'e-')]
					if len(baseNumber) == 0:
						aNumberString = "1e-%s" %str( int(anIndexNumber) )
					self.theStepSize = string.atof( aNumberString )

				# for expample, '5e+6'
				if string.find(aNumberString, 'e+') != -1:
					anIndexNumber = aNumberString[string.find(aNumberString,'e+')+2:]
					anIndexNumber = string.atof( anIndexNumber )
					baseNumber =  aNumberString[:string.find(aNumberString,'e+')]
					if len(baseNumber) == 0:
						aNumberString = "1e+%s" %str( int(anIndexNumber) )
					self.theStepSize = string.atof( aNumberString )

				else:
					# When user input some character, tries to convert
					# it to numerical value.
					# following line is throwable except
					self.theStepSize = string.atof( aNumberString )

		except:
			# displays a Confirm Window.
			aMessage = "\"%s\" is not numerical character." %anObject.get_text()
			aMessage += "\nInput numerical character" 
			self.printMessage(aMessage)
			aDialog = ConfirmWindow(0,aMessage,'Error!')

			# set the previous value to GtkField.
			anObject.set_text(str(self.theStepSize))

	# end of setStepSize


	# ---------------------------------------------------------------
	# updateByTimeOut
	#
	# obj:  textfield
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updateByTimeOut( self, obj ):

		#self.update()
		self.update()
		self.theTimer = gtk.timeout_add( self.theUpdateInterval, self.updateByTimeOut, 0 )

	# end of updateByTimeOut


	# ---------------------------------------------------------------
	# removeTimeOut
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def removeTimeOut( self ):

		gtk.timeout_remove( self.theTimer )

	# end of removeTimeOut


	# ---------------------------------------------------------------
	# update
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def update( self ):

		#print "MainWindow.update"
		aTime = self.theSession.theSimulator.getCurrentTime()
		self.theCurrentTime = aTime
		self['time_entry'].set_text( str( self.theCurrentTime ) )
		self.thePluginManager.updateAllPluginWindow()
        
	# end of update

    
	# ---------------------------------------------------------------
	# clickEntityListWindow
	#
	# button_obj : button
	# *objects : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def clickEntityListWindow( self, *objects ):

		#fix me: this part is Root System's bug.
		#if self.theStepperChecker == 1 and self.theEntityChecker == 0:
		if self.theStepperChecker == 1:

			anEntityListWindow = EntityListWindow.EntityListWindow( self )
			self.theEntityListWindowList.append(anEntityListWindow)
			self.theEntityChecker = 1
		
		self.updateFundamentalWindows()


	# end of toggleEntityList


	# ---------------------------------------------------------------
	# toggleLoggerWindowByMenu
	#   - called when logger menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleLoggerWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindowByMenu( self, button_obj ) :

		self.theLoggerWindow.isShown = self['logger_window_menu'].active 
		self.toggleLoggerWindow()

	# end of toggleLoggerWindowByMenu


	# ---------------------------------------------------------------
	# toggleLoggerWindowByButton
	#   - called when logger menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleLoggerWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindowByButton( self, button_obj ) :

		self.theLoggerWindow.isShown = self['logger_togglebutton'].get_active()
		self.toggleLoggerWindow()

	# end of toggleLoggerWindowByButton

	# ---------------------------------------------------------------
	# toggleLoggerWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleLoggerWindow( self ):

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		#if button_obj.get_active() :
		if self.theLoggerWindow.isShown == TRUE:

			# --------------------------------------------------
			# If instance of Logger Window Widget has destroyed,
			# creates new instance of Logger Window Widget.
			# --------------------------------------------------
			if ( self.theLoggerWindow.getExist() == 0 ):
				self.theLoggerWindow.openWindow()
				#self.theLoggerWindow = LoggerWindow.LoggerWindow( self.theSession , self )
				self.theLoggerWindow.update()

			# --------------------------------------------------
			# If instance of Logger Window Widget has not destroyed,
			# calls show method of Logger Window Widget.
			# --------------------------------------------------
			else:
				self.theLoggerWindow['LoggerWindow'].hide()
				self.theLoggerWindow['LoggerWindow'].show_all()

			self['logger_togglebutton'].set_active(TRUE)
			self['logger_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Logger Window Widget has destroyed,
			# do nothing.
			# --------------------------------------------------
			if ( self.theLoggerWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Logger Window Widget has not destroyed,
			# calls hide method of Logger Window Widget.
			# --------------------------------------------------
			else:
				self.theLoggerWindow['LoggerWindow'].hide()

			self['logger_togglebutton'].set_active(FALSE)
			self['logger_window_menu'].set_active(FALSE)

	# end of toggleLoggerWindow



	# ---------------------------------------------------------------
	# toggleStepperWindowByMenu
	#   - called when stepper menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleStepperWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleStepperWindowByMenu( self, button_obj ) :

		self.theStepperWindow.isShown = self['stepper_window_menu'].active 
		self.toggleStepperWindow()

	# end of toggleStepperWindowByMenu


	# ---------------------------------------------------------------
	# toggleStepperWindowByButton
	#   - called when stepper menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleStepperWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleStepperWindowByButton( self, button_obj ) :

		self.theStepperWindow.isShown = self['stepper_togglebutton'].get_active()
		self.toggleStepperWindow()

	# end of toggleStepperWindowByButton

	# ---------------------------------------------------------------
	# toggleStepperWindow
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleStepperWindow( self ):

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		#if button_obj.get_active() :
		if self.theStepperWindow.isShown == TRUE:

			# --------------------------------------------------
			# If instance of Stepper Window Widget has destroyed,
			# creates new instance of Stepper Window Widget.
			# --------------------------------------------------
			if ( self.theStepperWindow.getExist() == 0 ):
				self.theStepperWindow.openWindow()
				#self.theStepperWindow = StepperWindow.StepperWindow( self.theSession , self )
				self.theStepperWindow.update()

			# --------------------------------------------------
			# If instance of Stepper Window Widget has not destroyed,
			# calls show method of Stepper Window Widget.
			# --------------------------------------------------
			else:
				self.theStepperWindow['StepperWindow'].hide()
				self.theStepperWindow['StepperWindow'].show_all()
				self.theStepperWindow.update()

			self['stepper_togglebutton'].set_active(TRUE)
			self['stepper_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Stepper Window Widget has destroyed,
			# do nothing.
			# --------------------------------------------------
			if ( self.theStepperWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Stepper Window Widget has not destroyed,
			# calls hide method of Stepper Window Widget.
			# --------------------------------------------------
			else:
				self.theStepperWindow['StepperWindow'].hide()

			self['stepper_togglebutton'].set_active(FALSE)
			self['stepper_window_menu'].set_active(FALSE)

	# end of toggleStepperWindow


	# ---------------------------------------------------------------
	# toggleMessageWindowByMenu
	#   - called when message menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleMessageWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindowByMenu( self, button_obj ) :

		self.theMessageWindow.isShown = self['message_window_menu'].active 
		self.toggleMessageWindow()

	# end of toggleMessageWindowByMenu


	# ---------------------------------------------------------------
	# toggleMessageWindowByMenu
	#   - called when message button is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleMessageWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindowByButton( self, button_obj ) :

		self.theMessageWindow.isShown = self['message_togglebutton'].get_active()
		self.toggleMessageWindow()

	# end of toggleMessageWindowByButton

	# ---------------------------------------------------------------
	# toggleMessageWindow
	#
	# button_obj : button or menu
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleMessageWindow( self ) :

		# ------------------------------------------------------
		# button or menu is toggled as active 
		# ------------------------------------------------------
		if self.theMessageWindow.isShown == FALSE:

			# --------------------------------------------------
			# hide handlebox, resize window
			# --------------------------------------------------
			self['handlebox24'].hide()
			self.resize_vertically(0)


			self['message_togglebutton'].set_active(FALSE)
			self['message_window_menu'].set_active(FALSE)

		# ------------------------------------------------------
		# button or menu is toggled as non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# show handlebox, resize window			# 
			# --------------------------------------------------
			self['handlebox24'].show()
			if self.MessageWindow_attached:
			    self.resize_vertically(self.MW_actual_size[1])
			else:
			    self.resize_vertically(0)

			self['message_togglebutton'].set_active(TRUE)
			self['message_window_menu'].set_active(TRUE)


	# end of toggleMessageWindow

	# ---------------------------------------------------------------
	# MW_child_attached
	# called when MessageBox is reatached to MainWindow
	# must resize msgbox scrolledwindow to minimal size
	# and the Mainwindow to extended size
	# ---------------------------------------------------------------
	
	def MW_child_attached(self,obj,obj2):
	    self['scrolledwindow1'].set_size_request(self.MW_minimal_size[0],\
		    self.MW_minimal_size[1])
	    self.resize_vertically(self.MW_actual_size[1])
	    self.MessageWindow_attached=TRUE

	# ---------------------------------------------------------------
	# MW_child_detached
	# called when MessageBox is detached from MainWindow
	# must resize msgbox scrolledwindow to actual size
	# and the Mainwindow to minimalsize
	# ---------------------------------------------------------------
	    
	def MW_child_detached(self,obj,obj2):
	    self['scrolledwindow1'].set_size_request(self.MW_actual_size[0],\
		    self.MW_actual_size[1])
	    self.resize_vertically(0)
	    self.MessageWindow_attached=FALSE
	        
	# ---------------------------------------------------------------
	# openAbout
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openAbout( self, button_obj ):

		anAboutDialog = gnome.ui.About( NAME,
		                                     VERSION,
		                                     COPYRIGHT,
						     DESCRIPTION,
		                                     AUTHORLIST)
		anAboutDialog.set_title( 'about osogo' )
		anAboutDialog.show_all()

	# end of openAbout


	# ---------------------------------------------------------------
	# openPreference
	#
	# button_obj : button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openPreferences( self, button_obj ):

		#aPropertyBox = gnome.ui.GnomePropertyBox()
		aPropertyBox = gnome.ui.PropertyBox()
		aLabel = gtk.Label( 'NOT IMPLEMENTED YET' )
		aTabLabel = gtk.Label( 'warning' )
		aPropertyBox.append_page( aLabel, aTabLabel )

		#aPropertyBox = gnome.ui.GnomePropertyBox()
		#aLabel = gtk.Label( 'NOT IMPLEMENTED YET' )
		#aTabLabel = gtk.Label( 'warning' )
		#aPropertyBox.append_page( aLabel, aTabLabel )
		#aPropertyBox.hide()
		#aPropertyBox.show_all()
		aMessage = ' Sorry ! Not implemented... [%s]\n' %'06/Mar/2003'
		self.printMessage(aMessage)
		aDialog = ConfirmWindow(0,aMessage,'Sorry!')
		return None

	# end of openPreference

	# ---------------------------------------------------------------
	# toggleInterfaceWindowByMenu
	#   - called when interface menu is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleInterfaceWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceWindowByMenu( self, *objects ) :

		self.theInterfaceWindow.isShown = self['interface_window_menu'].active 
		self.toggleInterfaceWindow()

	# end of toggleInterfaceWindowByMenu


	# ---------------------------------------------------------------
	# toggleInterfaceWindowByButton
	#   - called when interface button is toggled.
	#   - sets the "isShown" attribute.
	#   - calls toggleInterfaceWindow
	#
	# *objects : dammy objects 
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceWindowByButton( self, *objects ) :

		self.theInterfaceWindow.isShown = self['interface_togglebutton'].get_active()
		self.toggleInterfaceWindow()

	# end of toggleInterfaceWindowByButton


	# ---------------------------------------------------------------
	# toggleInterfaceWindow
	#   - show or hide InterfaceWindow according to "isShown" attribute.
	#
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def toggleInterfaceWindow( self ) :

		# ------------------------------------------------------
		# button is toggled to active 
		# ------------------------------------------------------
		if self.theInterfaceWindow.isShown == TRUE:

			# --------------------------------------------------
			# If instance of Interface Window Widget has destroyed,
			# creates new instance of Interface Window Widget.
			# --------------------------------------------------
			if ( self.theInterfaceWindow.getExist() == 0 ):
				self.theInterfaceWindow.openWindow()
				self.theInterfaceWindow.update()

			# --------------------------------------------------
			# If instance of Interface Window Widget has not destroyed,
			# calls show method of Interface Window Widget.
			# --------------------------------------------------
			else:
				self.theInterfaceWindow['InterfaceWindow'].hide()
				self.theInterfaceWindow['InterfaceWindow'].show_all()

			self['interface_togglebutton'].set_active(TRUE)
			self['interface_window_menu'].set_active(TRUE)

		# ------------------------------------------------------
		# button is toggled to non-active
		# ------------------------------------------------------
		else:

			# --------------------------------------------------
			# If instance of Message Window Widget has destroyed,
			# does nothing.
			# --------------------------------------------------
			if ( self.theInterfaceWindow.getExist() == 0 ):
				pass

			# --------------------------------------------------
			# If instance of Interface Window Widget has not destroyed,
			# calls hide method of Interface Window Widget.
			# --------------------------------------------------
			else:
				self.theInterfaceWindow['InterfaceWindow'].hide()

			self['interface_togglebutton'].set_active(FALSE)
			self['interface_window_menu'].set_active(FALSE)

	# end of toggleInterfaceListWindow


	# ---------------------------------------------------------------
	# saveCellStateToTheFile ( NOT IMPLEMENTED ON 08/Jul/2002)
	#
	# *objects : dammy objects
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def saveCellStateToTheFile( self, *objects ):

		aMessage = ' Sorry ! Not implemented... [%s]\n' %'08/Jul/2002'
		self.printMessage(aMessage)
		aDialog = ConfirmWindow(0,aMessage,'Sorry!')
		return None

	# end of saveCellStateToTheFile


	# ---------------------------------------------------------------
	# printMessage
	#
	# aMessage : message
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def printMessage( self, aMessage ):

		self.theMessageWindow.printMessage( aMessage )

	# end of printMessage


	# ---------------------------------------------------------------
	# updateFundamentalWindows
	#  - update MessageWindow, LoggerWindow, InterfaceWindow
	#  - update status of each menu and button
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def updateFundamentalWindows( self ):

		# -------------------------------------------
		# calls update method of each Window
		# -------------------------------------------

#		self.theMessageWindow.update()
		self.theLoggerWindow.update()
		self.theInterfaceWindow.update()
		self.theStepperWindow.update()

		# -------------------------------------------
		# checks buttons  ane menus
		# -------------------------------------------
		if self.getExist() == FALSE:
			pass
			#self.exit()
		else:

			# checks message button
			if self.theMessageWindow.isShown == TRUE:
				self['message_togglebutton'].set_active(TRUE)
				self['message_window_menu'].set_active(TRUE)
			else:
				self['message_togglebutton'].set_active(FALSE)
				self['message_window_menu'].set_active(FALSE)

			# checks logger button
			if self.theLoggerWindow.isShown == TRUE:
				self['logger_togglebutton'].set_active(TRUE)
				self['logger_window_menu'].set_active(TRUE)
			else:
				self['logger_togglebutton'].set_active(FALSE)
				self['logger_window_menu'].set_active(FALSE)

			# checks stepper button
			if self.theStepperWindow.isShown == TRUE:
				self['stepper_togglebutton'].set_active(TRUE)
				self['stepper_window_menu'].set_active(TRUE)
			else:
				self['stepper_togglebutton'].set_active(FALSE)
				self['stepper_window_menu'].set_active(FALSE)


			# checks interface button
			if self.theInterfaceWindow.isShown == TRUE :
				self['interface_togglebutton'].set_active(TRUE)
				self['interface_window_menu'].set_active(TRUE)
			else:
				self['interface_togglebutton'].set_active(FALSE)
				self['interface_window_menu'].set_active(FALSE)

			# entity window
			# detects the destroyed EntityWindows, and delete them from
			# self.theEntityListWindow.
			aDeleteIndexList = []
			for anIndex in range(0,len(self.theEntityListWindowList)):
				anEntityListWindow = self.theEntityListWindowList[anIndex]

				if anEntityListWindow.getExist() == TRUE:
					anEntityListWindow.update()
				else:
					aDeleteIndexList.append(anIndex)

			aDeleteIndexList.sort()
			aDeleteIndexList.reverse()
			for anIndex in aDeleteIndexList:
				del self.theEntityListWindowList[anIndex]


		# When model file is loaded
		if self.theSession.theModelName != "":
			self.theStepperChecker = TRUE

			self['start_button'].set_sensitive(TRUE)
			self['stop_button'].set_sensitive(TRUE)
			self['step_button'].set_sensitive(TRUE)
			self['entitylist'].set_sensitive(TRUE)
			#self['palette_togglebutton'].set_sensitive(TRUE)
			#self['palette_window_menu'].set_sensitive(TRUE)
			self['create_new_entity_list_menu'].set_sensitive(TRUE)
			self['load_rule_menu'].set_sensitive(FALSE)
			self['load_script_menu'].set_sensitive(FALSE)
			self['save_model_menu'].set_sensitive(TRUE)

	# end of updateFundamentalWindow


# end of MainWindow







