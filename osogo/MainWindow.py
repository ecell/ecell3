#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-Cell Session Monitor package
#
#                Copyright (C) 2001-2004 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-Cell is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-Cell -- see the file COPYING.
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
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#


from OsogoWindow import *
from AboutSessionMonitor import *

from main import *


import gtk
import gobject

import MessageWindow

import string
import sys
import traceback
import os
import math
import time
import datetime

if GNOME_INSTALLED =='yes':
    import gnome.ui

#
#import pyecell module
#
from ecell.GtkSessionMonitor import *
from ecell.ecssupport import *
from ecell.ecs_constants import *
from ConfirmWindow import *



class SimulationButton:

	def __init__( self ):

                # Image
                self.startImage = os.environ['OSOGOPATH'] + os.sep + "icon_start.png"
                self.stopImage = os.environ['OSOGOPATH'] + os.sep + "icon_stop.png"

		self.image = gtk.Image()
		self.image.set_from_file( self.startImage )
                self.__currentState = 'stop'
		self.image.show()

	def getCurrentImage( self ):

		return self.image

        def getCurrentState( self ):

                return self.__currentState

        def setCurrentState( self, aCurrentState ):

                self.__currentState = aCurrentState
            
                if ( self.__currentState == 'run' ):

                    self.image.set_from_file( self.stopImage )

                elif ( self.__currentState == 'stop' ):

                    self.image.set_from_file( self.startImage )


class LogoAnimation:

    def __init__( self ):


        #self.image = []
        
        self.image = gtk.Image()
        self.image.set_from_file( os.environ['OSOGOPATH'] + os.sep + "ecell32.png" )
        self.image.show()

        self.iconList = [ os.environ['OSOGOPATH'] + os.sep + "ecell32-1.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-2.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-3.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-4.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-5.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-6.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-7.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32-8.png",
                          os.environ['OSOGOPATH'] + os.sep + "ecell32.png" ]

        #for i in range( len( self.iconList ) ): 
        #    self.image.append( gtk.Image() )
        #    self.image[i].set_from_file( self.iconList[i] )

        self.START_ROTATION = 0
        self.END_ROTATION = len( self.iconList) - 1

        self.__currentImage = 0
        self.__running = False

        self.extraCount = 0
        self.delay = 100
        self.theTimer = None

        #def setLogoWidget( self, aLogoAnimationWidget ):

        #self.widget = aLogoAnimationWidget
        #self.widget.add( self.image[self.START_ROTATION] )
        #self.widget.add( self.image[self.END_ROTATION] )

    def getImage( self ):

        return self.image


    def start( self ):

        self.__currentImage = 0
        self.__running = True

        self.animate()


    def stop( self ):

        self.__running = False
        

    def animate( self ):
        
        if ( self.__running ):

            if ( self.__currentImage == self.END_ROTATION ):

                if( self.extraCount == 5 ):
                    self.__currentImage = self.START_ROTATION
                    self.extraCount = 0
                else:
                    self.__currentImage = self.END_ROTATION - 1
                    self.extraCount += 1


            self.image.set_from_file( self.iconList[self.__currentImage] )
            self.__currentImage += 1

            self.theTimer = gtk.timeout_add( self.delay, LogoAnimation.animate, self )

            
            #if ( self.__currentImage == self.END_ROTATION ):

            #    self.widget.remove( self.image[self.__currentImage-1] )
            #    self.__currentImage = self.START_ROTATION                
                
            #elif ( self.__currentImage == self.START_ROTATION ):

            #    self.widget.remove( self.image[self.END_ROTATION] )
            #    self.__currentImage += 1
            #else:
            #    self.widget.remove( self.image[self.__currentImage-1] )
            #    self.__currentImage += 1

            #print self.image[self.__currentImage]
            #self.widget.add( self.image[self.__currentImage] )
            #gobject.timeout_add( self.delay, LogoAnimation.animate, self )

        else:
            if ( self.__currentImage != self.END_ROTATION ):
                #self.widget.add( self.image[self.__currentImage] )

                self.image.set_from_file( self.iconList[self.__currentImage] )
                self.__currentImage += 1
                self.theTimer = gtk.timeout_add( 60, LogoAnimation.animate, self )
            else:
                if self.theTimer != None:
                    gtk.timeout_remove( self.theTimer )
                    self.theTimer = None
                    


class MainWindow(OsogoWindow):
	"""MainWindow
	"""

	def __init__( self, aSession ):
	
		# calls super class's constructor
		OsogoWindow.__init__( self, self, 'MainWindow.glade' )

                # -------------------------------------
		# stores pointer to Session
		# -------------------------------------
		self.theSession = aSession
		self.theMessageWindow = MessageWindow.MessageWindow()


                # initialize Timer components
                self.startTime = 0
                self.tempTime = 0
                self.isStarted = False
                self.timerVisible = False
                self.theLastTime = 0
                self.theLastRealTime = 0                
                self.updateCount = 0

                # initialize Indicator
                self.indicatorVisible = False

                # create datetime instance for Timer
                self.datetime = datetime.datetime(1970,1,1)


                
	def openWindow( self ):
		# calls superclass's method
		OsogoWindow.openWindow(self)

		self.__button_update = False
		

                self.logoMovable = True
                self.theToolbarVisible = True
                self.theStatusbarVisible = True
		self.theMessageWindowVisible = True
                self.theEntityListWindowVisible = True
                
		# -------------------------------------
		# create SimulationButton
		# -------------------------------------
                
		self.SimulationButton = SimulationButton()
		self['SimulationButton'].add( self.SimulationButton.getCurrentImage() )
		self['SimulationButtonLabel'].set_text('Start')


                # ---------------------------
                # create logo button
                # ---------------------------

                self.logoAnimation = LogoAnimation()
                self['logo_animation'].add( self.logoAnimation.getImage() )
                

                # --------------------------
                # initialize time entry
                # --------------------------

                self['sec_step_entry'].set_property( 'xalign', 1 )
                self['time_entry'].set_property( 'xalign', 1 )
                self['time_entry'].modify_base( gtk.STATE_NORMAL,
                                                gtk.gdk.Color(61000,61000,61000,0) )


		# -------------------------------------
		# creates MessageWindow 
		# -------------------------------------
		self.theMessageWindow.openWindow()
                self['messagearea'].add(self.theMessageWindow['top_frame'])
                
                self.theSession.setMessageMethod( self.__printMessage )
		self.__expose(None,None)
		messageWindowSize=self.theMessageWindow.getActualSize()
		self.theMessageWindow['scrolledwindow1'].set_size_request(\
		    messageWindowSize[0], messageWindowSize[1] )


		# -------------------------------------
		# append signal handlers
		# -------------------------------------
		self.theHandlerMap =  { 
		    # menu
                    'load_model_menu_activate'    : self.__openFileSelection,
                    'load_script_menu_activate'   : self.__openFileSelection,
                    'save_model_menu_activate'    : self.__openFileSelection,
                    'exit_menu_activate'          : self.__deleted,
                    'message_window_menu_activate': self.__toggleMessageWindow,
                    'entitylist_window_menu_activate': self.__toggleEntityListWindow,
                    'interface_window_menu_activate'
                    : self.__displayWindow ,
                    'entity_list_menu_activate'
                    : self.__createEntityListWindow ,
                    'logger_window_menu_activate' : self.__displayWindow,
                    'stepper_window_menu_activate': self.__displayWindow,
                    'board_window_menu_activate'  : self.__displayWindow,
                    #'preferences_menu_activate'   : self.openPreferences,
                    'about_menu_activate'         : self.__displayAbout,

                    #sbml
                    'on_import_sbml_activate'    : self.__openFileSelection,
                    'on_export_sbml_activate'   : self.__openFileSelection,
                    
                    # toolbars
                    'simulation_button_clicked'        : self.__handleSimulation,
                    'step_button_clicked'           : self.__stepSimulation,
                    
                    'on_sec_step_entry_activate'    : self.__setStepSizeOrSec,
                    'on_timer_clear_button_clicked' : self.__clearTimer,
                    'on_load_model_button_clicked'  : self.__openFileSelection,
                    'on_load_script_button_clicked' : self.__openFileSelection,
                    'on_save_model_button_clicked'  : self.__openFileSelection,
                    
                    'on_entitylist_button_clicked'
                    : self.__createEntityListWindow,
                    'on_logger_button_toggled'    : self.__displayWindow,
                    'on_message_togglebutton_toggled'
                    : self.__toggleMessageWindow,
                    'on_stepper_button_toggled'   : self.__displayWindow,
                    'on_interface_button_toggled' : self.__displayWindow,
                    'on_board_button_toggled'     : self.__displayWindow,
                    'logo_button_clicked'         : self.__displayAbout,
                    'on_timer_button_toggled'     : self.__displayTimer,
                    'on_indicator_button_toggled' : self.__displayIndicator,
                    'on_scrolledwindow1_expose_event'
                    : self.__expose,

                    # view
                    'on_toolbar_menu_activate'        : self.__displayToolbar,
                    'on_statusbar_menu_activate'      : self.__displayStatusbar,
                    'on_run_speed_indicator_activate' : self.__displayIndicator,  
                    'on_timer_activate'               : self.__displayTimer,
                    'on_logo_animation_menu_activate' : self.__setAnimationSensitive,
                    'on_logging_policy1_activate' : self.__openLogPolicy
                    }
                
		self.addHandlers( self.theHandlerMap )
                self.setIconList(
                    os.environ['OSOGOPATH'] + os.sep + "ecell.png",
                    os.environ['OSOGOPATH'] + os.sep + "ecell32.png" )
                

		self.__setMenuAndButtonsStatus( FALSE )
		#self.theSession.updateFundamentalWindows()


		# display MainWindow
		self[self.__class__.__name__].show_all()
		self.present()

		self.theStepSizeOrSec = 1

		# initializes FileSelection reference
		self.theFileSelection = None

		# initializes AboutDialog reference
		self.theAboutSessionMonitor = None
		self.openAboutSessionMonitor = False 

		self.update()
		# -------------------------------------
		# creates EntityListWindow 
		# -------------------------------------

                self.theEntityListWindow = self.theSession.createEntityListWindow( 'top_frame', self['statusbar'] )
                self['entitylistarea'].add( self.theEntityListWindow['top_frame'] )


                # --------------------
                # set Timer entry
                # --------------------
                self['timer_entry'].set_text( str( 0 ) )
                self['timer_entry'].modify_base( gtk.STATE_NORMAL,
                                                gtk.gdk.Color(61000,61000,61000,0) )
                self['timer_entry'].set_property( 'xalign', 1 )
                self['timer_box'].hide()



                #self['run_speed_entry'].modify_base( gtk.STATE_NORMAL,
                #gtk.gdk.Color(61000,61000,61000,0) )
                #self['run_speed_entry'].set_property('xalign', 1)

                
                # ---------------------
                # initialize Indicator
                # ---------------------
                
                self['indicator_box'].hide()


	def __expose( self, *arg ):
		"""expose
		Return None
		"""
                pass


	def __resizeVertically( self, height ): #gets entitylistarea or messagebox height
		"""__resizeVertically
		Return None
		"""

		# gets fix components height
		menu_height=self['handlebox22'].get_child_requisition()[1]
		toolbar_height=self['handlebox19'].get_child_requisition()[1]

		# gets window_width
		window_width=self['MainWindow'].get_size()[0]

		# resizes
		window_height=menu_height+toolbar_height+height
		self['MainWindow'].resize(window_width,window_height)
	

	def __setMenuAndButtonsStatus( self, aDataLoadedStatus ):
		"""sets initial widgets status
		aDataLoadedStatus  -- the status of loading data
		         (TRUE:Model or Script is loaded / FALSE:Not loaded)
		Returns None
		"""

		# toolbar
		self['simulation_button'].set_sensitive(aDataLoadedStatus)
		self['step_button'].set_sensitive(aDataLoadedStatus)
		self['timer_clear_button'].set_sensitive(aDataLoadedStatus)
		self['load_model_button'].set_sensitive(not aDataLoadedStatus)
		self['load_script_button'].set_sensitive(not aDataLoadedStatus)
		self['save_model_button'].set_sensitive(aDataLoadedStatus)
		self['entitylist_button'].set_sensitive(aDataLoadedStatus)
		self['logger_button'].set_sensitive(aDataLoadedStatus)
		self['stepper_button'].set_sensitive(aDataLoadedStatus)
		self['interface_button'].set_sensitive(aDataLoadedStatus)
		self['board_button'].set_sensitive(aDataLoadedStatus)
                self['indicator_button'].set_sensitive(aDataLoadedStatus)
                self['timer_button'].set_sensitive(aDataLoadedStatus)

		# file menu
		self['load_model_menu'].set_sensitive(not aDataLoadedStatus)
		self['load_script_menu'].set_sensitive(not aDataLoadedStatus)
		self['save_model_menu'].set_sensitive(aDataLoadedStatus)
		self['import_sbml'].set_sensitive(not aDataLoadedStatus)
                self['export_sbml'].set_sensitive(aDataLoadedStatus)

		# window menu
		self['logger_window_menu'].set_sensitive(aDataLoadedStatus)
		self['stepper_window_menu'].set_sensitive(aDataLoadedStatus)
		self['interface_window_menu'].set_sensitive(aDataLoadedStatus)
		self['board_window_menu'].set_sensitive(aDataLoadedStatus)
		self['entity_list_menu'].set_sensitive(aDataLoadedStatus)
		self['save_model_menu'].set_sensitive(aDataLoadedStatus)

                # preferences menu
                self['logging_policy'].set_sensitive(aDataLoadedStatus)
                self['run_speed_indicator'].set_sensitive(aDataLoadedStatus)
                self['timer_menu'].set_sensitive(aDataLoadedStatus)
                self['logo_animation_menu'].set_sensitive(aDataLoadedStatus)

	def __openFileSelection( self, *arg ) :
		"""checks argument and calls self.openFileSelection() method.
		arg[0]  ---  self['load_model_menu'] /
                self['load_script_menu'] /
                self['save_model_menu']

		Return None
		[Note]: When the FileSelection is already displayed,
                moves it to the top of desktop.
		"""

		# checks the length of argument, but this is verbose
		if len( arg ) < 1:
			return None

		# when 'Load Model' is selected
		if arg[0] == self['load_model_menu'] or \
                   arg[0] == self['load_model_button']:
			self.openFileSelection('Load','Model')

		# when 'Load Script' is selected
		elif arg[0] == self['load_script_menu'] or \
                     arg[0] == self['load_script_button']:
			self.openFileSelection('Load','Script')

		# when 'Save Model' is selected
		elif arg[0] == self['save_model_menu'] or \
                     arg[0] == self['save_model_button']:
			self.openFileSelection('Save','Model')

                # when 'Import SBML' is selected
                elif arg[0] == self['import_sbml']:
			self.openFileSelection('Load','SBML')

                # when 'Export SBML' is selected
                elif arg[0] == self['export_sbml']:
			self.openFileSelection('Save','SBML')

        def openFileSelection( self, aType, aTarget ) :
		"""displays FileSelection 
		aType     ---  'Load'/'Save' (str)
		aTarget   ---  'Model'/'Script'/'SBML' (str)
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
                        aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                              os.environ['OSOGOPATH'] + os.sep + 'ecell.png')
                        aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                              os.environ['OSOGOPATH'] + os.sep + 'ecell32.png')
                        self.theFileSelection.set_icon_list(
                                        aPixbuf16, aPixbuf32)

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

			# when 'Import SBML' is selected
			elif aType == 'Load' and aTarget == 'SBML':
				self.theFileSelection.ok_button.connect('clicked', self.__loadData, aTarget)
				self.theFileSelection.complete( '*.'+ MODEL_FILE_EXTENSION )
				self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )

			# when 'Save Model' is selected
			elif aType == 'Save' and aTarget == 'SBML':
				self.theFileSelection.ok_button.connect('clicked', self.__exportSBML)
				self.theFileSelection.complete( '*.'+ MODEL_FILE_EXTENSION )
				self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )

			else:
				raise "(%s,%s) does not match." %(aType,aTarget)

			# displays the created FileSelection
			self.theFileSelection.show_all()


	def __deleteFileSelection( self, *arg ):
		"""deletes FileSelection
		Return None
		"""

		# deletes the reference to FileSelection
		if self.theFileSelection != None:
			self.theFileSelection.destroy()
			self.theFileSelection = None


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
				# self.theSession.theSimulator.initialize()
			elif aFileType == 'Script':
				self.theSession.loadScript( aFileName )
                        elif aFileType == 'SBML':
                                self.theSession.importSBML( aFileName )
                                
                        self.theEntityListWindow.updateButtons()

                        self.update()
                        self.theSession.updateFundamentalWindows()
                        self.theEntityListWindow.theQueue.applyFullPNList()
		except:
                        # set load command not to be operated 
                        self['load_model_button'].set_sensitive(0)
                        self['load_script_button'].set_sensitive(0)
                        self['load_model_menu'].set_sensitive(0)
                        self['load_script_menu'].set_sensitive(0)

                        # expants message window, when it is folded.
			if self.exists():
				if ( self['message_togglebutton'].get_child() ).get_active() == FALSE:
					( self['message_togglebutton'].get_child() ).set_active(TRUE)

			# displays confirm window
			aMessage = 'Can\'t load [%s]\nSee MessageWindow for details.' %aFileName
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			# displays message on MessageWindow
			aMessage = 'Can\'t load [%s]' %aFileName
			self.theSession.message(aMessage)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type, sys.exc_value, \
					sys.exc_traceback), '\n' )
			self.theSession.message(anErrorMessage)



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
			if ( self['message_togglebutton'].get_child() ).get_active() == FALSE:
				( self['message_togglebutton'].get_child() ).set_active(TRUE)


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

	def __exportSBML( self, *arg ) :

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
			self.theSession.message('Export SBML file %s\n' % aFileName)                        
			# saves Model
			self.theSession.exportSBML( aFileName )

		except:

			# expants message window, when it is folded.
			if ( self['message_togglebutton'].get_child() ).get_active() == FALSE:
				( self['message_togglebutton'].get_child() ).set_active(TRUE)

			# displays confirm window
			aMessage = 'Can\'t save [%s]\nSee MessageWindow for details.' %aFileName
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

			# displays error message of MessageWindow
			self.theSession.message('Can\'t export [%s]' %aFileName)
			anErrorMessage = string.join( traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback), '\n' )
			self.theSession.message(anErrorMessage)

		# updates
		self.update()
		self.updateButtons()
		self.theSession.updateFundamentalWindows()


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

		self.setStopState()

		self.close()

		self.theSession.QuitGUI()

		return gtk.TRUE


	def close( self ):
		""" restores message method and closes window """
		self.theSession.restoreMessageMethod()

                if len( self.theSession.theModelName ) > 0:
                    self.theEntityListWindow.update()                    

                self.theEntityListWindow.close()
		OsogoWindow.close( self )



        def setStartState( self ):

            self.SimulationButton.setCurrentState( 'run' )
            self['SimulationButtonLabel'].set_text('Stop')

            if self.logoMovable:
                self.logoAnimation.start()

            self.isStarted = True
            self.startTime = time.time()

        def setStopState( self ):
            
            self.SimulationButton.setCurrentState( 'stop' )
            self['SimulationButtonLabel'].set_text('Start')
            self.logoAnimation.stop()
            
            self.setTempTime()


        def setTempTime( self ):

            self.tempTime = time.time() - self.startTime + self.tempTime


	def __handleSimulation( self, *arg ) :
		"""handles simulation
		arg[0]  ---  simulation button (gtk.Button)
		Returns None
		"""
                if not self.exists():
                    return

                if ( self.SimulationButton.getCurrentState() == 'stop' ):

                    self.setStartState()
                    self.theSession.run()

                elif ( self.SimulationButton.getCurrentState() == 'run' ):

                    self.setStopState()
                    self.theSession.stop()
                

	def handleSimulation( self ) :
		""" handles simulation """
		self.__handleSimulation( self, None )


	def __stepSimulation( self, *arg  ) : 
		"""steps simulation
		arg[0]  ---  stop button (gtk.Button)
		Returns None
		if step measure is second, then Session.run()
		if step measure is step than Session.step ()
		"""
		if self.getStepType():

                    if( self.SimulationButton.getCurrentState() == "stop" ):
                        self.setStartState()
                        self.theSession.run( self.getStepSize() )
                        self.setStopState()

                    else:                        
                        self.theSession.run( self.getStepSize() )
                        self.setTempTime()
                    
		else:
                    if( self.SimulationButton.getCurrentState() == "stop" ):
                        self.setStartState()
                        self.theSession.step( self.getStepSize() )
                        self.setStopState()
                    else:
                        self.theSession.step( self.getStepSize() )
                        self.setTempTime()
                        
                        
	def stepSimulation( self ) :
		""" steps simulation """
		self.__stepSimulation( self, None )


	def getStepType( self ):
		""" returns state of sec radiobutton
			gtk.TRUE: seconds
			gtk.FALSE: steps
		"""
		return self['sec_radiobutton'].get_active()

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


	def getStepSize( self ):
		""" returns user or script specifid step size """
		self.__setStepSizeOrSec( self, None )
	
		return self.theStepSizeOrSec

	def setStepSize( self, num ):
		""" sets Stepsize entry box to num """

		self['sec_step_entry'].set_text( str (num) )
		self.__setStepSizeOrSec( self, None )

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
			

        #def getTimeSubstruction( self, aTime1, aTime )

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


                if ( self.SimulationButton.getCurrentState() == 'run' and
                     self.timerVisible ):
                    self['timer_entry'].set_text( self.getCurrentTime( time.time() - self.startTime + self.tempTime ) )



                   
                if self.indicatorVisible:

                    if( self.updateCount == 0 ):
                        self.theLastTime = self.theCurrentTime
                        self.theLastRealTime = time.time()

                    self.updateCount += 1
                    if ( self.updateCount == 25 ):
                        if ( aTime != self.theLastTime ):
                            self['run_speed_label'].set_text(
                                str( round( ( self.theCurrentTime - self.theLastTime ) /
                                            ( time.time() - self.theLastRealTime ), 5 ) ) )
                        

                        self.updateCount = 0
                

		# when Model is already loaded.
		if len(self.theSession.theModelName) > 0:
			# updates status of menu and button 
			self.__setMenuAndButtonsStatus( TRUE )

			self.updateButtons()


        def getCurrentTime( self, aTime ):

            theTime = self.datetime.fromtimestamp( aTime )

            return str( theTime.hour-9 )+" : "+str( theTime.minute ) + " : " + str( theTime.second )


	def updateButtons( self ):
		""" updates Buttons and menus with 
		latest FundamentalWindow status
		"""
		
		if not self.exists():
			return

		self.__button_update = True

		# boardwindow:
		if self.theSession.doesExist('BoardWindow' ):
			flag = gtk.TRUE
		else:
			flag = gtk.FALSE
		self['board_window_menu'].set_active( flag )
		( self['board_button'].get_child() ).set_active(flag)

		# Loggerwindow:
		if self.theSession.doesExist('LoggerWindow' ):
			flag = gtk.TRUE
		else:
			flag = gtk.FALSE
		self['logger_window_menu'].set_active( flag )
		( self['logger_button'].get_child() ).set_active(flag)
			
		# interface window:
		if self.theSession.doesExist('InterfaceWindow' ):
			flag = gtk.TRUE
		else:
			flag = gtk.FALSE
		self['interface_window_menu'].set_active( flag )
		( self['interface_button'].get_child() ).set_active(flag)

		# stepperwindow:
		if self.theSession.doesExist('StepperWindow' ):
			flag = gtk.TRUE
		else:
			flag = gtk.FALSE
		self['stepper_window_menu'].set_active( flag )
		( self['stepper_button'].get_child() ).set_active(flag)


	
		#MessageWindow
		if self.theMessageWindowVisible:
			flag = gtk.TRUE
		else:
			flag = gtk.FALSE
		self['message_window_menu'].set_active(flag)
		( self['message_togglebutton'].get_child() ).set_active( flag )


		self.__button_update = False

	def __openLogPolicy( self, *arg):
		"""
		signal handler to logging policy is called
		"""
		# get default logging policy
		aLogPolicy = self.theSession.getLogPolicyParameters()
		
		# open logpolicy window
		newLogPolicy = self.theSession.openLogPolicyWindow( aLogPolicy, "Set default log policy" )
		if newLogPolicy != None:
			# save logpolicy
			self.theSession.setLogPolicyParameters( newLogPolicy )


        def __displayToolbar( self, *arg ):
            # show Toolbar

            if self.theToolbarVisible:
                self['toolbar_handlebox'].hide()
                self.theToolbarVisible = False

                if self.theMessageWindowVisible == False and \
                   self.theEntityListWindowVisible == False:

                       self.__resizeVertically( 0 )

            else:
                self['toolbar_handlebox'].show()
                self.theToolbarVisible = True



        def __displayStatusbar( self, *arg ):
            # show Statusbar

            if self.theStatusbarVisible:
                self['statusbar'].hide()
                self.theStatusbarVisible = False

                if self.theMessageWindowVisible == False and \
                   self.theEntityListWindowVisible == False:

                       self.__resizeVertically( 0 )

            else:
                self['statusbar'].show()
                self.theStatusbarVisible = True
                

        def __displayIndicator( self, *arg ):
            # show Indicator

            if self.indicatorVisible:
                self['indicator_box'].hide()
                self.setIndicatorActive(False)
                self.indicatorVisible = False
            else:
                self['indicator_box'].show()
                self.setIndicatorActive(True)
                self.indicatorVisible = True

		self.theLastTime = self.theSession.theSimulator.getCurrentTime()

                
        def __displayTimer( self, *arg ):
            # show Indicator

            if self.timerVisible:
                self['timer_box'].hide()
                self.setTimerActive(False)
                self.timerVisible = False
            else:
                self['timer_box'].show()                
                self.setTimerActive(True)
                self.timerVisible = True


                if ( self.isStarted == False ):
                    self['timer_entry'].set_text( "0 : 0 : 0" )
                else:
                    if ( self.SimulationButton.getCurrentState() == 'stop' ):
                        self['timer_entry'].set_text( self.getCurrentTime( self.tempTime ) )
                    else:
                        self['timer_entry'].set_text( self.getCurrentTime( time.time() - self.startTime + self.tempTime ) )


        def setIndicatorActive( self, isActive ): 

            self['run_speed_indicator'].set_active(isActive)
            ( self['indicator_button'].get_child() ).set_active(isActive)


        def setTimerActive( self, isActive ): 

            self['timer_menu'].set_active(isActive)
            ( self['timer_button'].get_child() ).set_active(isActive)


	def __displayWindow( self, *arg ):
		"""This method is called, when the menu or buttons on MainWindow is pressed.
		arg[0]   ---  menu or button
		Returns None
		"""
		if self.__button_update:
			return



		# checks the length of argument, but this is verbose
		if len( arg ) < 1:
			return None

		# --------------------------------------
		# LoggerWindow
		# --------------------------------------
		# When LoggerWindow is selected
		if arg[0] == self['logger_button'] or \
		   arg[0] == self['logger_window_menu']:
			self.theSession.toggleWindow('LoggerWindow')

		# When StepperWindow is selected
		elif arg[0] == self['stepper_button'] or \
		     arg[0] == self['stepper_window_menu']:
			self.theSession.toggleWindow('StepperWindow')

		# When InterfaceWindow is selected
		elif arg[0] == self['interface_button'] or \
                     arg[0] == self['interface_window_menu']:
			self.theSession.toggleWindow('InterfaceWindow')

		# When BoardWindow is selected
		elif arg[0] == self['board_button'] or \
		     arg[0] == self['board_window_menu']:
			self.theSession.toggleWindow('BoardWindow')

		self.update()



	def hideMessageWindow( self ):            
            
                self[ 'messagehandlebox' ].hide()
		( self['message_togglebutton'].get_child() ).set_active(gtk.FALSE)


	def showMessageWindow( self ):
		self[ 'messagehandlebox' ].show()
		( self['message_togglebutton'].get_child() ).set_active(gtk.TRUE)


        def __toggleMessageWindow( self, *arg ) :
		"""expands or folds MessageWindow
		arg[0]   ---  self['message_togglebutton'] or self['message_window_menu']
		Returns None
		"""
		if self.__button_update:
			return

		# checks the length of argument, but this is verbose
		if len(arg) < 1 :
                    return None

                if ( arg[0].get_name() != "message_window_menu" ):
                    anObject = arg[0].get_child()
                else:
                    anObject = arg[0]

                # show
                if anObject.get_active() == TRUE:
                    self.theMessageWindowVisible = True
                    self.showMessageWindow() 
                    self.__resizeVertically( self.theMessageWindow.getActualSize()[1] )
                    # hide
                else:
                    self.theMessageWindowVisible = False
                    self.hideMessageWindow()
                    
                    if self.theEntityListWindowVisible:
                        self.__resizeVertically( self['entitylistarea'].get_allocation()[3] )
                    else:
                        self.__resizeVertically( 0 )


		self.updateButtons()


        def __toggleEntityListWindow( self, *arg ):

            if arg[0].get_active() == TRUE:
                self.theEntityListWindowVisible = True
                self['entitylistarea'].show()
                self.__resizeVertically( self['entitylistarea'].get_allocation()[3] )
                    
            else:
                self.theEntityListWindowVisible = False
                self['entitylistarea'].hide()

                if self.theMessageWindowVisible:
                    self.__resizeVertically( self.theMessageWindow.getActualSize()[1] )
                else:
                    self.__resizeVertically( 0 )


	def __displayAbout ( self, *args ):
		# show about information
		self.createAboutSessionMonitor()
		

        def __clearTimer( self, *arg ):

            self['timer_entry'].set_text( "0 : 0 : 0" )
            self.tempTime = 0.0

            if self.SimulationButton.getCurrentState() == 'run':
                self.startTime = time.time()
            else:
                self.isStarted = False
                self.startTime = 0.0

        def __setAnimationSensitive( self, *arg ):

            if self.logoMovable:
                self.logoMovable = False
                if self.SimulationButton.getCurrentState() == 'run':
                    self.logoAnimation.stop()

            else:
                self.logoMovable = True
                if self.SimulationButton.getCurrentState() == 'run':
                    self.logoAnimation.start()
                
	def createAboutSessionMonitor(self):
		if not self.openAboutSessionMonitor:
			AboutSessionMonitor(self)

	def toggleAboutSessionMonitor(self,isOpen,anAboutSessionMonitor):
		self.theAboutSessionMonitor = anAboutSessionMonitor
		self.openAboutSessionMonitor=isOpen



	def openPreferences( self, *arg ):
		"""display the preference window
		arg[0]   ---  self['preferences_menu']
		Return None
		"""

		# Preference Window is not implemented yet.
		# So, displays Warning Dialog
		aMessage = ' Sorry ! Not implemented... [%s]\n' %'03/April/2003'
		aDialog = ConfirmWindow(OK_MODE,aMessage,'Sorry!')


	def __printMessage( self, aMessage ):
		"""prints message on MessageWindow
		aMessage   ---  a message (str or list)
		Return None
		"""

		# prints message on MessageWindow
		self.theMessageWindow.printMessage( aMessage )

	def __createEntityListWindow( self, *arg ):
		anEntityListWindow = self.theSession.createEntityListWindow( "EntityListWindow", self['statusbar'] )

	def deleted( self, *arg ):
		""" When 'delete_event' signal is chatcked( for example, [X] button is clicked ),
		delete this window.
		Returns TRUE
		"""
		return self.__deleted( *arg )






