#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import os
import sys
import traceback
import math
import time
import datetime

import gtk
import gobject

from ecell.ecssupport import *
from ecell.ecs_constants import *

import ecell.ui.osogo.config as config
from ecell.ui.osogo.OsogoWindow import *
from ecell.ui.osogo.AboutSessionMonitor import *
from ecell.ui.osogo.main import *
from ecell.ui.osogo.GtkSessionMonitor import *
from ecell.ui.osogo.ConfirmWindow import *
from ecell.ui.osogo.FileSelection import FileSelection
import ecell.ui.osogo.MessageWindow as MessageWindow

class SimulationButton:
    def __init__( self, container ):
        self.startImage = os.path.join(
            config.GLADEFILE_PATH, "icon_start.png" )
        self.stopImage = os.path.join(
            config.GLADEFILE_PATH, "icon_stop.png" )
        for widget in container.get_children():
            if isinstance( widget, gtk.Image ):
                self.image = widget
            elif isinstance( widget, gtk.Label ):
                self.label = widget

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
            self.label.set_text( 'Stop' )
        elif ( self.__currentState == 'stop' ):
            self.image.set_from_file( self.startImage )
            self.label.set_text( 'Start' )

class LogoAnimation:
    iconList = (
         "ecell32-1.png",
         "ecell32-2.png",
         "ecell32-3.png",
         "ecell32-4.png",
         "ecell32-5.png",
         "ecell32-6.png",
         "ecell32-7.png",
         "ecell32-8.png",
         "ecell32.png"
        )

    def __init__( self ):
        self.image = gtk.Image()
        self.image.set_from_file(
            os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )
        self.image.show()

        self.iconList = map(
            lambda x: os.path.join( config.GLADEFILE_PATH, x ),
            self.__class__.iconList )

        self.START_ROTATION = 0
        self.END_ROTATION = len( self.iconList) - 1

        self.__currentImage = 0
        self.__running = False

        self.extraCount = 0
        self.delay = 100
        self.theTimer = None

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

            self.theTimer = gobject.timeout_add( self.delay, LogoAnimation.animate, self )
        else:
            if ( self.__currentImage != self.END_ROTATION ):
                #self.widget.add( self.image[self.__currentImage] )
                self.image.set_from_file( self.iconList[self.__currentImage] )
                self.__currentImage += 1
                self.theTimer = gobject.timeout_add( 60, LogoAnimation.animate, self )
            else:
                if self.theTimer != None:
                    gobject.source_remove( self.theTimer )
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
                
        self.SimulationButton = SimulationButton( self['SimulationButton'] )

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
        aHandlerMap =  { 
            # menu
            'load_model_menu_activate'        : self.__openFileDlgForLoadModel,
            'load_script_menu_activate'       : self.__openFileDlgForLoadScript,
            'save_model_menu_activate'        : self.__openFileDlgForSaveModel,
            'exit_menu_activate'              : self.__deleted,
            'message_window_menu_activate'    : self.__toggleMessageWindow,
            'entitylist_window_menu_activate' : self.__toggleEntityListWindow,
            'interface_window_menu_activate'  : self.__displayInterfaceWindow,
            'entity_list_menu_activate'       : self.__createEntityListWindow ,
            'logger_window_menu_activate'     : self.__displayLoggerWindow,
            'stepper_window_menu_activate'    : self.__displayStepperWindow,
            'board_window_menu_activate'      : self.__displayBoardWindow,
            'about_menu_activate'             : self.__displayAbout,
            #sbml
            'on_import_sbml_activate'         : self.__openFileDlgForImportSBML,
            'on_export_sbml_activate'         : self.__openFileDlgForExportSBML,
            # toolbars
            'simulation_button_clicked'       : self.__handleSimulation,
            'step_button_clicked'             : self.__stepSimulation,
            
            'on_sec_step_entry_activate'      : self.__setStepSizeOrSec,
            'on_timer_clear_button_clicked'   : self.__clearTimer,
            'on_load_model_button_clicked'    : self.__openFileDlgForLoadModel,
            'on_load_script_button_clicked'   : self.__openFileDlgForLoadScript,
            'on_save_model_button_clicked'    : self.__openFileDlgForSaveModel,
            'on_entitylist_button_clicked'    : self.__createEntityListWindow,
            'on_logger_button_toggled'        : self.__displayLoggerWindow,
            'on_message_togglebutton_toggled' : self.__toggleMessageWindow,
            'on_stepper_button_toggled'       : self.__displayStepperWindow,
            'on_interface_button_toggled'     : self.__displayInterfaceWindow,
            'on_board_button_toggled'         : self.__displayBoardWindow,
            'logo_button_clicked'             : self.__displayAbout,
            'on_timer_button_toggled'         : self.__displayTimer,
            'on_indicator_button_toggled'     : self.__displayIndicator,
            'on_scrolledwindow1_expose_event' : self.__expose,

            # view
            'on_toolbar_menu_activate'        : self.__displayToolbar,
            'on_statusbar_menu_activate'      : self.__displayStatusbar,
            'on_run_speed_indicator_activate' : self.__displayIndicator,  
            'on_timer_activate'               : self.__displayTimer,
            'on_logo_animation_menu_activate' : self.__setAnimationSensitive,
            'on_logging_policy1_activate'     : self.__openLogPolicy
            }

        self.__togglableWindows = {
            'BoardWindow': (
                self['board_window_menu'],
                self['board_button'].get_child() ),
            'LoggerWindow': (
                self['logger_window_menu'],
                self['logger_button'].get_child() ),
            'InterfaceWindow': (
                self['interface_window_menu'],
                self['interface_button'].get_child() ),
            'StepperWindow': (
                self['stepper_window_menu'],
                self['stepper_button'].get_child() ),
            'MessageWindow': (
                self['message_window_menu'],
                self['message_togglebutton'].get_child() ),
            }
                
        self.addHandlers( aHandlerMap )
        self.setIconList(
            os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
            os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )
                

        # display MainWindow
        self[self.__class__.__name__].show_all()
        self.present()

        self.theStepSizeOrSec = 1

        # initializes FileSelection reference
        self.theFileSelection = None

        # initializes AboutDialog reference
        self.theAboutSessionMonitor = None
        self.openAboutSessionMonitor = False 

        # -------------------------------------
        # creates EntityListWindow 
        # -------------------------------------

        self.theEntityListWindow = self.theSession.createEntityListWindow( 'top_frame', self['statusbar'] )
        self['entitylistarea'].add( self.theEntityListWindow['top_frame'] )


        # --------------------
        # set Timer entry
        # --------------------
        self['timer_entry'].set_text( str( 0 ) )
        self['timer_entry'].set_property( 'xalign', 1 )
        self['timer_box'].hide()

        # ---------------------
        # initialize Indicator
        # ---------------------
        
        self['indicator_box'].hide()

        self.update()

    def __expose( self, *arg ):
        """expose
        Return None
        """
        pass

    def __setMenuAndButtonsStatus( self, aDataLoadedStatus ):
        """sets initial widgets status
        aDataLoadedStatus  -- the status of loading data
                 (True: Model or Script is loaded / False: Not loaded)
        Returns None
        """

        # toolbar
        self['simulation_button'].set_sensitive(aDataLoadedStatus)
        self['step_button'].set_sensitive(aDataLoadedStatus)
        self['timer_clear_button'].set_sensitive(aDataLoadedStatus)
        # self['load_model_button'].set_sensitive(not aDataLoadedStatus)
        # self['load_script_button'].set_sensitive(not aDataLoadedStatus)
        self['save_model_button'].set_sensitive(aDataLoadedStatus)
        self['entitylist_button'].set_sensitive(aDataLoadedStatus)
        self['logger_button'].set_sensitive(aDataLoadedStatus)
        self['stepper_button'].set_sensitive(aDataLoadedStatus)
        self['interface_button'].set_sensitive(aDataLoadedStatus)
        self['board_button'].set_sensitive(aDataLoadedStatus)
        self['indicator_button'].set_sensitive(aDataLoadedStatus)
        self['timer_button'].set_sensitive(aDataLoadedStatus)

        # file menu
        # self['load_model_menu'].set_sensitive(not aDataLoadedStatus)
        # self['load_script_menu'].set_sensitive(not aDataLoadedStatus)
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

    def __openFileDlgForLoadModel( self, *arg ) :
        self.openFileDlg('Load','Model')

    def __openFileDlgForLoadScript( self, *arg ) :
        # when 'Load Script' is selected
        self.openFileDlg('Load','Script')

    def __openFileDlgForSaveModel( self, *arg ):
        # when 'Save Model' is selected
        self.openFileDlg('Save','Model')

    def __openFileDlgForImportSBML( self, *arg ):
        # when 'Import SBML' is selected
        self.openFileDlg('Load','SBML')

    def __openFileDlgForExportSBML( self, *arg ):
        # when 'Export SBML' is selected
        self.openFileDlg('Save','SBML')

    def openFileDlg( self, aType, aTarget ) :
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
            self.theFileSelection = FileSelection()
            self.theFileSelection.connect('delete_event', self.__deleteFileSelection )
            self.theFileSelection.cancel_button.connect('clicked', self.__deleteFileSelection) 
            aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                os.path.join(
                    config.GLADEFILE_PATH, 'ecell.png') )
            aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                os.path.join(
                    config.GLADEFILE_PATH, 'ecell32.png') )
            self.theFileSelection.set_icon_list(
                            aPixbuf16, aPixbuf32)
            # when 'Load Model' is selected
            if aType == 'Load' and aTarget == 'Model':
                self.theFileSelection.action = 'open'
                self.theFileSelection.ok_button.connect('clicked', self.__loadData, aTarget)
                self.theFileSelection.complete( '*.em; *.eml' )
                self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )

            # when 'Load Script' is selected
            elif aType == 'Load' and aTarget == 'Script':
                self.theFileSelection.action = 'open'
                self.theFileSelection.ok_button.connect('clicked', self.__loadData, aTarget)
                self.theFileSelection.complete( '*.'+ SCRIPT_FILE_EXTENSION )
                self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,SCRIPT_FILE_EXTENSION) )

            # when 'Save Model' is selected
            elif aType == 'Save' and aTarget == 'Model':
                self.theFileSelection.action = 'save'
                self.theFileSelection.ok_button.connect('clicked', self.__saveModel)
                self.theFileSelection.complete( '*.'+ MODEL_FILE_EXTENSION )
                self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )

            # when 'Import SBML' is selected
            elif aType == 'Load' and aTarget == 'SBML':
                self.theFileSelection.action = 'open'
                self.theFileSelection.ok_button.connect('clicked', self.__loadData, aTarget)
                self.theFileSelection.complete( '*.'+ MODEL_FILE_EXTENSION )
                self.theFileSelection.set_title("Select %s File (%s)" %(aTarget,MODEL_FILE_EXTENSION) )

            # when 'Save Model' is selected
            elif aType == 'Save' and aTarget == 'SBML':
                self.theFileSelection.action = 'save'
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
            elif aFileType == 'Script':
                self.theSession.loadScript( aFileName )
            elif aFileType == 'SBML':
                    self.theSession.importSBML( aFileName )
            self.theSession.updateWindows()
        except:
                        # expants message window, when it is folded.
            if self.exists():
                if not ( self['message_togglebutton'].get_child() ).get_active():
                    ( self['message_togglebutton'].get_child() ).set_active( True )

            # displays confirm window
            aMessage = 'Can\'t load [%s]\nSee MessageWindow for details.' %aFileName
            aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

            # displays message on MessageWindow
            aMessage = 'Can\'t load [%s]' %aFileName
            self.theSession.message(aMessage)
            anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type, sys.exc_value, sys.exc_traceback ) )
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
            if not ( self['message_togglebutton'].get_child() ).get_active():
                ( self['message_togglebutton'].get_child() ).set_active( True )


            # displays confirm window
            aMessage = 'Can\'t save [%s]\nSee MessageWindow for details.' %aFileName
            aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

            # displays error message of MessageWindow
            self.theSession.message('Can\'t save [%s]' %aFileName)
            anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
            self.theSession.message(anErrorMessage)

        # updates
        self.update()
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
            if not ( self['message_togglebutton'].get_child() ).get_active():
                ( self['message_togglebutton'].get_child() ).set_active( True )

            # displays confirm window
            aMessage = 'Can\'t save [%s]\nSee MessageWindow for details.' %aFileName
            aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')

            # displays error message of MessageWindow
            self.theSession.message('Can\'t export [%s]' %aFileName)
            anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
            self.theSession.message(anErrorMessage)

        # updates
        self.update()
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

        if self.theSession.theSession is not None:
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
                    return True        

        self.setStopState()
        self.close()
        self.theSession.QuitGUI()

        return True

    def close( self ):
        """ restores message method and closes window """
        self.theSession.restoreMessageMethod()

        if self.theSession.theSession is not None:
            self.theEntityListWindow.update()                    

        self.theEntityListWindow.close()
        OsogoWindow.close( self )

    def setStartState( self ):
        self.SimulationButton.setCurrentState( 'run' )

        if self.logoMovable:
            self.logoAnimation.start()

        self.isStarted = True
        self.startTime = time.time()

    def setStopState( self ):
        self.SimulationButton.setCurrentState( 'stop' )
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
            True: seconds
            False: steps
        """
        return self['sec_radiobutton'].get_active()

    def setStepType( self, aState ):
        """ sets Step Type radiobutton state 
            values for aState
            True : seconds
            False : step            
            """
        if aState == True :
            self['sec_radiobutton'].set_active( True )
        if aState == False:
            self['sec_radiobutton'].set_active( False )

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
        aNewValue = self['sec_step_entry'].get_text().strip()

        try:
            # converts string to float
            aNewValue = float(aNewValue)

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
            if self['step_radiobutton'].get_active():

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
        aTime = self.theSession.getCurrentTime()
        self.theCurrentTime = aTime
        self['time_entry'].set_text( not math.isnan( aTime ) and str( self.theCurrentTime ) or '' )
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
        self.__setMenuAndButtonsStatus( self.theSession.theSession is not None )
        self.updateButtons()

    def getCurrentTime( self, aTime ):
        theTime = self.datetime.fromtimestamp( aTime )

        return str( theTime.hour-9 )+" : "+str( theTime.minute ) + " : " + str( theTime.second )

    def updateButton( self, name, state ):
        self.__button_update = True
        for w in self.__togglableWindows[ name ]:
            if state != w.get_active():
                w.set_active( state )
        self.__button_update = False

    def toggleWindow( self, name, state ):
        self.updateButton( name, state )
        self.theSession.toggleWindow( name, state )

    def updateButtons( self ):
        """ updates Buttons and menus with 
        latest FundamentalWindow status
        """
        if not self.exists():
            return

        for n in self.__togglableWindows:
            self.updateButton( n, self.theSession.doesExist( n ) )

        self.updateButton( 'MessageWindow', self.theMessageWindowVisible )

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
        else:
            self['toolbar_handlebox'].show()
            self.theToolbarVisible = True

    def __displayStatusbar( self, *arg ):
        # show Statusbar

        if self.theStatusbarVisible:
            self['statusbar'].hide()
            self.theStatusbarVisible = False
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
            self.theLastTime = self.theSession.getCurrentTime()

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

    def __displayLoggerWindow( self, *arg ):
        if self.__button_update:
            return
        self.toggleWindow( 'LoggerWindow', arg[ 0 ].get_active() )

    def __displayStepperWindow( self, *arg ):
        if self.__button_update:
            return
        self.toggleWindow( 'StepperWindow', arg[ 0 ].get_active()  )

    def __displayInterfaceWindow( self, *arg ):
        if self.__button_update:
            return
        self.toggleWindow( 'InterfaceWindow', arg[ 0 ].get_active()  )

    def __displayBoardWindow( self, *arg ):
        if self.__button_update:
            return
        self.toggleWindow( 'BoardWindow', arg[ 0 ].get_active()  )

    def hideMessageWindow( self ):            
        self[ 'messagehandlebox' ].hide()
        ( self['message_togglebutton'].get_child() ).set_active(False)

    def showMessageWindow( self ):
        self[ 'messagehandlebox' ].show()
        ( self['message_togglebutton'].get_child() ).set_active(True)

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
        if anObject.get_active():
            self.theMessageWindowVisible = True
            self.showMessageWindow() 
        else:
            self.theMessageWindowVisible = False
            self.hideMessageWindow()

        self.updateButtons()

    def __toggleEntityListWindow( self, *arg ):
        if arg[0].get_active():
            self.theEntityListWindowVisible = True
            self['entitylistarea'].show()
        else:
            self.theEntityListWindowVisible = False
            self['entitylistarea'].hide()

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
        Returns True
        """
        pass
        #return self.__deleted( *arg )






