#!/usr/bin/env python

from Window import *
from main import *
from Plugin import *

from gtk import *
from ecssupport import *

import ecs

import MessageWindow
import PaletteWindow
import EntryListWindow
import LoggerWindow

import string

def false():
    return false

class MainWindow(Window):

    def __init__( self ):

        self.theSimulator = ecs.Simulator()

        Window.__init__( self )

        self.theHandlerMap = \
            { 'load_rule_menu_activate'              : self.openRuleFileSelection ,
              'load_script_menu_activate'            : self.openScriptFileSelection ,
              'save_cell_state_menu_activate'        : self.saveCellStateToTheFile ,
              'save_cell_state_as_menu_activate'     : self.openSaveCellStateFileSelection ,
              'exit_menu_activate'                   : self.exit ,
              'message_window_menu_activate'         : self.toggleMessageWindow ,
              'interface_window_menu_activate'       : self.toggleInterfaceListWindow ,
              'palette_window_menu_activate'         : self.togglePaletteWindow ,
              'create_new_entry_list_menu_activate'  : self.createNewEntryList ,
              'create_new_logger_list_menu_activate' : self.createNewLoggerList ,
              'preferences_menu_activate'            : self.openPreferences ,
              'about_menu_activate'                  : self.openAbout ,
              'start_button_clicked'     : self.startSimulation ,
              'stop_button_clicked'      : self.stopSimulation ,
              'step_button_clicked'      : self.stepSimulation ,
              'entry_button_clicked'     : self.createNewEntryList ,
              'logger_button_clicked'    : self.createNewLoggerList ,
              'palette_togglebutton_toggled'   : self.togglePaletteWindow ,
              'message_togglebutton_toggled'   : self.toggleMessageWindow ,
              'Interface_togglebutton_toggled' : self.toggleInterfaceListWindow ,
              }
        self.addHandlers( self.theHandlerMap )

        self.thePluginManager = PluginManager( self )

        self.theCurrentTime = 0

        #### create Script File Selection ####
        self.theScriptFileSelection = gtk.GtkFileSelection( 'Select Script File' )
        self.theScriptFileSelection.ok_button.connect('clicked', self.loadScript)
        self.theScriptFileSelection.cancel_button.connect('clicked', self.closeParentWindow)

        #### create Rule File Selection ####
        self.theRuleFileSelection = gtk.GtkFileSelection( 'Select Rule File' )
        self.theRuleFileSelection.ok_button.connect('clicked', self.loadRule)
        self.theRuleFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
        
        #### create Save Cell State File Selection ####
        self.theSaveFileSelection = gtk.GtkFileSelection( 'Select Rule File for Saving' )
        self.theSaveFileSelection.ok_button.connect('clicked', self.saveCellState)
        self.theSaveFileSelection.cancel_button.connect('clicked', self.closeParentWindow)
        
        #### create Palette Window ####
        self.thePaletteWindow = PaletteWindow.PaletteWindow()

        #### create Message Window ####
        self.theMessageWindow = MessageWindow.MessageWindow()
        self.theMessageWindowWindow = self.theMessageWindow[ 'message_window' ]
        self.theMessageWindowWindow.hide()

        ### initialize for run method ###
        self.theSimulator.setPendingEventChecker( gtk.events_pending )
        #self.theSimulator.setPendingEventChecker( false )
        self.theSimulator.setEventHandler( gtk.mainiteration  )
        

    ###### window operation ####
    def closeParentWindow( self, button_obj):
        aParentWindow = button_obj.get_parent_window()
        aParentWindow.hide()

    ###### Load Rule ######
    def openRuleFileSelection( self, obj ) :
        self.theRuleFileSelection.show_all()

    def loadRule( self, button_obj ) :
        aFileName = self.theRuleFileSelection.get_filename()
        self.theRuleFileSelection.hide()
        aGlobalNameMap = { 'aMainWindow' : self }
        execfile( aFileName, aGlobalNameMap )

    ###### Load Script ######
    def openScriptFileSelection( self, obj ) :
        self.theScriptFileSelection.show_all()
        
    def loadScript( self, button_obj ):
        aFileName = self.theScriptFileSelection.get_filename()
        self.theScriptFileSelection.hide()
        aGlobalNameMap = { 'aMainWindow' : self }
        execfile(aFileName, aGlobalNameMap)
        
    ###### Save Cell State As ######
    def openSaveCellStateFileSelection( self, obj ) :
        self.theSaveFileSelection.show_all()

    def saveCellState ( self, button_obj ) : pass

    ###### Exit ######
    def exit( self, obj ):
        mainQuit()

    def startSimulation( self, a ) :
        self.printMessage( "start\n" )
        self.theSimulator.run()

    def stopSimulation( self, a ) :
        self.printMessage( 'this function STOP is not supported.\n' )
        self.theSimulator.stop();

    def stepSimulation( self, a ) : 
        self.theCurrentTime += 1
        self.printMessage( "step " )
        self.printMessage( str( self.theCurrentTime ) )
        self.printMessage( "\n" )
        
    def createNewEntryList( self, button_obj ) :
        aEntryList = EntryListWindow.EntryListWindow( self )
    
    def createNewLoggerList( self, a ) :
        aLoggerList = LoggerWindow.LoggerWindow( self )

    ###### Message Window ######
    def toggleMessageWindow( self, button_obj ) :
        if button_obj.get_active() :
            self.theMessageWindowWindow.show_all()
        else :
            self.theMessageWindowWindow.hide()

    def printMessage( self, aMessageString ):
        self.theMessageWindow.printMessage( aMessageString )

    def printProperty( self, fullpn ):
        value = self.theSimulator.getProperty( fullpn )
        self.printMessage( getFullPNString( fullpn ) )
        self.printMessage( ' = ' )
        if len(value) == 1:
            self.printMessage( str(value[0]) )
        else:
            for i in value:
                self.printMessage( str(i) )
                self.printMessage( ',' )
        self.printMessage( "\n" )
    
    def printAllProperties( self, fullid ):
        properties = self.theSimulator.getProperty( fullid +  ('PropertyList',) )
        for property in properties:
            self.printProperty( fullid + ( property, ) )

    def printList( self, primitivetype, systempath,list ):
        for i in list:
            printAllProperties( ( primitivetype, systempath, i ) )

##########################################################################


    ###### Toggle Palette Window ######
    def togglePaletteWindow( self, button_obj ) :
        if button_obj.get_active() :
            self.thePaletteWindow.show_all()
        else :
            self.thePaletteWindow.hide()
        
    def toggleInterfaceListWindow( self, a ) : pass

    #### these method is not supported in summer GUI project
    def loadCellState( self ) : pass
    def saveCellStateToTheFile( self ) : pass
    def openPreferences( self ) : pass
    def openAbout( self ) : pass
    #### these method is not supported in summer GUI project

if __name__ == "__main__":

    def mainLoop():
        aMainWindow = MainWindow()
        gtk.mainloop()

    def main():
        mainLoop()

    main()




