#!/usr/bin/env python

from Window import *
from main import *
from Plugin import *

from gtk import *
from ecssupport import *

from script import simulator
import ecs

import PaletteWindow
import EntryListWindow

import string

class MainWindow(Window):

    def __init__( self ):

        # self.thePluginWindowManager = PluginWindowManager()

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
              'Message_togglebutton_toggled'   : self.toggleMessageWindow ,
              'Interface_togglebutton_toggled' : self.toggleInterfaceListWindow ,

              'logo_button_clicked' : self.goTestMode
              }
        self.addHandlers( self.theHandlerMap )

        self.thePluginManager = PluginManager()

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

        #######################
        #### For Test Mode ####
        #######################

        self.theTestModeWindow = GtkWindow( title='test mode window' )

        aVBox = GtkVBox()
        self.theTestModeWindow.add( aVBox )
        
        aButton = GtkButton( label = 'open interface' )
        aButton.connect( 'clicked', self.openInterfaceTest )
        aVBox.add( aButton )

        ##### modified by zhaiteng (9/5) ####

        aButton = GtkButton( label = 'update interface' )
        aButton.connect( 'clicked', self.updateInterfaceTest )
        aVBox.add( aButton )

        #####################################

        aToolbar = GtkToolbar( ORIENTATION_VERTICAL, TOOLBAR_TEXT )
        aVBox.add( aToolbar )
        
        aFirstButton = GtkRadioButton( label = 'ATP' )
        aToolbar.append_widget( aFirstButton, '', '' )
        self.theTestModeWindow.set_data( 'ATP', aFirstButton )
        aButton = GtkRadioButton( aFirstButton, label = 'ADP' )
        aToolbar.append_widget( aButton, '', '' )
        self.theTestModeWindow.set_data( 'ADP', aButton )
        aButton = GtkRadioButton( aFirstButton, label = 'AMP' )
        aToolbar.append_widget( aButton, '', '' )
        self.theTestModeWindow.set_data( 'AMP', aButton )

    ###### window operation ####
    def closeParentWindow( self, button_obj):
        aParentWindow = button_obj.get_parent_window()
        aParentWindow.hide()

    ###### Load Rule ######
    def openRuleFileSelection( self, obj ) :
        self.theRuleFileSelection.show_all()

    def loadRule( self, button_obj ) : pass

    ###### Load Script ######
    def openScriptFileSelection( self, obj ) :
        self.theScriptFileSelection.show_all()

    def loadScript( self, button_obj ):
        aFileName = self.theScriptFileSelection.get_filename()
        self.theScriptFileSelection.hide()
        print aFileName
        execfile(aFileName)
        
    ###### Save Cell State As ######
    def openSaveCellStateFileSelection( self, obj ) :
        self.theSaveFileSelection.show_all()

    def saveCellState ( self, button_obj ) : pass

    ###### Exit ######
    def exit( self, obj ):
        mainQuit()

    def startSimulation( self, a ) : pass
    def stopSimulation( self, a ) : pass
    def stepSimulation( self, a ) : pass

    def createNewEntryList( self, button_obj ) :
        aEntryList = EntryListWindow.EntryListWindow( self.theSimulator )
    
    def createNewLoggerList( self, a ) : pass

    ###### Toggle Palette Window ######
    def togglePaletteWindow( self, button_obj ) :
        if button_obj.get_active() :
            self.thePaletteWindow.show_all()
        else :
            self.thePaletteWindow.hide()
        
    def toggleInterfaceListWindow( self, a ) : pass

    #### these method is not supported in summer GUI project
    def toggleMessageWindow( self, a ) : pass
    def loadCellState( self ) : pass
    def saveCellStateToTheFile( self ) : pass
    def openPreferences( self ) : pass
    def openAbout( self ) : pass
    #### these method is not supported in summer GUI project

    #######################
    #### For Test Mode ####
    #######################
    def goTestMode( self, button_obj ) :

        print 'go test mode ...'
#          self.theSimulator = ecs.Simulator()

#          self.theSimulator.createEntity('Substance','Substance:/:A','substance A')
#          self.theSimulator.createEntity('Substance','Substance:/:B','substance B')
#          self.theSimulator.createEntity('Substance','Substance:/:C','substance C')

#          self.theSimulator.setProperty( 'Substance:/:A', 'Quantity', (15,) )
#          self.theSimulator.setProperty( 'Substance:/:B', 'Quantity', (30,) )
#          self.theSimulator.setProperty( 'Substance:/:C', 'Quantity', (40,) )

#          print 'initialize()...'
#          self.theSimulator.initialize()

        self.theSimulator = simulator()

        print 
        print 'now the simulator has created'
        print 

        self.theTestModeWindow.show_all()

    def openInterfaceTest( self, button_obj ) :
        
        aPluginList = self.thePaletteWindow.get_data( 'plugin_list' )

        for plugin_name in aPluginList :
            aButton = self.thePaletteWindow.get_data( plugin_name )
            if aButton.get_active() :
                aPluginName = plugin_name

        aSubstanceIDList = [ 'ATP', 'ADP', 'AMP' ]

        for id in aSubstanceIDList :
            aButton = self.theTestModeWindow.get_data( id )
            if aButton.get_active() :
                aSelectedID = id

        aFullPN = (SUBSTANCE, '/CELL/CYTOPLASM', aSelectedID, 'Quantity' )
        aFullPNList = ( aFullPN, )

        self.thePluginManager.createInstance( aPluginName, self.theSimulator, aFullPNList)
          
    def updateInterfaceTest( self, button_obj ):

        self.thePluginManager.update()
    
if __name__ == "__main__":

    def mainLoop():
        aMainWindow = MainWindow()
        gtk.mainloop()

    def main():
        mainLoop()

    main()




