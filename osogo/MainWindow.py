#!/usr/bin/env python

from Window import *
from main import *
from Plugin import *
import PaletteWindow

class MainWindow(Window):

    def __init__( self ):

        # self.thePluginWindowManager = PluginWindowManager()

        Window.__init__( self )

        self.theHandlerMap = \
            { 'load_rule_menu_activate'              : self.loadRule ,
              'load_script_menu_activate'            : self.loadScript ,
              'load_cell_state_menu_activate'        : self.loadCellState ,
              'save_cell_state__menu_activate'       : self.saveCellState ,
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
              }
        self.addHandlers( self.theHandlerMap )

        #### create Script File Selection ####
        self.theScriptFileSelection = gtk.GtkFileSelection( 'Select Script File' )
        self.theScriptFileSelection.ok_button.connect('clicked', self.executeScript)
        self.theScriptFileSelection.cancel_button.connect('clicked', self.cancelLoadingScript)

        self.thePaletteWindow = PaletteWindow.PaletteWindow()

    ###### Load Rule ######
    def loadRule( self ) : pass

    ###### Load Script ######
    def loadScript( self, obj ) :
        self.theScriptFileSelection.show_all()

    def executeScript( self, button_obj ):
        aFileName = self.theScriptFileSelection.get_filename()
        self.theScriptFileSelection.hide()
        print aFileName
        execfile(aFileName)
        
    def cancelLoadingScript( self, button_obj):
        self.theScriptFileSelection.hide()

    ###### Exit ######
    def exit( self, obj ):
        mainQuit()

    def startSimulation( self, a ) : pass
    def stopSimulation( self, a ) : pass
    def stepSimulation( self, a ) : pass

    def createNewEntryList( self, a ) : pass
    def createNewLoggerList( self, a ) : pass

    ###### Toggle Palette Window ######
    def togglePaletteWindow( self, button_obj ) :
        if button_obj.get_active() :
            self.thePaletteWindow.show_all()
        else :
            self.thePaletteWindow.hide()
        
    def toggleMessageWindow( self, a ) : pass
    def toggleInterfaceListWindow( self, a ) : pass

    #### these method is not supported in summer GUI project
    def loadCellState( self ) : pass
    def saveCellState( self ) : pass
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
