#!/usr/bin/env python


from Window import *
from main import *
from Plugin import *

class MainWindow(Window):

    def __init__( self ):

        self.thePluginWindowManager = PluginWindowManager()

        self.theHandlerMap = \
            { 'load_rule_menu_activate'              : self.loadRule ,
              'load_cell_state_menu_activate'        : self.loadCellState ,
              'save_cell_state__menu_activate'       : self.saveCellState ,
              'execute_script_menu_activate'         : self.executeScriptMenu ,
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
              'palette_button_clicked'   : self.togglePaletteWindow ,
              'Message_button_clicked'   : self.toggleMessageWindow ,
              'Interface_button_clicked' : self.toggleInterfaceListWindow ,
              }
        Window.__init__( self )
        self.addHandlers( self.theHandlerMap )

    def loadRule( self ) : pass

    def exit( self, obj ):
        mainQuit()

    def startSimulation( self, a ) : pass
    def stopSimulation( self, a ) : pass
    def stepSimulation( self, a ) : pass

    def createNewEntryList( self, a ) : pass
    def createNewLoggerList( self, a ) : pass

    def togglePaletteWindow( self, a ) : pass
    def toggleMessageWindow( self, a ) : pass
    def toggleInterfaceListWindow( self, a ) : pass

    # these method is not supported in summer GUI project
    def loadCellState( self ) : pass
    def saveCellState( self ) : pass
    def executeScriptMenu( self ) : pass
    def openPreferences( self ) : pass
    def openAbout( self ) : pass


if __name__ == "__main__":

    def mainLoop():
        aMainWindow = MainWindow()
        gtk.mainloop()

    def main():
        mainLoop()

    main()
