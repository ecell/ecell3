#!/usr/bin/env python

from Window import *
from gtk import *

class MessageWindow(Window):

    def __init__( self ):
        Window.__init__( self )
        self.printMessage('')

    def printMessage( self, aMessageString ):
        self["message_text_box"].insert_defaults( aMessageString )



if __name__ == "__main__":

    def mainLoop():
        gtk.mainloop()

    def main():
        aWindow = MessageWindow( 'EntryListWindow.glade' )
        mainLoop()


    main()




