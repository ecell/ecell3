#!/usr/bin/env python

import string

import gtk
import gnome.ui
import GDK
import libglade

def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()



if __name__ == "__main__":

    from Window import *
    from plugin import *

    from MainWindow import *

    MainWindow.createWindow()

    mainLoop()






