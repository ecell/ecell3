#!/usr/bin/env python2

from config import *

import os

import gtk
import gnome.ui
import GDK
import libglade


class Window:

    def __init__( self, gladefile=None, root=None ):

        self.theGladeFile = gladefile
        self.theRoot = root

        self.openWindow()

    def openWindow( self ):

        # load GLADEFILE_PATH/CLASSNAME.glade by default
        if self.theGladeFile == None:
            self.theGladeFile = GLADEFILE_PATH
            self.theGladeFile += '/' + self.__class__.__name__ + ".glade"
        else:
            if os.path.isabs( self.theGladeFile) :
                pass
            else:
                self.theGladeFile = GLADEFILE_PATH + '/' + self.theGladeFile

        if os.access( os.path.join( GLADEFILE_PATH, self.theGladeFile ), os.R_OK ):
            self.widgets = libglade.GladeXML( filename=self.theGladeFile, root=self.theRoot )
        else:
            raise IOError( "can't read %s." % self.theGladeFile )

    def addHandlers( self, handlers ):
        self.widgets.signal_autoconnect( handlers )
        
    def addHandler( self, name, handler, *args ):
        self.widgets.signal_connect( name, handler, args )

    def getWidget( self, key ):
        return self.widgets.get_widget( key )

    def __getitem__( self, key ):
        return self.widgets.get_widget( key )










