#!/usr/bin/env python2

from config import *

import os

import gtk
import gnome.ui
import GDK
import libglade



class Window:

    def __init__( self, gladefile=None, root=None ):

        # load GLADEFILE_PATH/CLASSNAME.glade by default
        if gladefile == None:
            gladefile = GLADEFILE_PATH
            gladefile += '/' + self.__class__.__name__ + ".glade"
        else:
            if os.path.isabs(gladefile) :
                pass
            else:
                gladefile = GLADEFILE_PATH + '/' + gladefile

        if os.access( os.path.join( GLADEFILE_PATH, gladefile ), os.R_OK ):
            print gladefile
            print root
            self.widgets = libglade.GladeXML( filename=gladefile, root=root )
        else:
            raise IOError( "can't read %s." % gladefile )

    def addHandlers( self, handlers ):
        self.widgets.signal_autoconnect( handlers )
        
    def addHandler( self, name, handler, *args ):
        self.widgets.signal_connect( name, handler, args )

    def getWidget( self, key ):
        return self.widgets.get_widget( key )

    def __getitem__( self, key ):
        return self.widgets.get_widget( key )



