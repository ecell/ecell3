import os

from config import *
from ViewWindow import *


class PluginWindow(ViewWindow):

    def __init__( self, dirname, sim, data ):
        aGladeFileName = os.path.join( dirname ,
                                       self.__class__.__name__ + ".glade" )
        ViewWindow.__init__( self, aGladeFileName, sim, data )

