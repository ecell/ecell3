#!/usr/bin/env python

from Window import *

import string
import sys

class ViewWindow(Window):

    theClipBoard = ''

    def __init__(self, gladefile, sim, fpns):
        Window.__init__( self, gladefile )

        self.theSimulator = sim
        
        self.theFPNs = fpns
        self.addHandlers(
            { 'copy_fqpps':            self.copyFPNs,
              'paste_fqpps':           self.pasteFPNs,
              'add_fqpps':             self.addFPNs,
              'print_fqpps_to_stdout': self.printFPNs,
              'drag_data_received':    self.printTest,
              'drag_motion':           self.printTest,
              'drag_leave':            self.printTest,
              'drag_drop':             self.printTest,
              'drag_data_get':         self.printTest,
              'drag_data_delete':      self.printTest }
            )

    def update( self ):
        pass

    def initialize( self ):
        pass

    def printTest( self, obj, data ):
        print obj, data

    def copyFPNs(self, a, b):
        ViewWindow.theClipBoard = self.theFPNs
        print 'copy'

    def pasteFPNs(self, a, b):
        self.theFPNs = ViewWindow.theClipBoard
        initialize()
        print 'paste' + ' : ' + self.theFPNs

    # overwrite in subclass if needed
    def addFPNs(self, a, b):
        self.theFPNs = self.theFPNs + ',' + ViewWindow.theClipBoard
        print 'add' + ' : ' + self.theFPNs

    def printFPNs(self, a, b):
        print self.theFPNs
    
        



if __name__ == "__main__":
    def mainLoop():
        gtk.mainloop()

    def main():
        mainLoop()

    main()

















