#!/usr/bin/env python

from Window import *

import string
import sys

class ViewWindow(Window):

    theClipBoard = ''

    def __init__(self, gladefile, fqpps):
        Window.__init__( self, gladefile )
        self.theFQPPs = fqpps
        self.addHandlers(
            { 'copy_fqpps':            self.copyFQPPs,
              'paste_fqpps':           self.pasteFQPPs,
              'add_fqpps':             self.addFQPPs,
              'print_fqpps_to_stdout': self.printFQPPs,
              'drag_data_received':    self.printTest,
              'drag_motion':           self.printTest,
              'drag_leave':            self.printTest,
              'drag_drop':             self.printTest,
              'drag_data_get':         self.printTest,
              'drag_data_delete':      self.printTest }
            )

    def printTest( self, obj, data ):
        print obj, data

    def copyFQPPs(self, a, b):
        ViewWindow.theClipBoard = self.theFQPPs
        print 'copy'

    def pasteFQPPs(self, a, b):
        self.theFQPPs = ViewWindow.theClipBoard
        self.initViewWindow(self.theFQPPs)
        print 'paste' + ' : ' + self.theFQPPs

    # overwrite in subclass if needed
    def addFQPPs(self, a, b):
        self.theFQPPs = self.theFQPPs + ',' + ViewWindow.theClipBoard
        print 'add' + ' : ' + self.theFQPPs

    def printFQPPs(self, a, b):
        print self.theFQPPs
    
        



if __name__ == "__main__":
    def mainLoop():
        gtk.mainloop()

    def main():
        mainLoop()

    main()

















