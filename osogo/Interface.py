#!/usr/bin/env python

from Window import *

import string
import sys

class Interface(Window):

    theClipBoard = ''

    def __init__(self, gladefile, fqpps):
        Window.__init__(self, gladefile)
        self.theFQPPs = fqpps
        self.addHandler( 'copy_fqpps', self.copyFQPPs)
        self.addHandler( 'paste_fqpps', self.pasteFQPPs)
        self.addHandler( 'add_fqpps', self.addFQPPs)
        self.addHandler( 'print_fqpps_to_stdout', self.printFQPPs)

        self.addHandler( 'drag_data_received', self.printTest)
        self.addHandler( 'drag_motion', self.printTest)
        self.addHandler( 'drag_leave', self.printTest)
        self.addHandler( 'drag_drop', self.printTest)
        self.addHandler( 'drag_data_get', self.printTest)
        self.addHandler( 'drag_data_delete', self.printTest)

    def printTest(self, a, b):
        print 'test'

    def copyFQPPs(self, a, b):
        Interface.theClipBoard = self.theFQPPs
        print 'copy'

    def pasteFQPPs(self, a, b):
        self.theFQPPs = Interface.theClipBoard
        self.initInterface(self.theFQPPs)
        print 'paste' + ' : ' + self.theFQPPs

    # overwrite in subclass if needed
    def addFQPPs(self, a, b):
        self.theFQPPs = self.theFQPPs + ',' + Interface.theClipBoard
        print 'add' + ' : ' + self.theFQPPs

    def printFQPPs(self, a, b):
        print self.theFQPPs
    
        
# class InterfaceForFQPP    





if __name__ == "__main__":
    def mainLoop():
        gtk.mainloop()

    def main():
        mainLoop()

    main()

















