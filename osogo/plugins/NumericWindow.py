#!/usr/bin/env python

import sys
sys.path.append('..')
sys.path.append('/home/ecell/ecell3/ecell/pyecs')
from ViewWindow import *
from ecssupport import *
import ecs

class NumericWindow( ViewWindow ):

    def __init__( self, fpn, sim ):
        self.sim = sim
        self.theGladeFileName = 'NumericWindow.glade'
        ViewWindow.__init__( self, self.theGladeFileName, fpn )
        self.addHandler('input_value', self.inputValue)
        self.addHandler('increase_value', self.increaseValue)
        self.addHandler('decrease_value', self.decreaseValue)
        self.initInterface( fpn )

    def initInterface( self, fpn ):
        self.fpnobj = FullPropertyName(fpn)
        self["id_label"].set_text(self.fpnobj['ID'])
        value = self.sim.getProperty( self.fpnobj['FullID'], self.fpnobj['PropertyName'])
        self.theCurValue = value[0]
        self["value_frame"].set_text(str(self.theCurValue))

    def inputValue( self, obj, n ):

        aNumberString =  obj.get_text()
        self.theCurValue = string.atof( aNumberString )
        value = (self.theCurValue,)
        self.sim.setProperty( self.fpnobj['fullID'], self.fpnobj['PropertyName'], value )
        print self.sim.getProperty( self.fpnobj['FullID'], self.fpnobj['PropertyName'])

    def increaseValue( self, value, n ):
        self.theCurValue = self.theCurValue * 2
        self["value_frame"].set_text(str(self.theCurValue))

    def decreaseValue( self, obj, n ):
        self.theCurValue = self.theCurValue / 2
        self["value_frame"].set_text(str(self.theCurValue))


### test code

def mainLoop():

    s = ecs.Simulator()

    s.createEntity('Substance','Substance:/:ATP','ATP')
    s.setProperty( 'Substance:/:ATP', 'Quantity', (30,) )
    s.initialize()

    aFPN1 = 'Substance:/:ATP:Quantity'
    aWindow1 = NumericWindow( aFPN1 , s)
#    aFPN2 = 'Substance:/:ADP:Quantity'
#    aWindow2 = NumericWindow( aFPN1 , s)

    gtk.mainloop()

def main():
    mainLoop()

if __name__ == "__main__":
    main()





