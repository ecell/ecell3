#! /usr/bin/python

from ecssupport import *

class ModelInterpreter:

    def __init__( self, driver ):
        self.theDriver = driver

    def load( self, cellmodelobject ):

        for entity in cellmodelobject:
            if entity[0] == 0:
                type = entity[1]
                fullid = entity[2]
                name = entity[3]
                self.theDriver.createEntity( type, fullid, name )
            else:
                fullpn = entity[1]
                value = entity[2]
                self.theDriver.setProperty( fullpn, value )
                
if __name__ == "__main__":

    from cellmodel import cellmodel

    ModelInterpreter( 0, cellmodel )

    
