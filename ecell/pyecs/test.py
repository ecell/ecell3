#!/usr/bin/python2

import ecs

ENTITY     = 1
SUBSTANCE  = 2
REACTOR    = 3
SYSTEM     = 4

def printProperty( sim, fullpropertyname ):
    value = sim.getProperty( fullpropertyname )
    print fullpropertyname, '\t=\t', value

def printAllProperties( sim, fullid ):
    properties = sim.getProperty( fullid +  ('PropertyList',) )
    for property in properties:
        printProperty( sim, fullid + ( property, ) )

def printList( sim, primitivetype, systempath,list ):
    for i in list:
        printAllProperties( sim, ( primitivetype, systempath, i ) )



print 'create Simulator instance.'
s = ecs.Simulator()

s.createEntity( 'System', ( SYSTEM, '/', 'CYTOPLASM' ), 'cytoplasm' )

print 'make substances...'
s.createEntity( 'Substance', ( SUBSTANCE, '/', 'A' ), 'substance A' )
s.createEntity( 'Substance', ( SUBSTANCE, '/', 'B' ), 'substance B' )
s.createEntity( 'Substance', ( SUBSTANCE, '/', 'C' ), 'substance C' )


print 'make reactors...'
try:
    s.createEntity('ConstantActivityReactor',
                   ( REACTOR, '/', 'RC1' ),
                   'constant reactor' )
except:
    print 'cannot instantiate ConstantActivityReactor'
    
print 'set Substance:/:A Quantity = 30'
s.setProperty( ( SUBSTANCE, '/', 'A', 'Quantity' ), (30,) )

print 'initialize()...'
s.initialize()


printAllProperties( s, ( SYSTEM, '', '/' ) )

printAllProperties( s, ( SYSTEM, '/', 'CYTOPLASM' ) )


substancelist = s.getProperty( ( SYSTEM, '/', '/', 'SubstanceList' ) )

printList( s, SUBSTANCE, '/' , substancelist )

print

printProperty( s, ( SUBSTANCE, '/', 'A', 'Quantity' ) )
print 'changing Quantity of Substance:/:A...'
s.setProperty( ( SUBSTANCE, '/', 'A', 'Quantity' ), (1, ) )
printProperty( s, ( SUBSTANCE, '/', 'A', 'Quantity' ) )

try:
    printAllProperties( s, ( REACTOR, '/', 'RC1' ) )
except:
    pass

print 'step()...'
printProperty( s, ( SYSTEM, '/', '/', 'CurrentTime' ) )
s.step()

printProperty( s, ( SYSTEM, '/', '/', 'CurrentTime' ) )
s.step()

printProperty( s, ( SYSTEM, '/', '/', 'CurrentTime' ) )
s.step()

printProperty( s, ( SYSTEM, '/', '/', 'CurrentTime' ) )
s.step()
