#!/usr/bin/python2

import ecs

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

print 'make substances...'
s.createEntity( 'Substance', ( 'Substance', '/', 'A' ), 'substance A' )
s.createEntity( 'Substance', ( 'Substance', '/', 'B' ), 'substance B' )
s.createEntity( 'Substance', ( 'Substance', '/', 'C' ), 'substance C' )


print 'make reactors...'
try:
    s.createEntity('ConstantActivityReactor',
                   ( 'Reactor', '/', 'RC1' ),
                   'constant reactor' )
except:
    print 'cannot instantiate ConstantActivityReactor'
    
print 'set Substance:/:A Quantity = 30'
s.setProperty( ( 'Substance', '/', 'A', 'Quantity' ), (30,) )

print 'initialize()...'
s.initialize()

substancelist = s.getProperty( ( 'System', '/', '/', 'SubstanceList' ) )

printList( s, 'Substance', '/' , substancelist )

print

printProperty( s, ( 'Substance', '/', 'A', 'Quantity' ) )
print 'changing Quantity of Substance:/:A...'
s.setProperty( ( 'Substance', '/', 'A', 'Quantity' ), (1, ) )
printProperty( s, ( 'Substance', '/', 'A', 'Quantity' ) )

try:
    printAllProperties( s, 'Reactor:/:RC1' )
except:
    pass

print 'step()...'
printProperty( s, ( 'System', '/', '/', 'CurrentTime' ) )
s.step()

printProperty( s, ( 'System', '/', '/', 'CurrentTime' ) )
s.step()

printProperty( s, ( 'System', '/', '/', 'CurrentTime' ) )
s.step()

printProperty( s, ( 'System', '/', '/', 'CurrentTime' ) )
s.step()
