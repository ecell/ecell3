#!/usr/bin/python

import ecs

def printProperty( sim, fqpi, propertyname ):
    value = sim.getMessage( fqpi, propertyname )
    print fqpi, '::', propertyname, '\t=\t', value

def printAllProperties( sim, fqpi ):
    properties = sim.getMessage( fqpi, 'PropertyList' )
    for property in properties:
        printProperty( sim, fqpi, property )

def printList( sim, primitivetype, systempath,list ):
    for i in list:
        printAllProperties( sim, primitivetype + ':' + systempath + ':' + i )



print 'create Simulator instance.'
s = ecs.Simulator()

print 'make substances...'
s.makePrimitive('Substance','Substance:/:A','substance A')
s.makePrimitive('Substance','Substance:/:B','substance B')
s.makePrimitive('Substance','Substance:/:C','substance C')

print 'make reactors...'
s.makePrimitive('ConstantParameterReactor','Reactor:/:RC1','constant reactor')

print 'set Substance:/:A Quantity = 30'
s.sendMessage( 'Substance:/:A', 'Quantity', (30,) )

print 'initialize()...'
s.initialize()

substancelist = s.getMessage( 'System:/:/', 'SubstanceList' )

printList( s, 'Substance', '/' , substancelist )

print

printProperty( s, 'Substance:/:A', 'Quantity' )
print 'changing Quantity of Substance:/:A...'
s.sendMessage( 'Substance:/:A', 'Quantity', (1,) )
printProperty( s, 'Substance:/:A', 'Quantity' )

print
printAllProperties( s, 'Reactor:/:RC1' )
print
print 'step()...'
s.step()
