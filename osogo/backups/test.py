#!/usr/bin/python2

import ecs

def printProperty( sim, fqpi, propertyname ):
    value = sim.getEntityProperty( fqpi, propertyname )
    print fqpi, '::', propertyname, '\t=\t', value

def printAllProperties( sim, fqpi ):
    properties = sim.getEntityProperty( fqpi, 'PropertyList' )
    for property in properties:
        printProperty( sim, fqpi, property )

def printList( sim, primitivetype, systempath,list ):
    for i in list:
        printAllProperties( sim, primitivetype + ':' + systempath + ':' + i )



print 'create Simulator instance.'
s = ecs.Simulator()

print 'make variables...'
s.createEntity('Variable','Variable:/:A','variable A')
s.createEntity('Variable','Variable:/:B','variable B')
s.createEntity('Variable','Variable:/:C','variable C')

print 'make processs...'
#s.createEntity('ConstantActivityProcess','Process:/:RC1','constant process')

print 'set Variable:/:A Value = 30'
s.setEntityProperty( 'Variable:/:A', 'Value', (30,) )

print 'initialize()...'
s.initialize()

variablelist = s.getEntityProperty( 'System:/:/', 'VariableList' )

printList( s, 'Variable', '/' , variablelist )

print

printProperty( s, 'Variable:/:A', 'Value' )
print 'changing Value of Variable:/:A...'
s.setEntityProperty( 'Variable:/:A', 'Value', (1,) )
printProperty( s, 'Variable:/:A', 'Value' )

print
printAllProperties( s, 'Process:/:RC1' )
print
print 'step()...'
printProperty( s, 'System:/:/', 'CurrentTime' )
s.step()

printProperty( s, 'System:/:/', 'CurrentTime' )
s.step()

printProperty( s, 'System:/:/', 'CurrentTime' )
s.step()

printProperty( s, 'System:/:/', 'CurrentTime' )
s.step()
