#!/usr/bin/python2

import ecs

ENTITY     = 1
VARIABLE  = 2
PROCESS    = 3
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
s.setProperty( ( SYSTEM, '/', 'CYTOPLASM', 'StepperClass' ),
               ('RungeKutta4SRMStepper',) )
s.setProperty( ( SYSTEM, '/', 'CYTOPLASM', 'StepInterval' ),
               (0.05,) )

print 'make variables...'
s.createEntity( 'Variable', ( VARIABLE, '/', 'A' ), 'variable A' )
s.createEntity( 'Variable', ( VARIABLE, '/', 'B' ), 'variable B' )
s.createEntity( 'Variable', ( VARIABLE, '/CYTOPLASM', 'CA' ), 's CA' )
s.createEntity( 'Variable', ( VARIABLE, '/CYTOPLASM', 'CB' ), 's CB' )


print 'make processs...'
try:
    s.createEntity('ConstantActivityProcess',
                   ( PROCESS, '/CYTOPLASM', 'RC1' ),
                   'constant process' )
except:
    print 'cannot instantiate ConstantActivityProcess'
    
print 'set Variable:/:A Value = 30'
s.setProperty( ( VARIABLE, '/', 'A', 'Value' ), (30,) )

l = s.getLogger( ( VARIABLE, '/', 'A', 'Value') )
#print l
print dir(l)

print 'initialize()...'
s.initialize()


printAllProperties( s, ( SYSTEM, '', '/' ) )

printAllProperties( s, ( SYSTEM, '/', 'CYTOPLASM' ) )


variablelist = s.getProperty( ( SYSTEM, '', '/', 'VariableList' ) )

printList( s, VARIABLE, '/' , variablelist )

variablelist = s.getProperty( ( SYSTEM, '/', 'CYTOPLASM', 'VariableList' ) )

printList( s, VARIABLE, '/CYTOPLASM' , variablelist )


print

printProperty( s, ( VARIABLE, '/', 'A', 'Value' ) )
print 'changing Value of Variable:/:A...'
s.setProperty( ( VARIABLE, '/', 'A', 'Value' ), (10.0, ) )
printProperty( s, ( VARIABLE, '/', 'A', 'Value' ) )

try:
    printAllProperties( s, ( PROCESS, '/CYTOPLASM', 'RC1' ) )
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

for i in xrange( 10000 ):
    s.step()

printProperty( s, ( SYSTEM, '/', '/', 'CurrentTime' ) )

printProperty( s, ( SYSTEM, '/', '/', 'StepInterval' ) )
printProperty( s, ( SYSTEM, '/', 'CYTOPLASM', 'StepInterval' ) )
    
print 'logger getData...'
d = l.getData()
print len( d )

d = l.getData( 2, 8 )
print len( d )

d = l.getData( 0, 5,0.005 )
print len( d )

