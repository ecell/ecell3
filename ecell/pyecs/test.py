#!/usr/bin/python

import ecs

print 'create Simulator instance.'
s = ecs.Simulator()

print 'makePrimitive()...'
s.makePrimitive('Substance','Substance:/:A','substance A')

s.makePrimitive('ConstantParameterReactor','Reactor:/:RC1','constant')

print 'sendMessage()...'
s.sendMessage( 'Substance:/:A', 'quantity', '30' )

print 'initialize()...'
s.initialize()

print 'getMessage()...'
tuple = s.getMessage( 'Substance:/:A', 'id' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])

print 'getMessage()...'
tuple = s.getMessage( 'Substance:/:A', 'quantity' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])

print 'sendMessage()...'
s.sendMessage( 'Substance:/:A', 'quantity', '0' )


print 'getMessage()...'
tuple = s.getMessage( 'Substance:/:A', 'quantity' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])

print 'id...'
tuple = s.getMessage( 'System:/:CELL', 'id' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])

print 'substance list...'
tuple = s.getMessage( 'System:/:/', 'substanceList' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])

#print 'getMessage()...'
#tuple = s.getMessage( 'System:/:CELL', 'id' )
#print 'got key=%s body=%s' % (tuple[0], tuple[1])


print 'step()...'
s.step()
