#!/usr/bin/python

import ecs

print 'create Simulator instance.'
s = ecs.Simulator()

print 'makePrimitive()...'
s.makePrimitive('Substance','Substance:/:A','substance A')

print 'sendMessage()...'
s.sendMessage( 'Substance:/:A', 'Quantity', '30' )

print 'initialize()...'
s.initialize()

print 'getMessage()...'
tuple = s.getMessage( 'Substance:/:A', 'id' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])

print 'getMessage()...'
tuple = s.getMessage( 'System:/:CELL', 'id' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])


print 'step()...'
s.step()
