#!/usr/bin/python

import ecs

print 'create Simulator instance.'
s = ecs.Simulator()

print 'makePrimitive()...'
s.makePrimitive('Cell','System:/:CELL','The cell')

print 'sendMessage()...'
s.sendMessage( 'System:/:CELL', 'supersystem', '/' )

print 'getMessage()...'
tuple = s.getMessage( 'System:/:CELL', 'id' )
print 'got key=%s body=%s' % (tuple[0], tuple[1])


print 'step()...'
s.step()
