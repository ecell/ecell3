#!/usr/bin/python

import ecs

s = ecs.makeSimulator()
s.step()
s.makePrimitive('Cell','/:CELL','The cell')
s.getMessage()
s.sendMessage()