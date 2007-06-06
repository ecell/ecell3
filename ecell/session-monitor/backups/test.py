#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
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
