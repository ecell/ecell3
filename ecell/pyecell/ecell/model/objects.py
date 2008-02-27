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

from weakref import WeakValueDictionary

from ecell.util import WeakSet
from ecell.DMInfo import DMClassInfo, DMInfo

from ecell.model import *
import ecell.ecs_constants as consts

__all__ = (
    'VariableReference',
    'FamilialReference',
    'StepperReference',
    'Stepper',
    'Entity',
    'System',
    'Process',
    'Variable',
    'Factory'
    )

ENTITY_CLASS              = 0
ENTITY_PROPERTYLIST       = 1
ENTITY_PARENT             = 2
ENTITY_CHILD_SYSTEMLIST   = 3
ENTITY_CHILD_PROCESSLIST  = 4
ENTITY_CHILD_VARIABLELIST = 5
ENTITY_INFO               = 6

class VariableReference( Relationship ):
    def __init__( self, first, second, coef, name ):
        Relationship.__init__( self, first, second )
        self.coefficient = coef
        self.name = name

class FamilialReference( Relationship ):
    pass

class StepperReference( Relationship ):
    pass

class Stepper( DMObject ):
    def __init__( self, *args ):
        DMObject.__init__( self, *args )
        object.__setattr__( self, 'entityListMap', {} )

    def addEntity( self, entity ):
        assert isinstance( entity, DMObject ) and hasattr( entity, 'stepper' )
        entity.stepper = self

    def _addEntity( self, entity ):
        self.addReference( entity, StepperReference )
        entity.addObserver(
            AttributeChangeEvent,
            self.handleAttributeChangeEvent
            )
        if not self.entityListMap.has_key( entity.__class__ ):
            self.entityListMap[ entity.__class__ ] = WeakSet()
        self.entityListMap[ entity.__class__ ].add( entity )

    def removeEntity( self, entity ):
        assert isinstance( entity, DMObject ) and hasattr( entity, 'stepper' )
        entity.stepper = None

    def getEntitiesByClass( self, entityClass ):
        return self.entityListMap.get( entityClass, [] ).__iter__()

    def getSystems( self ):
        return self.systemList.__iter__()

    def handleAttributeChangeEvent( self, ev ):
        if ev.type == 'changed' and ev.key == 'stepper' and \
           ev.value != self:
            self.entityListMap[ ev.source.__class__ ].remove( ev.source )
            self.removeReference( ev.source, type = StepperReference )
            ev.source.removeObserver( self.handleAttributeChangeEvent )

class Entity( DMObject ):
    def __init__( self, klass, localID, parent = None ):
        DMObject.__init__( self, klass, localID )
        DMObject.__setattr__( self, 'parent', parent )

    def __setattr__( self, key, val ):
        if key == 'referenceList':
            raise AttributeError, "%s is not writable" % key
        elif key == 'parent':
            if val != None:
                assert isinstance( val, System )
                val.addReference( self, FamilialReference )
            else:
                self.removeReferences( val, type = FamilialReference )
        DMObject.__setattr__( self, key, val )

    def __getitem__( self, anIndex ):
        if anIndex == ENTITY_CLASS:
            return self.klass
        elif anIndex == ENTITY_PROPERTYLIST:
            return self.propertySlots
        elif anIndex == ENTITY_PARENT:
            return self.parent
        elif anIndex == ENTITY_CHILD_SYSTEMLIST:
            return self.children[ consts.SYSTEM ]
        elif anIndex == ENTITY_CHILD_PROCESSLIST:
            return self.children[ consts.PROCESS ]
        elif anIndex == ENTITY_CHILD_VARIABLELIST:
            return self.children[ consts.VARIABLE ]
        else:
            return DMObject.__getitem__( self, anIndex )

    def __setitem__( self, anIndex, val ):
        if anIndex == ENTITY_PARENT:
            self.parent = val
        else:
            return DMObject.__setitem__( self, anIndex, val )

    def dispose( self ):
        if self.disposed:
            return
        self.parent = None
        DMObject.dispose( self )

class System( Entity ):
    def __init__( self, *args ):
        Entity.__init__( self, *args )
        object.__setattr__( self, 'children', {
            consts.DM_TYPE_SYSTEM:   [],
            consts.DM_TYPE_PROCESS:  [],
            consts.DM_TYPE_VARIABLE: []
            } )
        object.__setattr__( self, 'stepper', None )
        object.__setattr__( self, 'childrenIDMap', WeakValueDictionary() )

    def __setattr__( self, key, val ):
        if key in ( 'children', 'childrenIDMap' ):
            raise AttributeError, "%s is not writable" % key
        elif key == 'stepper':
            if val != None:
                assert isinstance( val, Stepper )
                val._addEntity( self )
        Entity.__setattr__( self, key, val )

    def addChild( self, child ):
        child.parent = self

    def removeChild( self, child ):
        assert isinstance( child, DMObject )
        assert child in self.children[ child.klass.typeCode ]
        child.parent = None

    def getChildren( self, entityType = None ):
        if entityType != None:
            return self.children[ entityType ].__iter__()
        return self.childrenIDMap.itervalues()

    def getChildByID( self, id ):
        return self.childrenIDMap.get( id, None )

    def handleAttributeChangeEvent( self, ev ):
        if ev.type == 'changing':
            if ev.key == 'localID':
                if self.childrenIDMap.has_key( ev.value ):
                    raise ValueError, "System already has the entity of the same local ID (%s)" % ev.value
        elif ev.type == 'changed':
            if ev.key == 'localID':
                assert self.childrenIDMap[ ev.oldValue ] == ev.source
                del self.childrenIDMap[ ev.oldValue ]
                self.childrenIDMap[ ev.value ] = ev.source

    def referenceAdded( self, rel ):
        Entity.referenceAdded( self, rel )
        if not isinstance( rel, FamilialReference ):
            return
        assert rel.first == self
        child = rel.second
        assert self.children.has_key( child.klass.typeCode )
        if self.childrenIDMap.has_key( child.localID ):
            raise RuntimeError, "Child of the same local ID (%s) already exists" % child.localID
        self.children[ child.klass.typeCode ].append( child )
        self.childrenIDMap[ child.localID ] = child
        child.addObserver(
            AttributeChangeEvent,
            self.handleAttributeChangeEvent )

    def referenceRemoved( self, rel ):
        Entity.referenceRemoved( self, rel )
        if not isinstance( rel, FamilialReference ):
            return
        assert rel.first == self
        child = rel.second
        self.children[ child.klass.typeCode ].remove( child )
        del self.childrenIDMap[ child.localID ]
        child.removeObserver( self.handleAttributeChangeEvent )

class Process( Entity ):
    def __init__( self, *args ):
        Entity.__init__( self, *args )
        object.__setattr__( self, 'stepper', None )
        object.__setattr__( self, 'variableReferences', WeakValueDictionary() )

    def __setattr__( self, key, val ):
        if key == 'stepper':
            if val != None:
                assert isinstance( val, Stepper )
                val.addEntity( self )
        Entity.__setattr__( self, key, val )

    def addVariableReference( self, referent, coef, name ):
        return self.addReference( referent, VariableReference,
            { 'coef': coef, 'name': name } )

    def removeVariableReference( self, name ):
        self.removeReference( self.variableReferences[ name ] )

    def referenceAdded( self, rel ):
        Entity.referenceAdded( self, rel )
        if isinstance( rel, VariableReference ):
            if self.variableReferences.has_key( rel.name ):
                raise KeyError, "Reference of the same name already exists."
            self.variableReferences[ rel.name ] = rel

    def refererenceRemoved( self, rel ):
        del self.variableReferences[ rel.name ]
        Entity.removeReference( self, rel )

class Variable( Entity ):
    def setProperty( self, key, val ):
        Entity.setProperty( self, key, val )
        if self.parent == None:
            return
        if key == 'Value':
            val = float( val )
            systemSize = self.parent.getProperty( consts.DMINFO_SYSTEM_SIZE )
            if systemSize != None and systemSize != 0.0:
                newMolarConc = val / ( consts.N_A * systemSize )
                newNumConc = val / systemSize
            else:
                newMolarConc = 0.0
                newNumConc = 0.0
            Entity.setProperty( self, 'MolarConc', newMolarConc )
            Entity.setProperty( self, 'NumberConc', newNumConc )
        elif key == 'MolarConc':
            systemSize = self.parent.getProperty( consts.DMINFO_SYSTEM_SIZE )
            if systemSize != None and systemSize != 0.0:
                newValue = int( val * N_A * systemSize )
                newNumConc = val * N_A
            else:
                newValue = 0.0
                newNumConc = 0.0
            Entity.setProperty( self, 'Value', newValue )
            Entity.setProperty( self, 'NumberConc', newNumConc )
        elif key == 'NumberConc':
            systemSize = self.parent.getProperty( consts.DMINFO_SYSTEM_SIZE )
            if systemSize != None and systemSize != 0.0:
                newValue = int( val * systemSize )
                newMolarConc = val / N_A
            else:
                newValue = 0.0
                newMolarConc = 0.0
            Entity.setProperty( self, 'Value', newValue )
            Entity.setProperty( self, 'MolarConc', newMolarConc )

class Factory:
    def __init__( self, dmInfo ):
        self.dmInfo = dmInfo

    def create( self, className, localID ):
        klass = self.dmInfo.getClassInfo( className )
        if klass.typeCode == consts.DM_TYPE_STEPPER:
            return Stepper( klass, localID )
        elif klass.typeCode == consts.DM_TYPE_SYSTEM:
            return System( klass, localID )
        elif klass.typeCode == consts.DM_TYPE_PROCESS:
            return Process( klass, localID )
        elif klass.typeCode == consts.DM_TYPE_VARIABLE:
            return Variable( klass, localID )
