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

from weakref import WeakKeyDictionary

from ecell.event import Event, EventDispatcher
from ecell.util import copyValue
import ecell.identifiers as identifiers
import ecell.values as value
import ecell.model as model
import ecell.model.objects as objs

from ecell.ecs_constants import *

class ModelStoreError( RuntimeError ):
    pass

class ModelStoreEvent( Event ):
    pass

class ModelStore( object, EventDispatcher ):
    def __init__( self, aFactory ):
        """make ModelStore"""
        object.__init__( self )
        self.idToStepperMap = {}
        self.stepperToIDMap = WeakKeyDictionary()
        self.idToEntityMap = {}
        self.entityToIDMap = WeakKeyDictionary()
        self.dmFactory = aFactory
        self.modified = False
        self.variableReferenceSynched = WeakKeyDictionary()
        EventDispatcher.__init__( self )
        self.root = self.createEntity(
            DM_SYSTEM_CLASS,
            identifiers.ROOT_SYSTEM_FULLID
            )

    def handleAttributeChangeEvent( self, ev ):
        if ev.type == 'changed':
            if ev.key == 'parent':
                origFullID = self.entityToIDMap[ ev.source ]
                if ev.oldValue != None:
                    del self.idToEntityMap[ origFullID ]
                parentPath = origFullID.getSuperSystemPath()
                newFullID = parentPath.createFullID(
                    origFullID.typeCode, ev.source.localID )
                self.entityToIDMap[ ev.source ] = newFullID
                self.idToEntityMap[ newFullID ] = ev.source
            elif ev.key == 'localID':
                assert isinstance( ev.source, objs.Entity )
                origFullID = self.entityToIDMap[ ev.source ]
                del idToEntityMap[ origFullID ]
                parentPath = origFullID.getSuperSystemPath()
                newFullID = parentPath.createFullID(
                    origFullID.typeCode, ev.val )
                self.entityToIDMap[ ev.source ] = newFullID
            elif ev.key == 'stepper':
                assert isinstance( ev.value, objs.Stepper )
                if isinstance( ev.source, objs.Process ):
                    ev.source.setProperty(
                        DMINFO_STEPPER_PROCESSLIST,
                        map(
                            lambda i: self.entityToIDMap[ i ],
                            ev.value.processList ) )
                elif isinstance( ev.source, objs.System ):
                    ev.source.setProperty(
                        DMINFO_STEPPER_SYSTEMLIST,
                        map(
                            lambda i: self.entityToIDMap[ i ],
                            ev.value.processList ) )

    def handleReferenceEvent( self, ev ):
        for obj in ev.object:
            if isinstance( obj, objs.Stepper ):
                if not self.stepperToIDMap.has_key( obj ):
                    raise RuntimeError, "Cannot associate entity with an unmanaged stepper"
            else:
                if not self.entityToIDMap.has_key( obj ):
                    raise RuntimeError, "Cannot associate entity with an unmanaged entity"

        if isinstance( ev.object, objs.VariableReference ):
            proc = ev.object.getReferentByClass( objs.Process )
            assert proc != None
            varRefListRepr = []
            for varRef in proc.variableReferences.itervalues():
                referentFullID = self.getFullIDOf(
                    varRef.counterpartOf( proc ) )
                assert referentFullID != None
                varRefListRepr.append(
                    value.VariableReference(
                        varRef.name,
                        referentFullID,
                        varRef.coefficient ))
            self.variableReferenceSynched[ proc ] = True
            proc.setProperty( DMINFO_PROCESS_VARREFLIST, varRefListRepr )

    def handlePropertyChangeEvent( self, ev ):
        if ev.type == 'changed':
            if isinstance( ev.source, objs.System ) and \
               ev.key == DMINFO_SYSTEM_SIZE:
                sizeVar = ev.source.getChildByID( 'SIZE' )
                if sizeVar == None:
                    sizeVar = self.createEntity(
                        DM_VARIABLE_CLASS,
                        self.getFullIDOf( ev.source ) \
                            .toSystemPath().createVariableFullID( 'SIZE' ) )
                sizeVar.setProperty( 'Value', ev.value )
            if isinstance( ev.source, objs.Process ) and \
               ev.key == DMINFO_PROCESS_VARREFLIST:
                needSynch = not self.variableReferenceSynched.get(
                    ev.source, False )
                self.variableReferenceSynched[ ev.source ] = False
                if not needSynch:
                    return
                self.variableReferenceSynched[ ev.source ] = False
                removeList = []
                addList = []
                newVarRefs = {}
                for varRefRepr in ev.value:
                    assert isinstance( varRefRepr, value.VariableReference )
                    newVarRefs[ varRefRepr.name ] = varRefRepr
                    if not ev.source.variableReferences.has_key(
                            varRefRepr.name ):
                        addList.append( varRefRepr )
                for varRef in ev.source.variableReferences.itervalues():
                    varRefRepr = newVarRefs.get( varRef.name, None )
                    if varRefRepr != None:
                        if varRefRepr.coefficient != varRef.coefficient or \
                                varRefRepr.fullID != self.getFullIDOf(
                                    varRef.counterpartOf( ev.source ) ):
                            removeList.append( varRef )
                    else:
                        removeList.append( varRef )
                for varRef in removeList:
                    ev.source.removeReference( varRef )
                for varRefRepr in addList:
                    ev.source.addVariableReference(
                        self.getEntity( varRefRepr.fullID ),
                        varRefRepr.coefficient,
                        varRefRepr.name )

    def handleDMObjectEvent( self, ev ):
        self.modified = True
        self.dispatchEvent( ev )

    def handleLifecycleEvent( self, ev ):
        if ev.type == 'disposed':
            if isinstance( ev.source, objs.Stepper ):
                del self.idToStepperMap[ ev.source.localID ]
                self.unmanageStepper( ev.source )
            else:
                if isinstance( ev.source, objs.System ):
                    for aChildNode in ev.source.getChildren():
                        aChildNode.dispose()
                self.unmanageEntity( ev.source )

    def manageStepper( self, stepper, anID ):
        stepper.addObserver( model.DMObjectEvent, self.handleDMObjectEvent )
        stepper.addObserver( model.LifecycleEvent, self.handleLifecycleEvent )
        stepper.addObserver( model.AttributeChangeEvent, self.handleAttributeChangeEvent )
        stepper.addObserver( model.PropertyChangeEvent, self.handlePropertyChangeEvent )
        self.idToStepperMap[ anID ] = stepper
        self.stepperToIDMap[ stepper ] = anID
        self.dispatchEvent(
            ModelStoreEvent(
                'managed', self, object = stepper, id = anID ) )

    def unmanageStepper( self, stepper ):
        anID = self.stepperToIDMap[ stepper ]
        del self.stepperToIDMap[ stepper ]
        del self.idToStepperMap[ anID ]
        self.dispatchEvent(
            ModelStoreEvent(
                'unmanaged', self, object = stepper, id = anID ) )

    def manageEntity( self, entity, fullID ):
        entity.addObserver( model.DMObjectEvent, self.handleDMObjectEvent )
        entity.addObserver( model.LifecycleEvent, self.handleLifecycleEvent )
        entity.addObserver( model.AttributeChangeEvent, self.handleAttributeChangeEvent )
        entity.addObserver( model.ReferenceEvent, self.handleReferenceEvent )
        entity.addObserver( model.PropertyChangeEvent, self.handlePropertyChangeEvent )
        self.idToEntityMap[ fullID ] = entity
        self.entityToIDMap[ entity ] = fullID
        self.dispatchEvent(
            ModelStoreEvent(
                'managed', self, object = entity, fullID = fullID ) )

    def unmanageEntity( self, entity ):
        if isinstance( entity, objs.System ):
            for child in entity.getChildren():
                self.unmanageEntity( child )
        fullID = self.entityToIDMap[ entity ]
        del self.idToEntityMap[ fullID ]
        del self.entityToIDMap[ entity ]
        self.dispatchEvent(
            ModelStoreEvent(
                'unmanaged', self, object = entity, fullID = fullID ) )

    def createStepper( self, aClass, anID ):
        """create a stepper"""
        if anID in self.idToStepperMap.keys():
            raise ModelStoreError( "Stepper %s already exists!" % anID )
        stepper = self.dmFactory.create( aClass, anID )
        assert isinstance( stepper, objs.Stepper )
        self.manageStepper( stepper, anID )
        return stepper

    def deleteStepper( self, anID ):
        """delete a stepper"""
        if not self.idToStepperMap.has_key( anID ):
            raise ModelStoreError( "Stepper %s does not exist!" % anID )
        entity = self.idToStepperMap[ anID ]
        self.unmanageStepper( entity )
        entity.dispose()

    def getStepperList( self ):
        return self.idToStepperMap.values()

    def getStepperIDList( self ):
        return self.idToStepperMap.keys()

    def getStepper( self, anID ):
        return self.idToStepperMap[ anID ]

    def getEntity( self, fullID ):
        assert isinstance( fullID, identifiers.FullID )
        return self.idToEntityMap.get( fullID, None )

    def createEntity( self, aClass, fullID ):
        assert isinstance( fullID, identifiers.FullID )
        if self.idToEntityMap.has_key( fullID ):
            raise ModelStoreError( "Entity %s already exists!" % fullID )

        aSuperSystemPath = fullID.getSuperSystemPath()
        parentSystem = None
        if aSuperSystemPath != None:
            parentFullID = aSuperSystemPath.toFullID()
            parentSystem = self.getEntity( parentFullID )
            if parentSystem == None:
                raise ModelStoreError(
                    "Parent system of %s does not exist!" % parentFullID )

        entity = self.dmFactory.create( aClass, fullID.id )
        self.manageEntity( entity, fullID )
        entity.parent = parentSystem
        entity.setAnnotation( 'info', 'User info not available' )
        return entity

    def deleteEntity( self, fullID ):
        """delete an entity"""
        assert isinstance( fullID, identifiers.FullID )
        entity = self.idToEntityMap[ fullID ]
        entity.dispose()

    def setEntityProperty( self, fullPN, value ):
        self.getEntity( fullPN.fullID ).setProperty(
            fullPN.propertyName, value )

    def getEntityProperty( self, fullPN ):
        return self.getEntity( fullPN.fullID ).getProperty(
            fullPN.propertyName )

    def changeEntityClass( self, fullID, newClassName ):
        oldSlots = self.getEntity( self, fullID ).getPropertySlots()
        self.deleteEntity( self, fullID )
        slots = self.createEntity( newClassName, fullID ).getPropertySlots()
        for slotName in slots:
            if oldSlots.has_key( slotName ):
                slots[ slotName ].value = oldSlots[ slotName ].value
                slots[ slotName ].annotations = oldSlots[ slotName ].annotations

    def moveEntity( self, oldFullID, newFullID ):
        assert oldFullID.typeCode == newFullID.typeCode
        entity = self.idToEntityMap[ oldFullID ]
        if oldFullID.getSuperSystemPath() == newFullID.getSuperSystemPath():
            entity.localID = newFullID.id
        else:
            parent = self.idToEntityMap[ newFullID.getSuperSystemPath().toFullID() ]
            entity.parent = parent

    def getFullIDOf( self, entity ):
        return identifiers.FullID( self.entityToIDMap[ entity ] )
