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

from ecell.event import Event, EventDispatcher
from ecell.util import WeakSet
from weakref import WeakKeyDictionary

import ecell.ecs_constants as consts

class DMObjectEvent( Event ):
    pass

class PropertyChangeEvent( DMObjectEvent ):
    pass

class AttributeChangeEvent( DMObjectEvent ):
    pass

class AnnotationChangeEvent( DMObjectEvent ):
    pass

class ReferenceEvent( DMObjectEvent ):
    pass

class LifecycleEvent( DMObjectEvent ):
    pass

class Relationship:
    def __init__( self, first, second ):
        self.first = first
        self.second = second

    def refers( self, tgt ):
        return tgt in ( self.first, self.second )

    def getReferentByClass( self, klass ):
        if isinstance( self.first, klass ):
            return self.first
        elif isinstance( self.second, klass ):
            return self.second
        return None

    def counterpartOf( self, tgt ):
        if self.first == tgt:
            return self.second
        else:
            return self.first

    def __iter__( self ):
        yield self.first
        yield self.second

class PropertySlot( object ):
    def __init__( self, desc, name, value = None ):
        self.descriptor = desc
        self.annotations = {}
        self.name = name
        self.value = value

    def setAnnotation( self, key, val ):
        self.annotations[ key ] = [ val ]

    def addAnnotation( self, key, val ):
        if self.annotations.has_key( key ):
            self.annotations[ key ].append( val )
        else:
            self.annotations[ key ] = [ val ]

    def getAnnotation( self, key ):
        if len( self.annotations[ key ] ) > 1:
            raise RuntimeError, "Annotation %s has multiple entries." % key
        return self.annotations[ key ]

    def getAnnotations( self, key ):
        return self.annotations[ key ]

class DMObject( object, EventDispatcher ):
    def __init__( self, klass, localID ):
        object.__init__( self )
        object.__setattr__( self, 'klass', klass )
        object.__setattr__( self, 'localID', localID )
        object.__setattr__( self, 'propertySlots', {} )
        object.__setattr__( self, 'annotations', {} )
        object.__setattr__( self, 'referenceList', [] )
        object.__setattr__( self, 'disposed', False )
        object.__setattr__( self, 'fullyPopulated', False )
        EventDispatcher.__init__( self )

    def __setattr__( self, key, val ):
        if key in ( 'propertySlots', 'annotations', 'disposed' ):
            raise AttributeError, "%s is an unchangable property" % key
        oldValue = None
        if hasattr( self, key ):
            oldValue = getattr( self, key )
            if oldValue == val:
                return

        self.dispatchEvent(
            AttributeChangeEvent(
                'changing', self,
                key = key, value = val, oldValue = oldValue ) )

        object.__setattr__( self, key, val )

        self.dispatchEvent(
            AttributeChangeEvent(
                'changed', self,
                key = key, value = val, oldValue = oldValue ) )

    def __getattribute__( self, key ):
        if key == 'disposed' or key == 'dispose':
            return object.__getattribute__( self, key )
        if self.disposed:
            raise RuntimeError, "Tried to retrive the attribute `%s' of an object that is already disposed." % key
        return object.__getattribute__( self, key )

    def hasProperty( self, key ):
        return self.propertySlots.has_key( key ) \
            or self.klass.propertyList.has_key( key )

    def getProperty( self, key ):
        return self.getPropertySlot( key ).value

    def getPropertySlot( self, key ):
        if self.propertySlots.has_key( key ):
            return self.propertySlots[ key ]
        else:
            if self.klass.propertyList.has_key( key ):
                desc = self.klass.propertyList[ key ]
            else:
                raise RuntimeError, "No such property: %s" % key
            retval = PropertySlot( desc, key )
            self.propertySlots[ key ] = retval
            return retval

    def getPropertySlots( self ):
        if not self.fullyPopulated:
            for key, desc in self.klass.propertyList.iteritems():
                if not self.propertySlots.has_key( key ):
                    self.propertySlots[ key ] = PropertySlot( desc, key )
        object.__setattr__( self, 'fullyPopulated', True )
        return self.propertySlots

    def setProperty( self, key, val ):
        propertySlot = self.getPropertySlot( key )
        if propertySlot.descriptor != None:
            val = propertySlot.descriptor.type.convertToPythonType( val )

        oldValue = propertySlot.value
        if val == oldValue:
            return

        self.dispatchEvent(
            PropertyChangeEvent(
            'changing', self,
            key = key, value = val, oldValue = oldValue ) )

        propertySlot.value = val

        self.dispatchEvent(
            PropertyChangeEvent(
                'changed', self,
                key = key, value = val, oldValue = oldValue ) )

    def addPropertySlot( self, key, desc = None ):
        if not self.klass.acceptsNewProperty:
            raise RuntimeError, \
                "Object doesn't accept dynamic properties" 
        if self.klass.propertyList.has_key( key ) \
           or self.propertySlots.has_key( key ):
            raise RuntimeError( "Property %s already exists" % key )
        propertySlot = PropertySlot( desc, key )
        self.propertySlots[ key ] = propertySlot
        self.dispatchEvent(
            PropertyChangeEvent(
                'added', self, key = key, slot = propertySlot ) )
        return propertySlot

    def removePropertySlot( self, key ):
        if self.klass.propertyList.has_key( key ):
            raise RuntimeError( "Cannot remove builtin properties" )
        if not self.propertySlots.has_key( key ):
            return
        propertySlot = self.propertySlots[ key ]
        del self.propertySlots[ key ]
        self.dispatchEvent(
            PropertyChangeEvent(
                'removed', self, key = key, slot = propertySlot ) )

    def __setitem__( self, key, val ):
        self.setProperty( key, val )

    def __getitem__( self, key ):
        return self.getProperty( key )

    def __delitem__( self, key ):
        self.removeProperty( key )

    def setAnnotation( self, key, val ):
        self.annotations[ key ] = [ val ]

    def addAnnotation( self, key, val ):
        if self.annotations.has_key( key ):
            self.annotations[ key ].append( val )
        else:
            self.annotations[ key ] = [ val ]

    def getAnnotation( self, key ):
        if len( self.annotations[ key ] ) > 1:
            raise RuntimeError, "Annotation %s has multiple entries." % key
        return self.annotations[ key ][ 0 ]

    def getAnnotations( self, key ):
        return self.annotations[ key ]

    def dispose( self ):
        if self.disposed:
            return
        self.dispatchEvent( LifecycleEvent( 'disposed', self ) )
        EventDispatcher.dispose( self )
        object.__setattr__( self, 'disposed', True )

    def __del__( self ):
        self.dispose()

    def generateFilterPredicate( referent = None, type = None, inverted = False ):
        if referent == None:
            if type == None:
                return lambda ref: True
            else:
                return lambda ref: isinstance( ref, type )
        else:
            if type == None:
                retval = lambda ref: ref.refers( referent ) 
            else:
                retval = lambda ref: \
                    ref.refers( referent ) and isinstance( ref, type )
        if inverted:
            retval = lambda retval: not retval
        return retval
    generateFilterPredicate = staticmethod( generateFilterPredicate )

    def getReferences( self, referent, **nargs ):
        return filter( self.generateFilterPredicate( self, referent, **nargs ),
            self.referenceList )

    def addReference( self, referent, type, options = {} ):
        assert issubclass( type, Relationship )
        rel = type( self, referent, **options )
        try:
            self.referenceList.append( rel )
            referent.referenceList.append( rel )
            self.referenceAdded( rel )
            referent.referenceAdded( rel )
        except:
            self.referenceList.remove( rel )
            referent.referenceList.remove( rel )
            raise
        self.dispatchEvent( ReferenceEvent( 'added', self, object = rel ) )
        return rel

    def removeReference( self, rel ):
        assert rel.refers( self )
        rel.first.referenceList.remove( rel )
        rel.second.referenceList.remove( rel )
        try:
            rel.first.referenceRemoved( rel )
            rel.second.referenceRemoved( rel )
        except:
            rel.first.referenceList.append( rel )
            rel.second.referenceList.append( rel )
            raise
        self.dispatchEvent( ReferenceEvent( 'removed', self, object = rel ) )

    def removeReferences( self, referent = None, **nargs ):
        filt = self.generateFilterPredicate( referent, **nargs )

        refsToLeave = []
        refsToRemove = []
        removedRefMap = {}
        leftRefMap = {}
        for ref in self.referenceList:
            counterpart = ref.counterpartOf( self )
            if not leftRefMap.has_key( counterpart ):
                leftRefMap[ counterpart ] = []
                removedRefMap[ counterpart ] = []
            if not filt( ref ):
                refsToLeave.append( ref )
                leftRefMap[ counterpart ].append( ref )
            else:
                refsToRemove.append( ref )
                removedRefMap[ counterpart ].append( ref )
        object.__setattr__( self, 'referenceList', refsToLeave )
        for counterpart in leftRefMap:
            object.__setattr__( counterpart, 'referenceList',
                leftRefMap[ counterpart ] )
            for ref in removedRefMap[ counterpart ]:
                counterpart.referenceRemoved( ref )
        for ref in refsToRemove:
            self.dispatchEvent( ReferenceEvent( 'removed', self, object = ref ) )

    def referenceAdded( self, rel ):
        pass

    def referenceRemoved( self, rel ):
        pass
