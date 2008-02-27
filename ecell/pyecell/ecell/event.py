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

import traceback as tb
import sys

__all__ = (
    'Event',
    'EventDispatcher',
    'Interuptions'
    )

class Event( object ):
    def __init__( self, type, source, **nargs ):
        self.type = type
        self.source = source
        self.params = nargs

    def __getattr__( self, key ):
        if not self.params.has_key( key ):
            raise AttributeError, "No such parameter: %s" % key
        return self.params[ key ]

    def __repr__( self ):
        return 'Event(%s: %s)' % ( self.type, self.params )

class Interuptions( Exception ):
    def __init__( self ):
        self.exceptions = []

    def append( self ):
        self.exceptions.append(
            {
                'type': sys.exc_type,
                'value': sys.exc_value,
                'traceback': sys.exc_traceback
                }
            )

    def __repr__( self ):
        return self.__str__()

    def __str__( self ):
        retval = ''
        cnt = 0
        for exc in self.exceptions:
            retval += "\n%d: " % ( cnt + 1, ) + \
                "\n   ".join( ''.join( tb.format_exception(
                exc['type'], exc['value'], exc['traceback'] ) ).split("\n") ) +\
                "\n"
        return retval

    def __len__( self ):
        return len( self.exceptions )

    def __iter__( self ):
        return self.exceptions.__iter__()

class Observable:
    def addObserver( self, type, observer ):
        raise NotImplementedError

    def removeObserver( self, observer ):
        raise NotImplementedError

class EventDispatcher( Observable ):
    def __init__( self ):
        object.__setattr__( self, 'eventQueue', [] )
        object.__setattr__( self, 'observers', {} )

    def addObserver( self, type, observer ):
        assert callable( observer )
        object.__getattribute__( self, 'observers' )[ observer ] = type

    def removeObserver( self, observer ):
        assert callable( observer )
        del object.__getattribute__( self, 'observers' )[ observer ]

    def dispatchEvent( self, ev ):
        eventQueue = object.__getattribute__( self, 'eventQueue' )
        eventQueue.append( ev )
        observers = object.__getattribute__( self, 'observers' )
        while len( eventQueue ) > 0:
            ev = eventQueue.pop( 0 )
            interuptions = Interuptions()
            for observer in observers.keys():
                if isinstance( ev, observers[ observer ] ):
                    try:
                        observer( ev )
                    except:
                        interuptions.append()
            if len( interuptions ) > 0:
                raise interuptions

    def dispose( self ):
        object.__setattr__( self, 'observers', None )
