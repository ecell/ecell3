#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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
#
# Designed by Koichi Takahashi <shafi@e-cell.org>
# Programmed by Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

import sys
import os
import time
import popen2
import threading
import signal
import urlparse

import ecell.eml
import ecell.emc
import ecell.ecs

__all__ = (
    'createScriptContext',
    'getCurrentShell',
    'lookupExecutableInPath',
    'checkCommandExistence',
    'pollForOutputs',
    'raiseExceptionOnError',
    'urlmerge',
    'getCurrentUserName'
    )

def createScriptContext( anInstance, parameters ):
    '''create script context
    '''

    # theSession == self in the script
    aContext = { 'theSession': anInstance,'self': anInstance }

    # flatten class methods and object properties so that
    # 'self.' isn't needed for each method calls in the script
    aKeyList = list ( anInstance.__dict__.keys() +\
                          anInstance.__class__.__dict__.keys() )
    aDict = {}
    for aKey in aKeyList:
        aDict[ aKey ] = getattr( anInstance, aKey )

        aContext.update( aDict )
        aContext.update( parameters )

        return aContext

def getCurrentShell():
    '''return the current shell
    Note: os.env('SHELL') returns not current shell but login shell.
    '''

    aShellName = os.popen('ps -p %s'%os.getppid()).readlines()[1].split()[3]
    aCurrentShell = os.popen('which %s' %aShellName).read()[:-1]

    return aCurrentShell

def lookupExecutableInPath( binName, pathList = None ):
    if pathList == None:
        pathList = os.environ.get('PATH', '').split( os.pathsep )
    for path in pathList:
        file = os.path.join( path, binName )
        try:
            if not os.access( file, os.X_OK | os.R_OK ):
                continue
        except:
            continue
        return file
    return None

def checkCommandExistence(command):
    '''constructor
    command(str) -- a command name to be checked.
    Return True(exists)/False(does not exist)
    '''
    return lookupExecutableInPath( command ) != None

def pollForOutputs( proc, timeout = 15.0 ):
    if sys.platform.startswith( 'win' ):
        def msg_fetch( msgs, key ):
            try:
                msgs[ key ] += getattr( proc, key ).read()
            except:
                pass
    else:
        from select import select
        import fcntl
        def msg_fetch( msgs, key ):
            try:
                fd = getattr( proc, key ).fileno()
                fcntl.fcntl(fd, fcntl.F_SETFL,
                        fcntl.fcntl( fd, fcntl.F_GETFL ) | os.O_NONBLOCK )
                while True:
                    ifd, ofd, efd = select( [ fd ], [], [], timeout )
                    if len( ifd ) == 0:
                        break
                    buf = os.read( fd, 1024 )
                    if len( buf ) == 0:
                        break
                    msgs[ key ] += buf
            except:
                pass

    proc.tochild.close()
    msgs = { 'fromchild': '', 'childerr': '' }

    threads = []

    timer = threading.Timer( timeout, lambda: os.kill( proc.pid, signal.SIGKILL ) )
    timer.start()
    try:
        for key in msgs.iterkeys():
            thr = threading.Thread( target = msg_fetch, args = ( msgs, key, ) )
            threads.append( thr )
            thr.start()
    finally:
        for thr in threads:
            thr.join()
    retval = ( os.WEXITSTATUS( proc.wait() ), msgs[ 'fromchild' ], msgs[ 'childerr' ] )
    timer.cancel()
    return retval

def raiseExceptionOnError( exc, tup ):
    if tup[ 0 ] != 0:
        if len( tup[ 2 ] ):
            errmsg = tup[ 2 ]
        else:
            errmsg = tup[ 1 ]
        raise exc, "%s (return code: %d)" % ( errmsg, tup[ 0 ] )
    return tup[ 1 ]

def urlmerge( base, url ):
    if isinstance( base, str ) or isinstance( base, unicode ):
        base = urlparse.urlsplit( base )
    if isinstance( url, str ) or isinstance( url, unicode ):
        url = urlparse.urlsplit( url )
    builtUrl = [ '', '', '', '', '' ]
    for k  in xrange( 0, 5 ):
        v = url[ k ]
        if v != '':
            builtUrl[ k ] = v
        else:
            builtUrl[ k ] = base[ k ]
    return builtUrl

def getCurrentUserName():
    retval = None
    try:
        retval = getattr( os, 'getlogin' )()
    except:
        try:
            import win32api
            retval = win32api.GetUserName()
        except:
            pass
    return retval
