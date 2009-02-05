#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
# Programmed by Giuseppe Aprea <giuseppe.aprea@gmail.com>

# imports standard modules
import sys
import os
import time
import signal
import re
import popen2
from string import *

# imports ecell modules
from ecell.session_manager.Util import *
from ecell.session_manager.SessionManager import *
from ecell.session_manager.Constants import *

BSUB = 'bsub'
BJOBS = 'bjobs'
BHOSTS = 'bhosts'
BKILL = 'bkill'

class SessionProxy( AbstractSessionProxy ):
    '''SessionProxy class
    Target environment is local PC that has only one cpu.
    '''

    def __init__(self, dispatcher, jobID ):
        # call superclass's constructor
        AbstractSessionProxy.__init__( self, dispatcher, jobID )
        
        # initialize parameter
        self.__theLSFJobID = None
        self.__theTmpScriptFileName = \
                "script." + os.path.basename( getCurrentShell() )

    def getLSFJobID(self):
        return self.__theLSFJobID 

    def run(self):
        '''run process
        Return None
        '''
        if not AbstractSessionProxy.run( self ):
            return False

        # save current directory
        aCwd = os.getcwd()

        try:
            os.chdir( self.getJobDirectory() )

            args = [
                SessionProxy.BSUB
                ]
            for key, val in self.getEnvironmentVariables().iteritems():
                args.extend( ( '-v', key + '=' + val ) )
            args.append( '-cwd' )
            args.extend( ( '-o', self.getStdoutFileName() ) )
            args.extend( ( '-e', self.getStderrFileName() ) )
            args.extend( ifilter(
                    lambda x: x not in ( '-cwd', '-o', '-e' ),
                    self.getOptionList() ) )
            args.append( self.getScriptFileName() )
            args.extend( self.getArguments() )

            msg = raiseExceptionOnError(
                RuntimeError,
                pollForOutputs( popen2.Popen3( args, True ) )
                )
            m = re.match(
                r'Your job (\d+) \("(?:[^"]|\\")*"\) has been submitted', msg
                )
            if m == None:
                raise RuntimeError, '%s returned unexpected result: %s' % (
                        BSUB, msg )
            self.__thLSFJobID = m.group( 1 )
            self.getSystemProxy().manageJob( self )
            stdout.close()
        finally:
            os.chdir( aCwd )
        return True

    def __cancel( self ):
        if self.__theSGEJobID >= 0:
            raiseExceptionOnError( RuntimeError,
                pollForOutputs(
                    popen2.Popen3( ( BKILL, self.__theSGEJobID ), True )
                    )
                )

    def stop(self):
        '''stop the job
        Return None
        '''

        # When this job is running, stop it.
        if self.getStatus() == RUN:
            self.__cancel()

        # set error status
        self.setStatus(ERROR) 

class SystemProxy( AbstractSystemProxy ):
    '''LSFSystemProxy
    '''

    def __init__( self, sessionManager ):
        '''constructor
        sessionManager -- an instance of SessionManager
        '''
        AbstractSystemProxy.__init__( self, sessionManager )

        # Check the existance of LSF command.
        for cmd in ( BSUB, BJOBS, BHOSTS ):
            if checkCommandExistence(cmd) == False:
                raise Exception(
                    "\"%s\" is not included in the $PATH environment" % cmd )
        # Initialize attributes.
        self.__theQueueList = None
        self.__jobsByLSFJobID = {}

    def getDefaultConcurrency(self):
        args = [
            BHOSTS,
            ]
        args.extend( self.getOptionList() )

        stdout, stdin = popen2.popen2( args )
        stdin.close()

        aCpuNumber = 0
        for i in stdout:
            aCpuNumber = aCpuNumber + 1

        return aCpuNumber

    def _createSessionProxy(self):
        '''creates and returns new SessionProxy instance
        Return SessionProxy
        '''
        return SessionProxy( self, self.getNextJobID() )

    def manageJob( self, job ):
        self.__jobsByLSFJobID[ job.getLSFJobID() ] = job

    def getJobByLSFJobID( self, jobID ):
        return self.__jobsByLSFJobID[ jobID ]

    def _updateStatus( self ):
        out = raiseExceptionOnError(
            RuntimeError,
            pollForOutputs(
                popen2.Popen3( ( BJOBS, '-u', self.getOwner(), '-a' ), True )
                )
            ).split( "\n" )
        header = out[ 0 ].split()
        header = dict( zip( header, xrange( 0, header.length ) ) )
        for row in out[ 1: ]:
            row = line.rstrip().split()
            jobID = row[ header[ 'JOBID' ] ]
            status = row[ header[ 'STAT' ] ]
            if status == 'DONE':
                self.getJobByLSFJobID( jobID ).setStatus( FINISHED )
            elif aLSFSessionProxyStatus not in ( "RUN", "PEND", "SSUSP" ):
                self.setStatus( ERROR )

    def removeSessionProxy( self, job ):
        AbstractSystemProxy.removeSessionProxy( self, job )
        try:
            del self.__jobsByLSFJobID[ self.getLSFJobID() ]
        except:
            pass

