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

import os
import popen2
import re
import signal
import sys
import time
from itertools import ifilter

from session_manager.Util import *
from session_manager.SessionManager import *
from session_manager.Constants import *

QSUB = 'qsub'
QSTAT = 'qstat'
QHOST = 'qhost'
QDEL = 'qdel'

class SessionProxy( AbstractSessionProxy ):
    '''SessionProxy class
    Target environment is local PC that has only one cpu.
    '''

    def __init__( self, dispatcher, jobID ):
        AbstractSessionProxy.__init__( self, dispatcher, jobID )
        self.__theSGEJobID = -1
        self.__theSessionProxyName = "script." + os.path.basename( getCurrentShell() )

    def __del__( self ):
        self.__cancel()

    def getSGEJobID(self):
        '''return job id 
        '''
        return self.__theSGEJobID 

    def run( self ):
        '''run process
        Return None
        '''
        if not AbstractSessionProxy.run( self ):
            return False

        # save current directory
        aCwd = os.getcwd()

        try:
            os.chdir( self.getJobDirectory() )

            args = [ QSUB ]
            for key, val in self.getEnvironmentVariables().iteritems():
                args.extend( ( '-v', key + '=' + val ) )
            args.append( '-cwd' )
            args.extend( ( '-S', self.getInterpreter()) )
            args.extend( ( '-o', self.getStdoutFileName() ) )
            args.extend( ( '-e', self.getStderrFileName() ) )
            args.extend( ifilter(
                    lambda x: x not in ( '-s' '-v', '-cwd', '-o', '-e' ),
                    self.getOptionList() ) )
            args.append( self.getScriptFileName() )
            args.extend( self.getArguments() )

            f = open('queue.sh', 'w')
            try:
                f.write('#!/usr/bin/bash\n%s\n' % (' '.join(args)))
            finally:
                f.close()

            msg = raiseExceptionOnError(
                RuntimeError,
                pollForOutputs( popen2.Popen3( args, True ) )
                )
            m = re.match(
                r'Your job (\d+) \("(?:[^"]|\\")*"\) has been submitted', msg
                )
            if m == None:
                raise RuntimeError, '%s returned unexpected result: %s' % (
                        QSUB, msg )
            self.__theSGEJobID = m.group( 1 )
        finally:
            os.chdir( aCwd )
        return True

    def __cancel( self ):
        if self.__theSGEJobID >= 0:
            raiseExceptionOnError( RuntimeError,
                pollForOutputs(
                    popen2.Popen3( ( QDEL, self.__theSGEJobID ), True )
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
        self.setStatus( ERROR ) 

class SystemProxy( AbstractSystemProxy ):
    def __init__( self, sessionManager ):
        # Call superclass's constructor.
        AbstractSystemProxy.__init__( self, sessionManager )

        # Check the existance of SGE command.
        # qsub
        for binName in ( QSUB, QSTAT, QHOST ):
            if checkCommandExistence( binName ) == False:
                raise Exception(
                    "\"%s\" is not included in the $PATH environment"\
                                % binName )
        # Initialize attributes.
        self.__theQueueList = None

    def getDefaultConcurrency( self ):
        '''returns default cpus
        Return int : the number of cpus
        The default number of cpu is calculated from the retuslt of qhost
        Here is the sample of standard output of qhost

        HOSTNAME             ARCH       NPROC  LOAD   MEMTOT   MEMUSE   SWAPTO   SWAPUS
        -------------------------------------------------------------------------------
        global               -              -     -        -        -        -        -
        i10                  glinux         2  0.00  1006.1M   637.3M     2.0G   111.6M
        i11                  glinux         2  0.01  1006.1M   666.7M     2.0G   144.0M
        i12                  glinux         2  0.00  1006.1M   295.6M     2.0G     2.1M
        i13                  glinux         2  0.00  1006.1M   297.2M     2.0G     2.4M

        In this case, this method returns 4 as a default number of CPUs.
        '''

        # returns the number of cpu
        aCpuNumber = -3 # for the headers to be reduced
        for i in os.popen( QHOST ):
            aCpuNumber = aCpuNumber + 1

        if aCpuNumber < 0:
            aCpuNumber = 0

        return aCpuNumber

    def _createSessionProxy( self ):
        '''creates and returns new SessionProxy instance'''
        return SessionProxy( self, self.getNextJobID() )

    def _updateStatus( self ):
        '''updates status
        Updates status using the result of qstat as below.

        job-ID  prior name       user         state submit/start at     queue      master  ja-task-ID 
        ---------------------------------------------------------------------------------------------
        4243651     0 test.py    hoge         Eqw   08/28/2003 11:24:26                       

        When the state is 'qw(queued/waiting)', 't(transfer)' or 'r(running)', it is set as RUN.
        When the state is 'Eqw(error)', the job is killed and its status is set as ERROR.
        In most of latter case, the interpreter can't find remote machine could not be found.
        However, above error does not be written stderr.
        '''

        # initializes a dict whose key is SGE job id and value is status
        aStatusDict = {}

        # reads the result of qstat
        out = raiseExceptionOnError(
            RuntimeError,
            pollForOutputs(
                popen2.Popen3( ( QSTAT, '-u', self.getOwner() ), True )
                )
            ).split( "\n" )

        # When there are running jobs, gets SGE job id and status
        for aLine in out[ 2: -1 ]:
            comps = aLine.split()
            aStatusDict[ comps[ 0 ] ] = comps[ 4 ]

        # checks ths status of each SessionProxy
        for job in self.getSessionProxies():
            # considers only running jobs
            if job.getStatus() == RUN:
                # gets SGE job id
                aSGEJobID = job.getSGEJobID()

                # there is no SGE job id in the result of qstat, the job is 
                # considered to be finished
                if not aStatusDict.has_key( aSGEJobID ):
                    # read standard error file
                    aStderrFile = job.getStderrFilePath()
                    # When something is written in standard error,
                    if os.path.exists( aStderrFile ) and \
                       os.stat( aStderrFile )[ 6 ] > 0:
                        job.setStatus( ERROR )
                    else:
                        job.setStatus( FINISHED )
                else:
                    # When job is running,
                    if aStatusDict[ aSGEJobID ].find( 'E' ) != -1:
                        # When character 'E' is included in the status,
                        job.stop()
                    else:
                        pass

    def __populateQueueList( self ):
        # Get queue list.
        if self.__theQueueList != None:
            return

        lines = os.popen("%s -q" % QHOST)
        lines.readline()
        lines.readline()

        hostname = None
        queueList = {}
        for line in lines:
            m = re.match( r'^([^ ]*) *([^ ]*)' )
            if m != None:
                if m.groups( 0 ) != '':
                    if hostname != None:
                        queueList[ hostname ] = queuesInHost
                    hostname = m.groups( 0 )
                    queuesInHost = []
                else:
                    queuesInHost.append( m.groups( 1 ) )
        queueList[ hostname ] = queuesInHost
        lines.close()
        self.__theQueueList = queueList
