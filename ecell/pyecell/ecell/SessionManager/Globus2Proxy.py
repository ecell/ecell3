#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
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
# Designed by Koichi Takahashi <shafi@e-cell.org>
# Programmed by Masahiro Sugimoto <msugi@sfc.keio.ac.jp>

# imports standard modules
import sys
import os
import time
import signal
import re
import popen2


# imports ecell modules
from ecell.SessionManager.Util import *
from ecell.SessionManager.SessionManager import *
from ecell.SessionManager.Constants import *

GRID_INFO_SEARCH = 'grid-info-search'
GLOBUS_JOB_RUN='globus-job-run'
GLOBUS_JOB_SUBMIT='globus-job-submit'
GLOBUS_JOB_GET_STATUS='globus-job-status'
GLOBUS_JOB_GET_OUTPUT='globus-job-get-output'
GLOBUS_JOB_CLEAN='globus-job-clean'
GLOBUS_URL_COPY='globus-url-copy'
MKDIR='/bin/mkdir'
RM='/bin/rm'

class SessionProxy( AbstractSessionProxy ):
    def __init__( self, dispatcher, jobID ):
        AbstractSessionProxy.__init__( self, dispatcher, jobID )

        # initialize parameter
        self.__theContactString = None

        # initialize script name
        # if current shell is '/bin/tcsh', set 'script.tcsh' 
        self.__theTmpScriptFileName = "script." \
                + os.path.basename( getCurrentShell() )
        self.__theCpu = None

    def getContactString( self ):
        '''return job id 
        '''
        return self.__theContactString 

    def __submitSessionProxy( self, args, environmentVars = {}, dir = None ):
        _args = [
            GLOBUS_JOB_RUN,
            self.__theCpu
            ]
        if dir != None:
            _args.extend( ( '-dir', dir ) )
        for i in environmentVars:
            _args.extend( ( '-env', '%s=%s' % i ) )
        _args.extend( args )
        stdout, stdin = popen2.popen2( args )
        stdin.close()
        return stdout.read()

    def __transferFiles( self, destCpu, files, dest ):
        for file in files:
            aLocalFile  = "file://" + file.replace( os.sep, '/' )
            aRemoteFile = "gsiftp://" + destCpu + '/' \
                    + dest.replace( os.sep, '/' ) \
                    + os.path.basename( file )
            os.spawnlp( os.P_WAIT, GLOBUS_URL_COPY,
                GLOBUS_URL_COPY, aLocalFile, aRemoteFile )

    def run( self ):
        '''run process
        Return None
        '''
        if not AbstractSessionProxy.run( self ):
            return False

        self.__theCpu = self.getSystemProxy().getFreeCpu()

        aCwd = os.getcwd()
        try:
            os.chdir( self.getJobDirectory() )
            self.__submitSessionProxy( ( MKDIR, '-p',
                    os.path.abspath( self.getJobDirectory() ) ) )

            files = [ self.getScriptFileName() ] 
            files.extend( self.getExtraFileList() )
            self.__transferFiles( self.__theCpu, map( os.path.abspath, files ),
                    self.getJobDirectory() )
            self.__theContactString = self.__submitSessionProxy( (
                self.getInterpreter(),
                '-stdin', '-s', self.getScriptFileName() ),
                self.getEnvironmentVariables(),
                self.getJobDirectory() )
        except:
            self.setStatus( ERROR )
            return False
        finally:
            os.chdir( aCwd )
        return True

    def getCpu( self ):
        '''Return cpu name
        Return str : cpu name
        '''

        return self.__theCpu

    def __cancel( self ):
        raiseExceptionOnError( RuntineError,
            pollForOutputs( popen2.Popen3(
                ( GLOBUS_JOB_CLEAN, self.__theContactString ), True ) )
            )

    def stop( self ):
        '''stop the job
        Return None
        '''

        # When this job is running, stop it.
        if self.getStatus() == RUN:
            self.__cancel()

        # set error status
        self.setStatus(ERROR) 

class SystemProxy( AbstractSystemProxy ):
    '''Globus2SystemProxy'''
    def __init__( self, sessionManager ):
        '''Constructor
        sessionManager -- the reference to SessionManager
        '''

        # calls superclass's constructor
        AbstractSystemProxy.__init__( self, sessionManager )

        self.__theIdentity = None
        self.__theHostList = []
        self.__theFreeCpuList = None
        self._updateStatus()

    def __del__( self ):
        '''When Globus2SystemProxy exits, this method id called.
        Remove tmp root directory on remote hosts.
        '''

        for aHost in self.__theHostList:
            if aHost != self.getLocalHost():
                self.__submitSessionProxy( aHost, RM, "-rf",
                        self.getSessionManager().getTmpRootDir(),
                        dir = os.getcwd() )

    def setHosts( self, hostlist ):
        '''Set host names on which the jobs will be conducted
        hostlist(list of str) -- list of host name
        Return None
        '''
        self.__theHostList = list( hostlist )

    def getDefaultConcurrency( self ):
        '''returns default cpus
        Return int : the number of cpus
        '''
        return 1

    def _createSessionProxy( self ):
        '''creates and returns new SessionProxy instance
        Return SessionProxy
        '''
        return SessionProxy( self, self.getNextJobID() )

    def getFreeCpuList( self ):
        '''Return free cpus
        Return list of str : cpu names
        '''
        if self.__theFreeCpuList == None:
            self._updateStatus()
        return self.__theFreeCpuList

    def getFreeCpu( self ):
        if self.__theFreeCpuList == None:
            self._updateStatus()
        return self.__theFreeCpuList.pop()

    def _updateStatus( self ):
        '''update jobs's status
        Return None
        '''
        freeCpuList = []
        for aHost in self.__theHostList:
            stdout, stdin, stderr = popen2.popen3( (
                GRID_INFO_SEARCH,
                '-h', aHost,
                'objectClass=MdsCpu'
                'Mds-Cpu-Free-1minX100'
                ) )
            stdin.close()
            for aLine in stdout:
                m = re.match( r'Mds-Cpu-Free-1minX100:\s*(\d+)', aLine )
                if m != None:
                    aFreeCpu = int( m.groups( 0 ) )
                    if aFreeCpu > 60:
                        freeCpuList.append( aHost )
                    break
            stdout.close()
        self.__theFreeCpuList = freeCpuList

        for job in self.getSessionProxies():
            stdout, stdin = popen2.popen2(
                ( GLOBUS_JOB_GET_STATUS, job.getContactString() ) )
            stdin.close()
            status = stdout.readline()[:-1]

            # if the status is done, copy remote output files
            # to local machine.
            if status == 'DONE' and job.getStatus() != FINISHED:
                job.setStatus( FINISHED ) 

                # save current directory
                aCwd = os.getcwd()
                try:
                    os.chdir( job.getJobDirectory() )

                    dest = open(
                        os.path.join( os.getcwd(), job.getStdoutFileName() ),
                            'wb' )

                    try:
                        stdout, stdin = popen2.popen2(
                            ( GLOBUS_JOB_GET_OUTPUT, job.getContactString() ) )
                        stdin.close() 
                        copyfileobj( stdout, dest )
                    finally:
                        close( dest )

                    dest = open(
                        os.path.join( os.getcwd(), job.getStderrtFileName() ),
                            'wb' )

                    try:
                        stdout, stdin = popen2.popen2(
                            ( GLOBUS_JOB_GET_OUTPUT, '-err',
                              job.getContactString() ) )
                        stdin.close() 
                        copyfileobj( stdout, dest )
                    finally:
                        close( dest )
                finally:
                    os.chdir( aCwd )

