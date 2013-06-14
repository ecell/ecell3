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
import signal
import sys
import time

try:
#    raise Exception
    import subprocess
except:
    subprocess = None
    class Process:
        def __init__( self, pid ):
            self.pid = pid

        def poll( self ):
            status = os.waitpid( self.pid, os.WNOHANG )
            if not os.WIFEXITED( status[ 1 ] ):
                return None
            else:
                return os.WEXITSTATUS( status[ 1 ] )

        def wait( self ):
            return os.WEXITSTATUS( os.waitpid( self.pid, 0 )[ 1 ] )

from session_manager.SessionManager import *
from session_manager.Constants import *
from session_manager.Util import *

class SessionProxy( AbstractSessionProxy ):
    '''LocalSessionProxy class
    Target environment is local PC that has only one cpu.
    '''

    def __init__( self, dispatcher, jobID ):
        # call superclass's constructor
        AbstractSessionProxy.__init__( self, dispatcher, jobID )
        # initialize parameter
        self.__theSubProcess = None

    def run( self ):
        if not AbstractSessionProxy.run( self ):
            return False

        args = []
        args.append( self.getInterpreter() )
        args.append( self.getScriptFileName() )
        args.extend( self.getArguments() )

        env = dict( os.environ )
        env.update( self.getEnvironmentVariables() )

        stdout = stderr = None

        try:
            stdout = open( os.path.join(
                    self.getJobDirectory(),
                    self.getStdoutFileName() ), 'wb' )
            stderr = open( os.path.join(
                    self.getJobDirectory(),
                    self.getStderrFileName( )), 'wb' )
            if subprocess:
                self.__theSubProcess = subprocess.Popen(
                        args, stderr = stderr, stdout = stdout,
                        cwd = self.getJobDirectory(),
                        env = env )
            else:
                cwd = os.getcwd()
                pid = os.fork()
                if not pid:
                    os.chdir( self.getJobDirectory() )
                    os.dup2( stdout.fileno(), sys.stdout.fileno() )
                    os.dup2( stderr.fileno(), sys.stderr.fileno() )
                    os.execvpe( args[ 0 ], args, env )
                self.__theSubProcess = Process( pid )
            self.setStatus( RUN )
        finally:
            if stdout:
                stdout.close()
            if stderr:
                stderr.close()
        return True

    def getProcessStatus( self ):
        return self.__theSubProcess.poll()

    def stop(self):
        '''stop this job'''
        if self.getStatus() == RUN:
            os.kill( self.__theSubProcess.pid, 1 )
        # set error status
        self.setStatus(ERROR) 

class SystemProxy( AbstractSystemProxy ):
    '''SystemProxy'''

    def __init__( self, sessionManager ):
        '''constructor
        sessionManager -- the reference to SessionManager
        '''
        # calls superclass's constructor
        AbstractSystemProxy.__init__( self, sessionManager )

    def getDefaultConcurrency(self):
        '''returns default cpus
        Return int : the number of cpus
        '''
        # returns the number of cpu
        return 1

    def _createSessionProxy( self ):
        '''creates and returns new LocalSessionProxy instance
        Return LocalSessionProxy
        '''
        # creates and returns new LocalSession Proxy instance
        return SessionProxy( self, self.getNextJobID() )

    def createSessionProxy(self):
        '''creates and returns new SessionProxy instance
        Return SessionProxy
        '''
        id = self.getNextJobID()
        job = SessionProxy( self, id )
        self.jobs[ id ] = job
        return job

    def _updateStatus( self ):
        for job in self.getSessionProxies():
            # check the running status 
            if job.getStatus() != RUN:
                continue 
            retcode = job.getProcessStatus()
            if retcode == None:
                continue
            elif retcode == 0: 
                job.setStatus( FINISHED )
            else:
                job.setStatus( ERROR )
