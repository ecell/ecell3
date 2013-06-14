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

# imports standard modules
import os
import re
import imp
import shutil
import threading
import sys
import time
import weakref
import signal
import inspect
import ecell.eml

from session_manager.Constants import *
from session_manager.Util import *

__all__ = (
    'AbstractSessionProxy',
    'AbstractSystemProxy',
    'SessionManager',
    )

class AbstractSessionProxy:
    '''AbstractSessionProxy is an abstract class that manages one process'''

    def __init__( self, dispatcher, jobID ):
        self.__theDispatcher = weakref.ref( dispatcher )
        self.__theLastJobID = jobID

        # set status as QUEUED
        self.__theStatus = QUEUED

        # initializes attributes
        self.__theTimeout = 0
        self.__theScriptFileName = None
        self.__theInterpreter = None
        self.__theArgument = None
        self.__theExtraFileList = None
        self.__theRemovalStatus = False
        self.__theJobDirectory = None
        self.__theRetryCount = 0
        self.__theStartTime = 0
        self.theOutputCopyDoneStatus = False
        self.__theEnvironmentVariables = {}
        self.__theOptionList = []
        self.__theArguments = []

    def __del__( self ):
        '''Before destractor, this method is called. 
        Return None
        '''

        # When the job has ERROR status, do not
        # delete the job directory.
        if self.getStatus() != ERROR:
            self.clearJobDirectory()

    def getJobID( self ):
        '''Return the job id
        Return int : job id
        ''' 

        # return the job id
        return self.__theLastJobID 

    def getStartTime( self ):
        return self.__theStartTime

    def setScriptFileName( self,scriptfilename ):
        '''Set a script file name.
        scriptfilename(str) -- a script file name
        Return None
        ''' 

        # Set the script file name.
        self.__theScriptFileName = scriptfilename

    def getScriptFileName( self ):
        '''return the script file name
        Return str : script file name
        ''' 

        # Return the script file name
        return self.__theScriptFileName 

    def setInterpreter( self, interpreter ):
        '''Set an interpreter.
        interpreter(str) -- an interpreter
        Return None
        '''
        if not os.path.isabs( interpreter ):
            interpreter = lookupExecutableInPath( interpreter )
        self.__theInterpreter = interpreter

    def getInterpreter( self ):
        '''Return the interpreter.
        Return str : interpreter
        ''' 

        # return the interpreter
        return self.__theInterpreter 

    def setArgument( self, argument ):
        '''Set an argument.
        argument(str) -- an argument to be set to script
        Return None
        ''' 
        # set the argument
        self.__theArguments = argument.split()

    def setArguments( self, arguments ):
        self.__theArguments = arguments

    def getArgument( self ):
        '''Return the argument.''' 

        # return the artument
        return ' '.join( self.__theArguments )

    def getArguments( self ):
        return self.__theArguments

    def setExtraFileList( self, extrafilelist ):
        '''Set an extra file list.
        extrafilelist(list of str)  -- extra file list
        Return None
        ''' 
        self.__theExtraFileList = list( extrafilelist )

    def getExtraFileList( self ):
        '''Returns the extra file list.
        Return list of str : extra file list
        ''' 

        # return the extra file list
        return self.__theExtraFileList 

    def setTimeout( self, timeout ):
        '''Set a timeout.
        When timeout is 0, no limit is set.
        timeout(int) -- time out (sec.)
        Return None
        '''

        # set a timeout
        self.__theTimeout = timeout

    def getTimeout( self ):
        '''Return the timeout.
        Return int : time out (sec.)
        '''

        # return the time out
        return self.__theTimeout 

    def run( self ):
        '''run process
        Return None
        [Note] This method must be called from the run method of sub class.
        '''

        if self.getStatus() != QUEUED:
            return False

        # When status is RUN, and retry count is over max retry count,
        # set ERROR status and does nothing.
        if self.getRetryMaxCount() != 0:
            if self >= self.getRetryMaxCount():
                self.setStatus( ERROR )
                return False

        # set a status
        self.setStatus( RUN )
        # count up retry counter
        self.__theRetryCount += 1
        # save current time
        self.__theStartTime = time.time()

        return True

    def stop( self ):
        '''ABSTRACT : This method must be overwritteng in subclass
        stop job
        Return None
        raise NotImplementedError
        '''

        # When this method is not implemented in sub class, 
        # raise NotImplementedError
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + ' must be implemented in subclass')

    def setStatus( self, status ):
        '''Set a status.
        aStatus(int) -- QUEUED,RUN,ERROR, or FINISHED
        Return None
        '''
        oldStatus = self.__theStatus
        self.__theStatus = status
        self.getSystemProxy()._statusChanged( self, oldStatus )

    def getStatus( self ):
        '''Return the status.
        Return int : QUEUED,RUN,ERROR, or FINISHED
        '''

        # return the status
        return self.__theStatus

    def getRetryCount( self ):
        '''Returns the retry count.
        Return int : retry count
        '''

        # return the retry count
        return self.__theRetryCount

    def getRetryMaxCount( self ):
        '''Return the retry max count.
        Return int : retry max count
        '''
        # return the retry max count
        return self.getSystemProxy().getRetryMaxCount()

    def setJobDirectory( self, jobdirectory ):
        '''Set a job directory.
        jobdirectory(str) -- job directory
        Return None
        '''

        # save job directory
        self.__theJobDirectory = jobdirectory

    def getJobDirectory( self ):
        '''Return job directory.
        Return str : job directory
        '''

        # return job directory
        # if self.__theJobDirectory does notend with os.sep
        # append os.sep
        if self.__theJobDirectory[-1] != os.sep:
            return self.__theJobDirectory + os.sep
        else:
            return self.__theJobDirectory

    def clearJobDirectory( self ):
        '''Remove job directory if there is.
        Return None
        '''

        # when the job directory exists, deletes it.
        if os.path.isdir( self.__theJobDirectory ):
            shutil.rmtree( self.__theJobDirectory)

        # if not, do nothing
        else:
            pass

    def getStdoutFilePath( self ):
        return os.path.join( self.getJobDirectory(),
                self.getStdoutFileName() )

    def getStderrFilePath( self ):
        return os.path.join( self.getJobDirectory(),
                self.getStderrFileName() )

    def getStdout( self ):
        '''Return stdout(str)
        '''
        return open( self.getStdoutFilePath(), 'rb' ).read()

    def getStderr( self ):
        '''Return stderr(str)
        '''
        return open( self.getStderrFilePath(), 'rb' ).read()

    def setOptionList( self, optionlist ):
        '''Set an option list.
        optionlist(list of str) -- a list of options
        Return None
        '''
        self.__theOptionList = tuple( optionlist )

    def getOptionList( self ):
        '''Get the option list.
        Return list of str
        '''
        return self.__theOptionList 

    def getEnvironmentVariables( self ):
        return self.__theEnvironmentVariables

    def setEnvironmentVariable( self, name, val ):
        self.__theEnvironmentVariables[ name ] = val

    def getSystemProxy( self ):
        return self.__theDispatcher()

    def getStdoutFileName( self ):
        return self.getSystemProxy().getStdoutFileName()

    def getStderrFileName( self ):
        return self.getSystemProxy().getStderrFileName()

    def clear( self ):
        self.stop()
        self.getSystemProxy().removeSessionProxy( self )

class AbstractSystemProxy:
    '''AbstractSystemProxy abstract class'''

    def __init__( self, sessionManager ):
        '''constructor
        sessionManager -- a reference to SessionManager
        '''
        assert isinstance( sessionManager, SessionManager )
        self.jobs = {}
        self.optionList = []
        self.__theSessionProxyCountByStatus = {
            QUEUED: 0,
            RUN: 0,
            FINISHED: 0,
            ERROR:0
            }
        self.__theSessionManager = weakref.ref( sessionManager )
        self.__theLocalHost = os.popen('hostname').readline()[:-1]
        self.__theLastJobID = 0
        self.__theStdout = DEFAULT_STDOUT
        self.__theStderror = DEFAULT_STDERR
        self.__autoRetry = True
        self.__theRetryMaxCount = 0
        self.__theOwner = getCurrentUserName()

    def getLocalHost( self ):
        return self.__theLocalHost

    def getSessionManager( self ):
        return self.__theSessionManager()

    def getDefaultConcurrency( self ):
        '''ABSTRACT : This method must be overwritteng in subclass
        returns default cpus
        Return int : the number of cpus
        raise NotImplementedError
        '''
        # When this method is not implemented in sub class,
        # raise NotImplementedError
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + ' must be implemented in subclass')

    def _updateStatus( self ):
        '''ABSTRACT : This method must be overwritteng in subclass
        returns default cpus
        Return int : the number of cpus
        raise NotImplementedError
        '''
        # When this method is not implemented in sub class,
        # raise NotImplementedError
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + ' must be implemented in subclass')

    def update( self ):
        '''
        updates all AbstractSessionProxys' status
        Return None
        raise NotImplementedError
        '''
        self._updateStatus()
        # updates all AbstractSessionProxy's status
        for aSessionProxy in self.getSessionProxies():
            if self.__autoRetry:
                # When the job is running and running time is > timeout,
                # call retry method of sub class
                if aSessionProxy.getStatus() == RUN:
                    timeout = aSessionProxy.getTimeout()
                    if timeout != 0 and \
                       time.time() - aSessionProxy.getStartTime() >= timeout:
                        aSessionProxy.stop()
                        aSessionProxy.setStatus( QUEUED )

        # calculates the number of jobs to be run
        aDispatchNumber = self.getSessionManager().getConcurrency() \
                          - self.getNumOfSessionProxiesByStatus( RUN )

        # When some jobs to be runned,
        if aDispatchNumber == 0:
            return
        for aSessionProxy in self.getSessionProxies():
            # when the status is QUEUED
            if aSessionProxy.getStatus() == QUEUED:
                # calls run method
                aSessionProxy.run()

                # counts up
                aDispatchNumber -= 1
                if aDispatchNumber == 0:
                        break

    def _createSessionProxy( self ):
        '''ABSTRACT : This method must be overwritteng in subclass
        creates and returns new AbstractSessionProxy instance
        Return AbstractSessionProxy
        raise NotImplementedError
        '''

        # When this method is not implemented in sub class,
        # raise NotImplementedError
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + ' must be implemented in subclass')

    def createSessionProxy( self ):
        job = self._createSessionProxy()
        job.setOptionList( self.getOptionList() )
        self.jobs[ job.getJobID() ] = job
        self.__theSessionProxyCountByStatus[ job.getStatus() ] += 1
        return job

    def getOptionList( self ):
        return self.optionList

    def setOptionList( self, optionlist ):
        self.optionList = tuple( optionlist )

    def getNextJobID( self ):
        '''Return next job ID.
        Return int : job ID
        [Note]:private method, the first job id is 1
        '''
        # count up job id
        self.__theLastJobID += 1
        # return job id
        return self.__theLastJobID

    def setStdoutFileName( self, stdout ):
        '''Set an standard output.
        stdout(str or file) -- file name or file object to be wrote
        Return None
        [Note] default stdout is 'stdout'
        '''
        self.__theStdout = stdout

    def getStdoutFileName( self ):
        '''Return the standard output.
        Return str or file -- standard output
        [Note] default stdout is 'stdout'
        '''
        return self.__theStdout

    def setStderrFileName( self, stderror ):
        '''Set a standard error.
        stdout(str or file) -- file name or file object to be wrote
        Return None
        [Note] default stderr is 'stderr'
        '''
        self.__theStderror = stderror

    def getStderrFileName( self ):
        '''Return standard error.
        Return str or file -- standard error
        [Note] default stderr is 'stderr'
        '''
        return self.__theStderror

    def getStdout( self, jobid ):
        '''return the stdout of the job

        jobid(int) -- job id
        
        Return str : the stdout
        '''
        return self.jobs[ jobid ].getStdout()

    def getStderr( self, jobid ):
        '''return the stderr of the job

        jobid(int) -- job id
        
        Return str : the stderr
        '''
        return self.jobs[ jobid ].getStderr()

    def getJobDirectory( self, jobid ):
        '''Return the job directory name

        jobid(int) -- job id

        Return str : the path of the job directory
        '''

        # return the path of the job directory
        return self.jobs[ jobid ].getJobDirectory()

    def setRetryMaxCount( self, count ):
        '''Set retry max count.
        When count < 0, count is set as 0
        count(int) -- retry max count
        Return None
        [Note] default max count is 0
        '''
        if count < 0:
            count = 0
        self.__theRetryMaxCount = count
    setRetryLimit = setRetryMaxCount # for backwards compatibility

    def getRetryMaxCount( self ):
        return self.__theRetryMaxCount

        self.__theSystemProxy.setRetryMaxCount(retrylimit )
    getRetryLimit = getRetryMaxCount # for backwards compatibility

    def getOwner( self ):
        return self.__theOwner

    def setOwner( self, owner ):
        self.__theOwner = owner

    def getSessionProxies( self ):
        return self.jobs.itervalues()

    def getSessionProxy( self, jobID ):
        return self.jobs[ jobID ]

    def getNumOfSessionProxiesByStatus( self, stat ):
        return self.__theSessionProxyCountByStatus[ stat ]

    def getSessionProxiesByStatus( self, stat ):
        '''Return the list of queued job.
        Return list : the list of AbstractSessionProxy
        '''
        # initializes a job list
        return ( job for job in self.getSessionProxies() \
                 if job.getStatus() == stat )

    def removeSessionProxy( self, job ):
        del self.jobs[ job.getJobID() ]

    def _statusChanged( self, job, oldStatus ):
        self.__theSessionProxyCountByStatus[ oldStatus ] -= 1
        self.__theSessionProxyCountByStatus[ job.getStatus() ] += 1

class SessionManager( object ):
    '''SessionManager class
    Provide API to execute multiple jobs concurrently.
    You can access AbstractSessionProxy directory to control each job.
    '''
    __delegated__ = (
        'setRetryLimit',
        'getRetryLimit',
        'setRetryMaxCount',
        'getRetryMaxCount',
        'getSessionProxy',
        'getSessionProxies',
        'setStdoutFileName',
        'getStdoutFileName',
        'setStderrFileName',
        'getStderrFileName',
        'getJobDirectory',
        'getStdout',
        'getStderr',
        'setRetryMaxCount',
        'setOptionList',
        'getOptionList',
        'setOwner',
        'getOwner',
        )

    def __metaclass__( name, base, dict ):
        for mtd in dict[ '__delegated__']:
            def def_fun( mtd ):
                return lambda self, *args, **kwargs: \
                    getattr( self.__theSystemProxy, mtd )( *args, **kwargs )
            dict[ mtd ] = def_fun( mtd )
        return type( name, base, dict )

    def __del__( self ):
        '''When SessionManager exits, this method is called.
        In this method, delete tmp root directory.
        Running jobs will be deleted from each AbstractSessionProxy instance.
        This method should be substituded to sys.exitfunc.
        '''
        if self.__theTmpRemovable == True:
            # remove the temporary directory
            if os.access( self.getTmpRootDir(), os.W_OK ):
                shutil.rmtree( self.getTmpRootDir() )
        self.__theSystemProxy = None

    def __init__( self, modulepath, concurrency, environment ):
        '''Constructor
        Initialize parameters
        modulepath(str) -- a module path
        concurrency(int) -- a concurrency
        environment(str) -- an environment to be used
        '''

        # set module path
        self.__theModulePath = modulepath

        # initialize parameters
        self.__theSystemProxyDict = None
        self.__theMessageMethod = self.__plainMessageMethod

        self.__theConcurrency = concurrency
        self.__theTmpRootDir = None
        self.__theTmpDir = None
        self.__theFinishedSessionProxyRemovableFlag = True
        self.__theErroRemovableSessionProxyFlag = True
        self.__theUpdateInterval = 1
        self.__theGlobalRunTimeout = 0

        self.__theTmpRemovable = True

        # set default temporary directory
        # Temporary directory is created in run method.
        self.setTmpRootDir( DEFAULT_TMP_DIRECTORY )

        # let default environment
        self.setEnvironment( environment )

    def setEnvironment( self, environment ):
        '''Set environment parameter.
        Try to import the module whole name is environment + AbstractSystemProxy
        For example, if the 'environment' is Local, try to LocalSystemProxy
        and call constroctor of it.

        environment(str) -- Local,SMP,SGE,Globus,etc.
        Return None
        '''

        # create system proxy instance
        aSystemProxyName = environment + 'Proxy';

        # find module
        aFilePoint, aPath, self.__theDescription = imp.find_module(
                aSystemProxyName,
                self.__theModulePath )
        aSystemProxyModule = imp.load_module(
                aSystemProxyName,
                aFilePoint,
                aPath,
                self.__theDescription)

        aConstructor = aSystemProxyModule.__dict__[ SYSTEM_PROXY ]
        self.__theSystemProxy = aConstructor( self )

    def getEnvironment( self ):
        '''Return current environment parameter as str.

        Return str : 'Local','SMP','SGE','Globus', etc.
        '''

        # get class name of system proxy
        anEnvironmentStr = self.__theSystemProxy.__class__.__name__

        # remove the charactor of SYSTEM_PROXY, and return it
        return re.sub(SYSTEM_PROXY,'',anEnvironmentStr)

    def setConcurrency( self, number ):
        '''Set the number of jobs to be runned concurrently.

        number(int) -- the number of concurrent jobs
        Return None 
        '''

        # set the number of concurrency
        self.__theConcurrency = number

    def getConcurrency( self ):
        '''Return the number of jobs to be runned concurrently.

        Return int : the number of concurrent jobs
        '''

        if self.__theConcurrency == None:
            self.__theConcurrency = self.getDefaultConcurrency()

        # return the number of concurrency
        return self.__theConcurrency 

    def getDefaultConcurrency( self ):
        '''Return the default number of jobs to be runned concurrently.

        Return int : the default number of concurrency
        '''

        # return the default number of concurrency
        return self.__theSystemProxy.getDefaultConcurrency()

    def setTmpRootDir( self, tmprootdir ):
        '''Set temporary root directory.
        When run method is called, tmprootdir is created on current working 
        directory.  Then, below 'tmp' directory, the directory whose name 
        is same as pid of this process is also created.
        For example, if tmprootdir is 'work' and pid is 12345, './work/12345'
        is created whtn run method is called.

        tmprootdir(str) -- a temporary root directory to be set. default is 'tmp'
        Return None
        '''

        # sets tmp root directory
        self.__theTmpRootDir = tmprootdir

        # sets tmp directory name
        self.__theTmpDir = os.path.join(
            self.__theTmpRootDir, str( os.getpid() ) )

    def getTmpRootDir( self ):
        '''Return the temporary root directory
        See setTmpRootDir about the detail of temporary root directory

        Return str : temporaty root directory
        '''

        # return tmp root directory
        return self.__theTmpRootDir

    def getTmpDir( self ):
        '''Return temporary directory
        Temporary directory is created below temporary root directory,
        and named pid of this process.
        See setTmpRootDir about the detail of temporary directory
        For example, if tmprootdir is 'work' and pid is 12345, 
        this method returns './work/12345'.
        
        Return str : a temporaty directory
        '''

        # return temporary directory
        return self.__theTmpDir
    
    def setTmpDirRemovable( self, deleteflag ):
        '''Set a removable flag of tmp directory.

        deleteflag(boolean)  --  True  : delete tmp and tmp root directory,
                                         when destructor is called.
                                 False : tmp and tmp root directory is not
                                         deleted when destructor is called.
                                 True is set as default.
        Return None
        '''

        # save deleteflat to instance attribute
        self.__theTmpRemovable = deleteflag

    def getTmpDirRemovable( self ):
        '''Return the removable flag of tmp directory.

        Return boolean : True  : delete tmp and tmp root directory,
                                 when destructor is called.
                         False : tmp and tmp root directory is not
                                 deleted when destructor is called.
        '''

        # return delete flag of tmp directory
        return self.__theTmpRemovable 

    def registerSessionProxy( self, scriptfile, interpreter, arguments = None,
                     extrafilelist = [], timeout = 0 ):
        '''registers a new job
        scriptfile(str)            -- script file name
        interpreter(str)           -- interpreter name to run script
        argument(str)              -- argument set be set to script
        extrafilelist(list of str) -- list of extra file name
        timeout(int)               -- set time out (sec.). When timeout<=0, no limit is set.
        return int : job id
        '''
        # creates AbstractSessionProxy
        job = self.__theSystemProxy.createSessionProxy()
        job.setScriptFileName( scriptfile )
        job.setInterpreter( interpreter )
        if arguments != None:
            if isinstance( arguments, str ) \
               or isinstance( arguments, unicode ):
                arguments = arguments.split()
            job.setArguments( arguments )
        job.setExtraFileList( extrafilelist )
        job.setTimeout( timeout )

        # creates job directory name
        job.setJobDirectory(
            os.path.join( self.__theTmpDir, str( job.getJobID() ) ) )
        return job.getJobID()

    def registerEcellSession( self, ess, arguments = {},
                              extrafilelist = [], dmpath = "", timeout = 0 ):
        '''registers a new Ecell Session
        ess(str)                   -- ess file name
        arguments                  -- arguments to be set to script
        extrafilelist(list of str) -- list of extra file name
        dmpath(str)            -- set the ECELL3_DM_PATH
        timeout(int)               -- set time out (sec.). When timeout=0, no limit is set.
        return int : job id
        '''

        # creates AbstractSessionProxy
        job = self.__theSystemProxy.createSessionProxy()
        job.setScriptFileName( ess )
        job.setInterpreter( ECELL3_SESSION )
        job.setArguments(
            (
                '--parameters=' + str( arguments ),
                ) )
        job.setEnvironmentVariable( 'ECELL3_DM_PATH', dmpath )
        for i in INTERESTING_ENV_VARS:
            val = os.environ.get( i, None )
            if val != None:
                job.setEnvironmentVariable( i, val )
        job.setExtraFileList( extrafilelist )
        job.setTimeout( timeout )

        job.setJobDirectory(
            os.path.join( self.__theTmpDir, str( job.getJobID() ) ) )

        return job.getJobID()

    def clearSessionProxy( self, jobid ):
        '''Remove the job directory and the AbstractSessionProxy
           related to the jobid.
        jobid(int) -- job id
        return None
        '''
        self.__theSystemProxy.getSessionProxy( jobid ).clear()

    def clearAllSessionProxies( self ):
        for job in list(self.getSessionProxies()):
            job.clear()

    def clearRunningSessionProxies( self ):
        '''remove running jobs
        Return None
        '''
        for job in self.getRunningSessionProxyList():
            job.clear()

    def clearErrorSessionProxies( self ):
        '''remove error jobs
        Return None
        '''
        for job in self.getErrorSessionProxyList():
            job.clear()

    def clearFinishedSessionProxys( self ):
        '''remove finished jobs

        Return None
        '''
        # delete AbstractSessionProxy instance whose status is FINISHED
        for job in self.getFinishedSessionProxyList():
            job.clear()

    def update( self ):
        '''updates all jobs' status
        This method calls merely AbstractSystemProxy's update method.

        Return None
        '''

        # call AbstractSystemProxy's update method
        self.__theSystemProxy.update()

    def getQueuedSessionProxyList( self ):
        '''Return the list of queued job.
        Return list : the list of AbstractSessionProxy
        '''
        # initializes a job list
        return self.__theSystemProxy.getSessionProxiesByStatus( QUEUED )

    def getRunningSessionProxyList( self ):
        '''Return the list of running job. 
        Return list : the list of AbstractSessionProxy
        '''
        return self.__theSystemProxy.getSessionProxiesByStatus( RUN )

    def getErrorSessionProxyList( self ):
        '''Return the list of error job 

        Return list : the list of AbstractSessionProxy
        '''
        return self.__theSystemProxy.getSessionProxiesByStatus( ERROR )

    def getFinishedSessionProxyList( self ):
        '''Return the list of finished job. 

        Return list : list of AbstractSessionProxy
        '''
        return self.__theSystemProxy.getSessionProxiesByStatus( FINISHED )

    def isFinished( self ):
        '''Check all jobs are finished or not.
        When the number of jobs whose status is QUEUED or RUN = 0, return True.

        Return boolean 
        '''

        # If there is job who status is QUEUED or RUN, return False
        for job in self.getSessionProxies():
            if job.getStatus() == QUEUED or\
               job.getStatus() == RUN:
                return False

        # If not, return True.
        return True

    def isError( self ):
        '''Check the existance of error job.
        When the number of jobs whose status is ERROR > 0, return True.

        Return boolean 
        '''

        # If there is job who status is ERROR, return True.
        for job in self.getSessionProxies():
            if job.getStatus() == ERROR:
                return True

        # If not, return False.
        return False

    def isRunning( self ):
        '''Check the existance of running job.
        When the number of jobs who status is RUN > 0, return True.

        Return boolean 
        '''
        # If there is job who status is RUN, return True.
        for job in self.getSessionProxies():
            if job.getStatus() == RUN:
                return True

        # If not, return Fal.
        return False

    def setUpdateInterval( self, interval=1 ):
        '''Set an update interval.
        When the argument block is True, update method of AbstractSystemProxy
        is called in the update interval time.
        When the argument block is False, the update interval time
        is not used.

        interval(int) -- an interval time (sec.). default is 1.
        Return None
        '''

        # save interval to instance attribute
        self.__theUpdateInterval = interval

    def getUpdateInterval( self ):
        '''Return the update interval.
        Update interval is used in run method.
        See setUpdateInterval about the detail of update interval

        Return int : interval time (sec.)
        '''

        # return the update interval
        return self.__theUpdateInterval 

    def setGlobalRunTimeout( self, timeout=0 ):
        '''Set a timeout of run method.
        When the time of run method reach the timeout,
        stop method is called. Then running jobs and queued jobs 
        are finished compulsorily and their status become ERROR.

        timeout(int) -- a timeout of run method (sec.) 
                        if timeout<=0, no limit is set. Default is 0.
        Return None
        '''

        # save timeout to instance attribute
        self.__theGlobalRunTimeout = timeout

    def getGlobalRunTimeout( self ):
        '''Return the timeout of run method.
        See setGlobalRunTimeout about the detail of timeout

        Return int : the timeout of run method (sec.)
        '''

        # return the timeout
        return self.__theGlobalRunTimeout 

    def run( self, block = True ):
        '''Execute the QUEUED jobs.
        The RUNNING or FINISHED jobs are not executed by this method.

        When block is True, this method waits until all jobs are finished
        by calling poll(). If global timeout value is set to a value
        greater than 0, every running job is aborted on timeout.

        When block is False, you have to call update() or poll() by yourself.
        If these aren't called, neither the statuses of running jobs
        are updated, nor queued jobs are executed.

        block(boolean) -- True: This method waits until all jobs are finished.
                          False: This method does not wait.
        Return : None
        '''
        # create a tmp root dir
        if not os.path.exists( self.__theTmpRootDir ):
            os.mkdir( self.__theTmpRootDir)


        # create a tmp directory.
        if not os.path.exists( self.__theTmpDir):
            os.mkdir( self.__theTmpDir )
        

        # create job directories and copies sources
        for job in self.getSessionProxies():
            if job.getStatus() == FINISHED:
                continue
            # create job directory name
            aJobDirectoryName = os.path.join(
                self.__theTmpDir, str( job.getJobID() ) )

            if not os.path.exists( aJobDirectoryName ):
                os.mkdir(aJobDirectoryName)

            # create dist filename
            aDstFileName = os.path.join(
                aJobDirectoryName, os.path.basename( job.getScriptFileName()) )

            # copie script
            shutil.copyfile( job.getScriptFileName(), aDstFileName )

            # sets up extra files
            for anExtraFile in job.getExtraFileList():

                # creates dist filename
                aDstFileName = os.path.join(
                    aJobDirectoryName, ( os.path.basename(anExtraFile) ) )

                # copy a directory
                if os.path.isdir( anExtraFile ):
                    shutil.copytree( anExtraFile, aDstFileName )

                # copy a file
                else:
                    shutil.copyfile( anExtraFile ,aDstFileName )
        if block == True:
            if not self.poll( self.__theGlobalRunTimeout ):
                self.stopRunningSessionProxies()
                return False
        return True

    def poll( self, timeout = 0):
        '''
        Waits for all the job to be finished and returns True.
        If timeout is specified and the time runs out, immediately returns
        False.
        '''
        abortionEvent = threading.Event()
        mutex = threading.Lock()
        emitted_signum = []
        def signal_handler( signum, ctx ):
            mutex.acquire()
            emitted_signum.append( signum )
            try:
                abortionEvent.set()
            finally:
                mutex.release()
        def poller():
            while not self.isFinished() and not abortionEvent.isSet():
                # updates status of all AbstractSessionProxy
                mutex.acquire()
                try:
                    self.update()
                finally:
                    mutex.release()
                # sleeps for update interval
                abortionEvent.wait( self.__theUpdateInterval )
                pass
        prev_handler = {}
        t = None
        try:
            for signum in ( signal.SIGINT, signal.SIGTERM ):
                prev_handler[ signum ] = signal.signal(
                    signum, signal_handler )
            t = threading.Thread( target = poller )
            t.start()
            if timeout:
                try:
                    t.join( timeout )
                except:
                    pass
                if t.isAlive():
                    abortionEvent.set()
                    return False
            else:
                t.join()
            return True
        finally:
            if t:
                del t
            del signal_handler
            del poller
            for signum, hdlr in prev_handler.iteritems():
                signal.signal( signum, hdlr )
            del prev_handler
            if emitted_signum:
                signal.getsignal( emitted_signum[ 0 ] )( emitted_signum,
                    inspect.currentframe() )

    def stop( self, jobid = 0 ):
        '''stop jobs.
        Change the statuses of QUEUED or RUNNING jobs to ERROR.
        About FINISHED jobs, do nothing.

        jobid(int) -- job id
                      When jobid<=0, stop all jobs
        Return None
        '''

        # stops all jobs
        if jobid == 0:
            for job in self.__theSystemProxy.jobs.values():
                job.stop()

        # stops one job
        else:
            self.__theSystemProxy.jobs[jobid].stop()

    def stopRunningSessionProxies( self ):
        '''Stop running jobs.
        Change all statuses of RUNNING jobs to ERROR.

        Return None
        '''

        for job in self.getRunningSessionProxyList():
            job.stop()

    def getSystemProxy( self ):
        return self.__theSystemProxy

    def restoreMessageMethod( self ):
        self.__theMessageMethod = self.__plainMessageMethod

    def setMessageMethod( self, aMethod ):
        self.__theMessageMethod = aMethod

    def message( self, message ):
        self.__theMessageMethod( message )

    def __plainMessageMethod( message ):
        print message
    __plainMessageMethod = staticmethod( __plainMessageMethod )

if __name__ == "__main__":
    pass
