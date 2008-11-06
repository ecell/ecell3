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
import socket
from StringIO import StringIO
from tempfile import mkstemp
import xml.sax as sax
from xml.sax.saxutils import XMLGenerator
import xml.sax.handler as saxhandler
from urlparse import urlunsplit

# imports ecell modules
from ecell.SessionManager.Util import *
from ecell.SessionManager.SessionManager import *
from ecell.SessionManager.Constants import *

GLOBUSRUN_WS = 'globusrun-ws'
GLOBUS_URL_COPY='globus-url-copy'
DEFAULT_FACTORY_ENDPOINT_URI = 'https://localhost:8443/wsrf/services/ManagedJobFactoryService'
MKDIR='/bin/mkdir'
RM='/bin/rm'

class JobDescription( object ):
    def __init__( self ):
        self._arguments = tuple()
        self._environment = {}
        self._filesCleanedUp = tuple()
        self._filesStagedIn = tuple()
        self._filesStagedOut = tuple()
        self.count = 1
        self.directory = None
        self.executable = None
        self.factoryEndpoint = None
        self.libraryPath = None
        self.stdin = None
        self.stderr = None
        self.stdout = None
        self.project = None
        self.queue = None
        self.minMemory = None
        self.maxMemory = None
        self.maxTime = None
        self.maxWallTime = None
        self.maxCpuTime = None

    arguments = property(
        lambda self: self._arguments,
        lambda self, value: setattr( self, '_arguments', tuple( value ) )
        )
    environment = property( lambda self: self._environment )
    filesCleanedUp = property(
        lambda self: self._filesCleanedUp,
        lambda self, value: setattr( self, '_filesCleanedUp', tuple( value ) )
        )
    filesStagedIn = property(
        lambda self: self._filesStagedIn,
        lambda self, value: setattr( self, '_filesStagedIn', tuple( value ) )
        )
    filesStagedOut = property(
        lambda self: self._filesStagedOut,
        lambda self, value: setattr( self, '_filesStagedOut', tuple( value ) )
        )

class Namespaces( object ):
    j = u'http://www.globus.org/namespaces/2004/10/gram/job'
    jd = u'http://www.globus.org/namespaces/2004/10/gram/job/description'
    wsa = u'http://schemas.xmlsoap.org/ws/2004/03/addressing'
    rft = u'http://www.globus.org/namespaces/2004/10/rft'

    def __metaclass__( name, base, dic ):
        dic[ '__rev__' ] = dict( (
            ( uri, prefix )
                for prefix, uri in dic.iteritems()
                if not prefix.startswith( '__' ) ) )
        return type( name, base, dic )

class EPRParser( object ):
    class Helper( object, saxhandler.ContentHandler ):
        def __init__( self, result ):
            self.result = result
            self.state = 0
            self.nestingLevel = 0
            self.capture = None

        def raiseError( self, descr ):
            raise RuntimeError( descr )
            

        def unexpectedElement( self, name, qname ):
            self.raiseError(
                u'Unexpected element <%s> (where %s specifies %s)' % (
                    qname, qname.split( ':' )[ 0 ], name[ 0 ] ) )

        def startElementNS( self, name, qname, attrs ):
            decodedNS = Namespaces.__rev__.get( name[ 0 ], None )
            if self.state == 0:
                if decodedNS == 'wsa' and name[ 1 ] == 'EndpointReferenceType':
                    self.state = 1
                else:
                    self.unexpectedElement( name, qname )
            elif self.state == 1:
                if decodedNS == 'wsa':
                    if name[ 1 ] == 'Address':
                        self.state = 2
                    elif name[ 1 ] == 'ReferenceProperties':
                        self.state = 3
                        self.nestingLevel = 0
                        self.capture = None
                    elif name[ 1 ] == 'ReferenceParameters':
                        self.state = 4
                        self.nestingLevel = 0
                        self.capture = None
                    elif name[ 1 ] == 'PortType':
                        self.state = 5
                    elif name[ 1 ] == 'ServiceName':
                        self.state = 6
                    elif name[ 1 ] == 'Policy':
                        self.state = 7
                else:
                    self.unexpectedElement( name, qname )
            elif self.state == 3:
                self.nestingLevel += 1
                if self.nestingLevel == 1 and \
                   decodedNS == 'j' and name[ 1 ] == 'ResourceID':
                    self.capture = 'resourceID' 
                else:
                    self.capture = None
            elif self.state == 4:
                self.nestingLevel += 1
            elif self.state == 7:
                self.nestingLevel += 1
            elif self.state == 8:
                if decodedNS == 'wsa':
                    if name[ 1 ] == 'ReferenceProperties':
                        self.state = 3
                    elif name[ 1 ] == 'ReferenceParameters':
                        self.state = 4
                    elif name[ 1 ] == 'PortType':
                        self.state = 5
                    elif name[ 1 ] == 'ServiceName':
                        self.state = 6
                    elif name[ 1 ] == 'Policy':
                        self.state = 7
                else:
                    self.unexpectedElement( name, qname )
            elif self.state == 9:
                if decodedNS == 'wsa':
                    if name[ 1 ] == 'ReferenceParameters':
                        self.state = 4
                    elif name[ 1 ] == 'PortType':
                        self.state = 5
                    elif name[ 1 ] == 'ServiceName':
                        self.state = 6
                    elif name[ 1 ] == 'Policy':
                        self.state = 7
                else:
                    self.unexpectedElement( name, qname )
            elif self.state == 10:
                if decodedNS == 'wsa':
                    if name[ 1 ] == 'PortType':
                        self.state = 5
                    elif name[ 1 ] == 'ServiceName':
                        self.state = 6
                    elif name[ 1 ] == 'Policy':
                        self.state = 7
                else:
                    self.unexpectedElement( name, qname )
            elif self.state == 11:
                if decodedNS == 'wsa':
                    if name[ 1 ] == 'ServiceName':
                        self.state = 6
                    elif name[ 1 ] == 'Policy':
                        self.state = 7
                else:
                    self.unexpectedElement( name, qname )
            elif self.state == 12:
                if decodedNS == 'wsa':
                    if name[ 1 ] == 'Policy':
                        self.state = 7
                else:
                    self.unexpectedElement( name, qname )
            else:
                self.unexpectedElement( name, qname )

        def characters( self, cdata ):
            if self.state == 2:
                self.result[ 'uri' ] = cdata
            elif self.state == 3:
                if self.capture != None:
                    self.result[ self.capture ] = cdata
            elif self.state == 4 or self.state == 7:
                pass
            elif self.state == 5:
                self.result[ 'portType' ] = cdata
            elif self.state == 6:
                self.result[ 'serviceName' ] = cdata
            else:
                self.raiseError( 'Unexpected character data' )

        def endElementNS( self, name, qname ):
            if self.state == 2:
                self.state = 8
            elif self.state == 3:
                self.nestingLevel -= 1
                if self.nestingLevel < 0:
                    self.state = 9
            elif self.state == 4:
                self.nestingLevel -= 1
                if self.nestingLevel < 0:
                    self.state = 10
            elif self.state == 7:
                self.nestingLevel -= 1
                if self.nestingLevel < 0:
                    self.state = 13
            elif self.state == 5:
                self.state = 11
            elif self.state == 6:
                self.state = 12
            elif self.state == 8:
                self.state = 0
            elif self.state == 9:
                self.state = 0
            elif self.state == 10:
                self.state = 0
            elif self.state == 11:
                self.state = 0
            elif self.state == 12:
                self.state = 0
            elif self.state == 13:
                self.state = 0
            elif self.state == 14:
                self.state = 0

        def endDocument( self ):
            if self.state != 0:
                raise RuntimeError( 'Unexpected end of document' )

    def __init__( self ):
        pass

    def parse( self, src ):
        reader = sax.make_parser()
        retval = {}
        reader.setFeature( saxhandler.feature_namespaces, True )
        reader.setContentHandler( self.Helper( retval ) )
        reader.parse( src )
        return retval

class JobDescriptionSerializer( object ):
    class Builder( Namespaces, object ):
        from xml.sax.xmlreader import AttributesNSImpl as Attrs
        EmptyAttrs = Attrs( {}, None )

        def __init__( self, generator ):
            self.__generator = generator
            self.__elementStack = []

        def buildExecutable( self, executable ):
            self.startElement( self.jd, u'executable' )
            self.characters( executable )
            self.endElement()

        def buildArguments( self, arguments ):
            if len( arguments ) == 0:
                return
            for arg in arguments:
                self.startElement( self.jd, u'argument' )
                self.characters( arg )
                self.endElement()

        def buildDirectory( self, directory ):
            if directory == None:
                return
            self.startElement( self.jd, u'directory' )
            self.characters( directory )
            self.endElement()

        def buildStdin( self, stdin ):
            if stdin == None:
                return
            self.startElement( self.jd, u'stdin' )
            self.characters( stdin )
            self.endElement()

        def buildStdout( self, stdout ):
            if stdout == None:
                return
            self.startElement( self.jd, u'stdout' )
            self.characters( stdout )
            self.endElement()

        def buildStderr( self, stderr ):
            if stderr == None:
                return
            self.startElement( self.jd, u'stderr' )
            self.characters( stderr )
            self.endElement()

        def buildCount( self, count ):
            if count == None:
                return
            self.startElement( self.jd, u'count' )
            self.characters( unicode( count ) )
            self.endElement()

        def buildLibraryPath( self, libraryPath ):
            if libraryPath == None or len( libraryPath ) == 0:
                return
            for elem in libraryPath:
                self.startElement( self.jd, u'libraryPath' )
                self.characters( elem )
                self.endElement()

        def buildEnvironment( self, environment ):
            if len( environment ) == 0:
                return
            for name, value in environment.iteritems():
                self.startElement( self.jd, u'environment' )
                self.startElement( self.jd, u'name' )
                self.characters( name )
                self.endElement()
                self.startElement( self.jd, u'value' )
                self.characters( value )
                self.endElement()
                self.endElement()

        def buildFactoryEndpoint( self, endpoint ):
            if endpoint == None:
                return
            self.startElement( self.jd, u'factoryEndpoint' )
            self.startElement( self.wsa, u'Address' )
            self.characters( endpoint[ 'uri' ] )
            self.endElement()
            if endpoint.has_key( u'resourceID' ):
                self.startElement( self.wsa, u'ReferenceProperties' )
                self.startElement( self.j, u'ResourceID' )
                self.characters( endpoint[ 'resourceID' ] )
                self.endElement()
                self.endElement()
            if endpoint.has_key( u'portType' ):
                self.startElement( self.wsa, u'PortType' )
                self.characters( endpoint[ 'portType' ] )
                self.endElement()
            if endpoint.has_key( u'serviceName' ):
                self.startElement( self.wsa, u'ServiceName' )
                self.characters( endpoint[ 'serviceName' ] )
                self.endElement()
            self.endElement()

        def buildFileCleanup( self, files ):
            if len( files ) == 0:
                return
            self.startElement( self.jd, u'fileCleanUp' )
            for file in files:
                self.startElement( self.rft, u'deletion' )
                self.startElement( self.rft, u'file' )
                self.characters( file )
                self.endElement()
                self.endElement()
            self.endElement()

        def buildFileStageIn( self, files ):
            if len( files ) == 0:
                return
            self.startElement( self.jd, u'fileStageIn' )
            for destUrl, srcUrl in files:
                self.startElement( self.rft, u'transfer' )
                self.startElement( self.rft, u'sourceUrl' )
                self.characters( srcUrl )
                self.endElement()
                self.startElement( self.rft, u'destinationUrl' )
                self.characters( destUrl )
                self.endElement()
                self.endElement()
            self.endElement()

        def buildFileStageOut( self, files ):
            if len( files ) == 0:
                return
            self.startElement( self.jd, u'fileStageOut' )
            for destUrl, srcUrl in files:
                self.startElement( self.rft, u'transfer' )
                self.startElement( self.rft, u'sourceUrl' )
                self.characters( srcUrl )
                self.endElement()
                self.startElement( self.rft, u'destinationUrl' )
                self.characters( destUrl )
                self.endElement()
                self.endElement()
            self.endElement()

        def buildProject( self, project ):
            if project == None:
                return
            self.startElement( self.jd, u'project' )
            self.characters( unicode( project ) )
            self.endElement()

        def buildQueue( self, queue ):
            if queue == None:
                return
            self.startElement( self.jd, u'queue' )
            self.characters( unicode( queue ) )
            self.endElement()

        def buildMaxTime( self, maxTime ):
            if maxTime == None:
                return
            self.startElement( self.jd, u'maxTime' )
            self.characters( unicode( maxTime ) )
            self.endElement()

        def buildMaxWallTime( self, maxWallTime ):
            if maxWallTime == None:
                return
            self.startElement( self.jd, u'maxWallTime' )
            self.characters( unicode( maxWallTime ) )
            self.endElement()

        def buildMaxCpuTime( self, maxCpuTime ):
            if maxCpuTime == None:
                return
            self.startElement( self.jd, u'maxCpuTime' )
            self.characters( unicode( maxCpuTime ) )
            self.endElement()

        def buildMaxMemory( self, maxMemory ):
            if maxMemory == None:
                return
            self.startElement( self.jd, u'maxMemory' )
            self.characters( unicode( maxMemory ) )
            self.endElement()

        def buildMinMemory( self, minMemory ):
            if minMemory == None:
                return
            self.startElement( self.jd, u'minMemory' )
            self.characters( unicode( minMemory ) )
            self.endElement()

        def build( self, jobDescription ):
            self.startElement( self.jd, u'job' )
            self.buildFactoryEndpoint( jobDescription.factoryEndpoint )
            self.buildExecutable( jobDescription.executable )
            self.buildDirectory( jobDescription.directory )
            self.buildArguments( jobDescription.arguments )
            self.buildEnvironment( jobDescription.environment )
            self.buildStdin( jobDescription.stdin )
            self.buildStdout( jobDescription.stdout )
            self.buildStderr( jobDescription.stderr )
            self.buildCount( jobDescription.count )
            self.buildLibraryPath( jobDescription.libraryPath )
            self.buildProject( jobDescription.project )
            self.buildQueue( jobDescription.queue )
            self.buildMaxTime( jobDescription.maxTime )
            self.buildMaxWallTime( jobDescription.maxWallTime )
            self.buildMaxCpuTime( jobDescription.maxCpuTime )
            self.buildMaxMemory( jobDescription.maxMemory )
            self.buildMinMemory( jobDescription.minMemory )
            self.buildFileStageIn( jobDescription.filesStagedIn )
            self.buildFileStageOut( jobDescription.filesStagedOut )
            self.buildFileCleanup( jobDescription.filesCleanedUp )
            self.endElement()

        def startElement( self, ns, name, attrs = EmptyAttrs ):
            t = ( ns, name )
            self.__generator.startElementNS( t, None, attrs )
            self.__elementStack.append( t )

        def endElement( self ):
            self.__generator.endElementNS( self.__elementStack.pop(), None )

        def characters( self, cdata ):
            self.__generator.characters( cdata )

    def __init__( self ):
        pass

    def serialize( self, generator, jobDescription ):
        for name in dir( Namespaces ):
            if not name.startswith( u'__' ):
                generator.startPrefixMapping(
                    name, getattr( Namespaces, name ) )
        if hasattr( jobDescription, '__iter__'):
            generator.startElementNS(
                ( Namespaces.jd, 'multiJob' ), None, self.Builder.EmptyAttrs )
            for descr in jobDescription:
                self.Builder( generator ).build( descr )
            generator.endElementNS( ( Namespaces.jd, 'multiJob' ), None )
        else:
            self.Builder( generator ).build( jobDescription )
        for name in dir( Namespaces ):
            if not name.startswith( u'__' ):
                generator.endPrefixMapping( name )

class SessionProxy( AbstractSessionProxy ):
    def __init__( self, dispatcher, jobID ):
        AbstractSessionProxy.__init__( self, dispatcher, jobID )
        self.__theEPR = None
        self.__theFactoryEndpoint = None

    def getEPR( self ):
        '''return the filename that stores EPR returned in the preceding
           invocation of globusrun-ws'''
        return self.__theEPR

    def run( self ):
        '''run process
        Return None
        '''
        # check status
        if not AbstractSessionProxy.run( self ):
            return False 

        self.__theFactoryEndpoint = self.getSystemProxy().getFactoryEndpoint()
        buildMyGSIFTPUrl = self.getSystemProxy().buildMyGSIFTPUrl
        absJobDirectory = os.path.abspath( self.getJobDirectory() )

        stagingJob = JobDescription()
        stagingJob.factoryEndpoint = self.__theFactoryEndpoint
        stagingJob.executable = MKDIR
        stagingJob.arguments = ( '-p', absJobDirectory )

        job = JobDescription()
        job.factoryEndpoint = self.__theFactoryEndpoint
        files = [ os.path.normpath(
            os.path.join( absJobDirectory, self.getScriptFileName() ) ) ] 
        for file in self.getExtraFileList():
            files.append( os.path.normpath(
                os.path.join( absJobDirectory, file ) ) )
        job.filesCleanedUp = job.filesStagedIn = (
            'file://' + file for file in files )
        job.stdout =  os.path.normpath(
            os.path.join( absJobDirectory, self.getStdoutFileName() ) )
        job.stderr =  os.path.normpath(
            os.path.join( absJobDirectory, self.getStderrFileName() ) )
        job.directory = absJobDirectory
        job.environment.update( self.getEnvironmentVariables() )
        job.executable = self.getInterpreter()
        job.arguments = [ self.getScriptFileName() ]
        job.arguments += self.getArguments()

        tmpfile = None
        tmpfile = mkstemp()
        try:
            jds = self.getSystemProxy().getJobDescriptionSerializer()
            tmpfile = mkstemp()
            tmpfile = ( os.fdopen( tmpfile[ 0 ], 'w+b' ), tmpfile[ 1 ] )
            gen = XMLGenerator( tmpfile[ 0 ], 'UTF-8' )
            gen.startDocument()
            jds.serialize( gen, ( stagingJob, job ) )
            gen.endDocument()
            tmpfile[ 0 ].close()

            epr = raiseExceptionOnError(
                RuntimeError,
                pollForOutputs(
                    popen2.Popen3(
                        (
                            GLOBUSRUN_WS, '-S', '-f', tmpfile[ 1 ],
                            '-submit', '-batch'
                            ),
                        True
                        )
                    )
                )
            parsedEpr = self.getSystemProxy().getEPRParser().parse(
                StringIO( epr ) )
            eprFile = os.path.join( absJobDirectory,
                '%s.epr' % parsedEpr[ 'resourceID' ] )
            open( eprFile, 'wb' ).write( epr )
            self.__theEPR = eprFile
        finally:
            if tmpfile != None:
                tmpfile[ 0 ].close()
                #os.unlink( tmpfile[ 1 ] )
        return True

    def getFactory( self ):
        '''Return cpu name
        Return str : cpu name
        '''

        return self.__theFactoryEndpoint

    def __cancel( self ):
        raiseExceptionOnError( RuntineError,
            pollForOutputs( popen2.Popen3(
                ( GLOBUSRUN_WS, '-cancel', '-j', self.__theEPR ), True ) )
            )

    def stop( self ):
        '''stop the job
        Return None
        '''

        # When this job is running, stop it.
        if self.getStatus() == RUN:
            self.__cancel()

        # set error status
        self.setStatus( ERROR )

class SystemProxy( AbstractSystemProxy ):
    '''Globus4SystemProxy'''
    def __init__( self, sessionManager ):
        '''Constructor
        sessionManager -- the reference to SessionManager
        '''

        # calls superclass's constructor
        AbstractSystemProxy.__init__( self, sessionManager )

        self.__theIdentity = None
        self.__theHostList = []
        self.__theFactoryEndpoint = None
        self.__theJobDescriptionSerializer = JobDescriptionSerializer()
        self.__theEPRParser = EPRParser()
        self.__theLocalHostName = socket.gethostname()
        self.__defaultResourceType = 'Fork'
        self._updateStatus()

    def getLocalHostName( self ):
        return self.__theLocalHostName

    def setLocalHostName( self, localHostName ):
        self.__theLocalHostName = localHostName

    def setDefaultResourceType( self, type ):
        self.__defaultResourceType = type

    def getDefaultConcurrency( self ):
        return self.__defaultResourceType

    def buildMyGSIFTPUrl( self, path ):
        return self.buildGSIFTPUrl( self.getLocalHostName(), path )

    def buildGSIFTPUrl( hostName, path ):
        return 'gsiftp://%s%s' % ( hostName, path )
    buildGSIFTPUrl = staticmethod( buildGSIFTPUrl )

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

    def getFactoryEndpoint( self ):
        return self.__theFactoryEndpoint

    def setFactoryEndpoint( self, endpoint ):
        if isinstance( endpoint, str ) or isinstance( endpoint, unicode ):
            endpoint = {
                'uri': endpoint,
                'resourceID': self.__defaultResourceType
                }
        if not re.match( r'[a-zA-Z][a-zA-Z0-9-]*://', endpoint[ 'uri' ] ):
            comps = endpoint[ 'uri' ].split( '/', 1 )
            endpoint[ 'uri' ] =  '//' + comps[ 0 ]
            if len( comps ) > 1:
                endpoint[ 'uri' ] += comps[ 1 ]
        endpoint[ 'uri' ] = urlunsplit(
            urlmerge(
                DEFAULT_FACTORY_ENDPOINT_URI,
                endpoint[ 'uri' ] )
                )
        self.__theFactoryEndpoint = endpoint

    def getEPRParser( self ):
        return self.__theEPRParser

    def getJobDescriptionSerializer( self ):
        return self.__theJobDescriptionSerializer

    def _updateStatus( self ):
        '''update jobs's status
        Return None
        '''
        for job in self.getSessionProxies():
            eprFile = job.getEPR()
            if eprFile == None:
                continue
            output = raiseExceptionOnError(
                RuntimeError,
                pollForOutputs(
                    popen2.Popen3(
                        ( GLOBUSRUN_WS, '-status', '-j', eprFile ),
                        True )
                    )
                )
            m = re.match('Current job state: (.+)', output )
            if not m:
                raise RuntimeError( 'Unexpected output: %s' % output.rstrip() )
            # if the status is done, copy remote output files
            # to local machine.
            if m.group( 1 ) == 'Done' and job.getStatus() != FINISHED:
                job.setStatus( FINISHED ) 

