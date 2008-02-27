#!/usr/bin/env python
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
#
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import numpy
import TableIO

from ecell.DataFile import DataFile
import ecell.ecs_constants as consts

__all__ = (
    'ECDDataFile'
    )

class ECDDataFile( DataFile ):
    """
    ECDDataFile -> DataFile
     - manages one ECD format file object
    """
    def __init__(self, data=None, filename=None):
        """
        Constructor
        
        return -> None
        """
        DataFile.__init__(self)
        self.theDataName=''
        self.theSizeOfColumn=0
        self.theSizeOfLine=0
        self.theLabel= ( 't', 'value', 'avg', 'min', 'max' )
        self.theNote=''

        if data is None:
            self.setData( numpy.array([[]]) )
        else:
            self.setData( data )

        if filename is not None:
            self.setFileName( filename )

    def save( self, aFileName = None ):
        """
        save ( override the method of DataFile class )
        
        return -> None
        This method may throw an exception.
        """
        if aFileName is not None:
            self.setFileName( aFileName )

        # open the file
        aOutputFile = open(self.theFileName,'w')

        # writes header
        aOutputFile.write(self.getHeaderString())
        aOutputFile.close()
        TableIO.writeArray( self.theFileName, self.theData, 1 )

    def load( self, aFileName = None ):
        """
        load ( override the method of DataFile class )
        
        return -> None
        This method may throw an exception.
        """
        if aFileName is not None:
            self.setFileName( aFileName )

        def readOneLineData( aInputFile, aKey ):
            aBuff = aInputFile.readline() 
            if aBuff.find( aKey ) != 0:
                raise "Error: %s is not ECD format. '%s' line can't be found." %aKey
            return aBuff[len(aKey):].strip()


        if( len(self.theFileName) == 0):
            raise "Error: empty filename."

        # open the file
        aInputFile = open(self.theFileName,'r')

        # read header

        #FIXME: do not depend on the order of header elements

        # [1] read DATA: 
        self.setDataName( readOneLineData(aInputFile,'#DATA:') )

        # [2] read SIZE:
        # ignore SIZE:
        readOneLineData(aInputFile,'#SIZE:')

        # [3] read LABEL:
        self.setLabel( readOneLineData(aInputFile,'#LABEL:') )

        # [4] read NOTE:
        self.setNote( readOneLineData(aInputFile,'#NOTE:') )

        # [5] read some lines before matrix data
        # read matrix
        while(1):   # while 1
            aBuff = aInputFile.readline()

            # if EOF is found, breaks this loop.
            if aBuff == '':
                break    

        # if separator is found, breaks this loop.
            if aBuff.find( '#----------------------' ) == 0:
                break

        # end of while 1

        # [6] reads matrix data
        #close the file 
        aInputFile.close()

        self.setData( TableIO.readTableAsArray( self.theFileName, '#' ) )

    def setFileName( self, aFileName ):
        """
        setFileName ( override method of DataFile)
        
        aFileName(string) : file name
        
        return -> None
        """
        if aFileName.find( consts.ECD_EXTENSION ) == \
            len( aFileName ) - len( consts.ECD_EXTENSION ) :
            DataFile.setFileName( self, aFileName )
        else:
            DataFile.setFileName( self, aFileName + '.' + consts.ECD_EXTENSION )

    def setDataName( self, aDataName ):
        """
        setDataName 
        
        aDataName(string) : a value of DATA
        
        return -> None
        """
        self.theDataName = aDataName

    def getDataName( self ):
        """
        getDataName 
        
        return -> the value of DATA
        """ 
        return self.theDataName 

    def setLabel( self, aLabel ):
        """
        setLabel
        
        aLabel(string) : a value of LABEL
        
        return -> None
        """
        self.theLabel = aLabel.split()

    def getLabel( self ):
        """
        getLabel
        
        return -> the value of label
        """
        return self.theLabel

    def setNote( self, aNote ):
        """
        setNote
        
        aNote(string) : a value of NOTE
        
        return -> None
        """
        self.theNote = aNote

    def getNote( self ):
        """
        getNote
        
        return -> the value of NOTE
        """
        return self.theNote

    def setSizeOfColumn( self, aSizeOfColumn ):
        """
        setSizeOfColumn
        
        aSizeOfColumn(integer) : a value of first element of SIZE
        
        return -> None
        """
        if type(aSizeOfColumn) != type(0):
            raise "Error : aSizeOfColumn(=%s) must be integer." %aSizeOfColumn

        self.theSizeOfColumn = aSizeOfColumn

    def getSizeOfColumn( self ):
        """
        getSizeOfColumn
        
        return -> the value of NOTE
        """
        return self.theSizeOfColumn 

    def setSizeOfLine( self, aSizeOfLine ):
        """
        setSizeOfLine
        
        aSizeOfLine(integer) : a value of second element of SIZE
        
        return -> None
        """
        if type(aSizeOfLine) != type(0):
            raise "Error : aSizeOfLine(=%s) must be integer." %aSizeOfLine

        self.theSizeOfLine = aSizeOfLine

    def getSizeOfLine( self ):
        """
        getSizeOfLine
        """
        return self.theSizeOfLine 

    def setSize( self, aSizeOfColumn, aSizeOfLine ):
        """
        setSize
        
        aSizeOfColumn(string) : a value of first element of SIZE
        aSizeOfLine(string) : a value of second element of SIZE
        
        return -> None
        """
        self.setSizeOfColumn( aSizeOfColumn )
        self.setSizeOfLine( aSizeOfLine )

    def getSize( self ):
        """
        getSize
        
        return -> None
        """
        return ( self.theSizeOfColumn, self.theSizeOfLine )

    def setData( self, aData ):
        """
        setData
          - checks only aData type
          - does not check each element type.
        
        aData(tuple of tuple) : a matrix data
        
        return -> None
        This method may throw an exception.
        """
        self.theData = aData
        self.setSize( len( aData[0] ), len( aData ) )

    def getData( self ):
        return self.theData

    def getHeaderList( self ):
        """
        getHeaderList
        
        return -> header (list of string)
        """
        aHeaderList = []        
        aHeaderList.append( '#DATA: %s' %self.theDataName )
        aHeaderList.append( '#SIZE: %d %d' %(self.theSizeOfColumn,self.theSizeOfLine) )
        aHeaderList.append( '#LABEL: %s' % "\t".join( self.theLabel ) )
        aHeaderList.append( '#NOTE: %s' %self.theNote )
        aHeaderList.append( '#' )
        aHeaderList.append( '#----------------------' )

        return aHeaderList

    def getHeaderString( self ):
        """
        getHeaderString
        
        return -> header (string)
        """
        aHeaderList = self.getHeaderList()
        aHeaderString = ''
        for line in aHeaderList:
            aHeaderString = aHeaderString + line + '\n'

        return aHeaderString

if __name__ == "__main__":
    a = ECDDataFile()
    a.setDataName('name')
    a.setLabel('testlabel')
    a.setNote('testnote')

    aMat = ((3,4),(10,20),(2000,111))

    a.setData( aMat )
    
    a.save('hoge')
    del a
    
    b = ECDDataFile()
    b.load('hoge')
    print b.getHeaderString()
    b.save('hoge1')

    main()


