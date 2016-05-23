#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import numpy

from ecell.DataFile import *
import ecell.TableIO as TableIO

# extension
ECD_EXTENSION='ecd'

# ------------------------------------------------------------------
# ECDDataFile -> DataFile
#  - manages one ECD format file object
# ------------------------------------------------------------------
class ECDDataFile( DataFile ):


    # ------------------------------------------------------------------
    # Constructor
    #
    # return -> None
    # ------------------------------------------------------------------
    def __init__(self, data=None, filename=None):

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

    # end of __init__


    # ------------------------------------------------------------------
    # save ( override the method of DataFile class )
    #
    # return -> None
    # This method is throwable exception.
    # ------------------------------------------------------------------
    def save( self, aFileName = None ):

        if aFileName is not None:
            self.setFileName( aFileName )

        # open the file
        aOutputFile = open(self.theFileName,'w')

        # writes header
        aOutputFile.write(self.getHeaderString())

        aOutputFile.close()
        TableIO.writeArray( self.theFileName, self.theData, 1 )


    # end of save


    # ------------------------------------------------------------------
    # load ( override the method of DataFile class )
    #
    # return -> None
    # This method is throwable exception.
    # ------------------------------------------------------------------
    def load( self, aFileName = None ):

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

        # --------------------------------------------------------
        # [1] read DATA: 
        # --------------------------------------------------------
        self.setDataName( readOneLineData(aInputFile,'#DATA:') )

        # --------------------------------------------------------
        # [2] read SIZE:
        # --------------------------------------------------------
        # ignore SIZE:
        readOneLineData(aInputFile,'#SIZE:')

        # --------------------------------------------------------
        # [3] read LABEL:
        # --------------------------------------------------------
        self.setLabel( readOneLineData(aInputFile,'#LABEL:') )

        # --------------------------------------------------------
        # [4] read NOTE:
        # --------------------------------------------------------
        self.setNote( readOneLineData(aInputFile,'#NOTE:') )

        # --------------------------------------------------------
        # [5] read some lines before matrix data
        # --------------------------------------------------------

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

        # ----------------------------------------------------------
        # [6] reads matrix data
        # ----------------------------------------------------------

         #close the file 
        aInputFile.close()

        self.setData( TableIO.readTableAsArray( self.theFileName, '#' ) )

    # end of load




    # ------------------------------------------------------------------
    # setFileName ( override method of DataFile)
    #
    # aFileName(string) : file name
    #
    # return -> None
    # ------------------------------------------------------------------
    def setFileName( self, aFileName ):

        if aFileName.find( ECD_EXTENSION ) == \
            len( aFileName ) - len( ECD_EXTENSION ) :
            DataFile.setFileName( self, aFileName )
        else:
            DataFile.setFileName( self, aFileName + '.' + ECD_EXTENSION )

    # end of setFileName

    # ------------------------------------------------------------------
    # setDataName 
    #
    # aDataName(string) : a value of DATA
    #
    # return -> None
    # ------------------------------------------------------------------
    def setDataName( self, aDataName ):
        
        self.theDataName = aDataName

    # end of setDataName

    # ------------------------------------------------------------------
    # getDataName 
    #
    # return -> the value of DATA
    # ------------------------------------------------------------------
    def getDataName( self ):
        
        return self.theDataName 

    # end of getDataName


    # ------------------------------------------------------------------
    # setLabel
    #
    # aLabel(string) : a value of LABEL
    #
    # return -> None
    # ------------------------------------------------------------------
    def setLabel( self, aLabel ):
        
        self.theLabel = aLabel.split()

    # end of setLabel


    # ------------------------------------------------------------------
    # getLabel
    #
    # return -> the value of label
    # ------------------------------------------------------------------
    def getLabel( self ):
        
        return self.theLabel

    # end of getLabel


    # ------------------------------------------------------------------
    # setNote
    #
    # aNote(string) : a value of NOTE
    #
    # return -> None
    # ------------------------------------------------------------------
    def setNote( self, aNote ):

        self.theNote = aNote

    # end of setNote


    # ------------------------------------------------------------------
    # getNote
    #
    # return -> the value of NOTE
    # ------------------------------------------------------------------
    def getNote( self ):
        
        return self.theNote
            
    # end of getNote

    # ------------------------------------------------------------------
    # setSizeOfColumn
    #
    # aSizeOfColumn(integer) : a value of first element of SIZE
    #
    # return -> None
    # ------------------------------------------------------------------
    def setSizeOfColumn( self, aSizeOfColumn ):

        if type(aSizeOfColumn) != type(0):
            raise "Error : aSizeOfColumn(=%s) must be integer." %aSizeOfColumn

        self.theSizeOfColumn = aSizeOfColumn

    # end of setSizeOfColumn


    # ------------------------------------------------------------------
    # getSizeOfColumn
    #
    # return -> the value of NOTE
    # ------------------------------------------------------------------
    def getSizeOfColumn( self ):

        return self.theSizeOfColumn 

    # end of getSizeOfColumn


    # ------------------------------------------------------------------
    # setSizeOfLine
    #
    # aSizeOfLine(integer) : a value of second element of SIZE
    #
    # return -> None
    # ------------------------------------------------------------------
    def setSizeOfLine( self, aSizeOfLine ):

        if type(aSizeOfLine) != type(0):
            raise "Error : aSizeOfLine(=%s) must be integer." %aSizeOfLine

        self.theSizeOfLine = aSizeOfLine

    # end of setSizeOfLine


    # ------------------------------------------------------------------
    # getSizeOfLine
    #
    # ------------------------------------------------------------------
    def getSizeOfLine( self ):

        return self.theSizeOfLine 

    # end of getSizeOfLine


    # ------------------------------------------------------------------
    # setSize
    #
    # aSizeOfColumn(string) : a value of first element of SIZE
    # aSizeOfLine(string) : a value of second element of SIZE
    #
    # return -> None
    # ------------------------------------------------------------------
    def setSize( self, aSizeOfColumn, aSizeOfLine ):

        self.setSizeOfColumn( aSizeOfColumn )
        self.setSizeOfLine( aSizeOfLine )

    # end of setSize


    # ------------------------------------------------------------------
    # getSize
    #
    # return -> None
    # ------------------------------------------------------------------
    def getSize( self ):

        return ( self.theSizeOfColumn, self.theSizeOfLine )

    # end of getSize


    # ------------------------------------------------------------------
    # setData
    #   - checks only aData type
    #   - does not check each element type.
    #
    # aData(tuple of tuple) : a matrix data
    #
    # return -> None
    # This method is throwable exception.
    # ------------------------------------------------------------------
    def setData( self, aData ):

        #if type(aData) != type(()):
        #	raise "Error : aData must be tuple of tuple."

        #if len(aData) > 0 :
        #	if type(aData[0]) != type(()):
        #        raise "Error : aData must be tuple of tuple."

        self.theData = aData
        self.setSize( len( aData[0] ), len( aData ) )

    # end of getData

    def getData( self ):
        return self.theData


    # ------------------------------------------------------------------
    # getHeaderList
    #
    # return -> header (list of string)
    # ------------------------------------------------------------------
    def getHeaderList( self ):

        aHeaderList = []        
        aHeaderList.append( '#DATA: %s' %self.theDataName )
        aHeaderList.append( '#SIZE: %d %d' %(self.theSizeOfColumn,self.theSizeOfLine) )
        aHeaderList.append( '#LABEL: %s' % '\t'.join( self.theLabel ) )
        aHeaderList.append( '#NOTE: %s' %self.theNote )
        aHeaderList.append( '#' )
        aHeaderList.append( '#----------------------' )

        return aHeaderList

    # end of getHeaderList


    # ------------------------------------------------------------------
    # getHeaderString
    #
    # return -> header (string)
    # ------------------------------------------------------------------
    def getHeaderString( self ):

        aHeaderList = self.getHeaderList()
        aHeaderString = ''
        for line in aHeaderList:
            aHeaderString = aHeaderString + line + '\n'

        return aHeaderString

    # end of getHeaderString



# end of ECDDataFile


if __name__ == "__main__":

    def main():
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


