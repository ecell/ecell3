#!/usr/bin/env python2

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from string import *
from DataFile import *

# ------------------------------------------------------------------
# ECDDataFile -> DataFile
#  - manages one ECD format file object
# ------------------------------------------------------------------
class ECDDataFile( DataFile ):

	# extension
	theECDExtension='ecd'

	# ------------------------------------------------------------------
	# Constructor
	#
	# return -> None
	# ------------------------------------------------------------------
	def __init__(self):

		DataFile.__init__(self)
		self.theDataName=''
		self.theSizeOfColumn=0
		self.theSizeOfLine=0
		self.theLabel=''
		self.theNote=''
		self.theMatrixData = [[]]

	# end of __init__


	# ------------------------------------------------------------------
	# setFileName ( override method of DataFile)
	#
	# aFileName(string) : file name
	#
	# return -> None
	# ------------------------------------------------------------------
	def setFileName( self, aFileName ):

		#print ECDDataFile.theECDExtension

		if find(aFileName,ECDDataFile.theECDExtension) == \
			len(aFileName)-len(ECDDataFile.theECDExtension) :
			DataFile.setFileName( self, aFileName )
		else:
			DataFile.setFileName( self, aFileName + '.' + ECDDataFile.theECDExtension )

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
		
		self.theLabel = aLabel

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
	# return -> the value of NOTE
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
	# setMatrixData
	#   - checks only aMatrixData type
	#   - does not check each element type.
	#
	# aMatrixData(tuple of tuple) : a matrix data
	#
	# return -> None
	# This method is throwable exception.
	# ------------------------------------------------------------------
	def setMatrixData( self, aMatrixData ):

		#if type(aMatrixData) != type(()):
		#	raise "Error : aMatrixData must be tuple of tuple."

		#if len(aMatrixData) > 0 :
		#	if type(aMatrixData[0]) != type(()):
		#		raise "Error : aMatrixData must be tuple of tuple."

		self.theMatrixData = aMatrixData
		self.setSize( len(aMatrixData[0]) ,len(aMatrixData) )

	# end of getMatrixData


	# ------------------------------------------------------------------
	# getHeaderList
	#
	# return -> header (list of string)
	# ------------------------------------------------------------------
	def getHeaderList( self ):

		aHeaderList = []		
		aHeaderList.append( 'DATA: %s' %self.theDataName )
		aHeaderList.append( 'SIZE: %d %d' %(self.theSizeOfColumn,self.theSizeOfLine) )
		aHeaderList.append( 'LABEL: %s' %self.theLabel )
		aHeaderList.append( 'NOTE: %s' %self.theNote )
		aHeaderList.append( '' )
		aHeaderList.append( '----------------------' )

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


	# ------------------------------------------------------------------
	# save ( override the method of DataFile class )
	#
	# return -> None
	# This method is throwable exception.
	# ------------------------------------------------------------------
	def save( self, aFileName = None ):

		if aFilename is not None:
			self.saveWithFileName( aFileName )
			return

		# open the file
		aOutputFile = open(self.theFileName,'w')

		# writes header
		aOutputFile.write(self.getHeaderString())

		# writes matrix
		if(self.theSizeOfColumn>=1):
			theSizeOfColumnMinusOne = self.theSizeOfColumn - 1
			for i in xrange(len(self.theMatrixData)):
				j=0
				while j < theSizeOfColumnMinusOne:
					aOutputFile.write("%s\t" %self.theMatrixData[i][j])
					j=j+1
				aOutputFile.write("%s\n" %self.theMatrixData[i][theSizeOfColumnMinusOne])

 		#close the file 
		aOutputFile.close()

	# end of save


	# ------------------------------------------------------------------
	# load ( override the method of DataFile class )
	#
	# return -> None
	# This method is throwable exception.
	# ------------------------------------------------------------------
	def load( self ):

		def readOneLineData( aInputFile, aKey ):
			aBuff = aInputFile.readline() 
			if find(aBuff,aKey) != 0:
				raise "Error: %s is not ECD format. '%s' line can't be found." %aKey
			return strip(aBuff[len(aKey):]) 


		if( len(self.theFileName) == 0):
			raise "Error: the length of filename is 0"

		# open the file
		aInputFile = open(self.theFileName,'r')

		# read header

		# ----------------------------------------------------------------------
		# [1] reads DATA: 
		# ----------------------------------------------------------------------
		self.setDataName( readOneLineData(aInputFile,'DATA:') )

		# ----------------------------------------------------------------------
		# [2] reads SIZE:
		# ----------------------------------------------------------------------
		aSizeList = split(readOneLineData(aInputFile,'SIZE:'),' ')
		if len(aSizeList) != 2:
			raise "Error: %s is not ECD format. 'SIZE:' line needs 2 elements." %self.theFileName
		try:
			self.setSize(atoi(aSizeList[0]),atoi(aSizeList[1]))
		except:
			raise "Error: %s is not ECD format. elements of 'SIZE:' must be integer." %self.theFileName

		# ----------------------------------------------------------------------
		# [3] reads LABEL:
		# ----------------------------------------------------------------------
		self.setLabel( readOneLineData(aInputFile,'LABEL:') )

		# ----------------------------------------------------------------------
		# [4] reads NOTE:
		# ----------------------------------------------------------------------
		self.setNote( readOneLineData(aInputFile,'NOTE:') )

		# ----------------------------------------------------------------------
		# [5] reads some lines before matrix data
		# ----------------------------------------------------------------------

		# read matrix
		while(1):   # while 1
			aBuff = aInputFile.readline()

			# if EOF is found, breaks this loop.
			if aBuff == '':
				break	

			# if separator is found, breaks this loop.
			if find(aBuff,'----------------------' ) == 0:
				break

		# end of while 1

		# ----------------------------------------------------------------------
		# [6] reads matrix data
		# ----------------------------------------------------------------------

		aMatrixList = []

		for aBuff in aInputFile.readlines():
			aDataListOfOneLine = split( strip(aBuff),'\t')
			aDataList = []
			for anElement in aDataListOfOneLine:
				try:
					aDataList.append( atoi(anElement) )
				except:
					try:
						aDataList.append( atof(anElement) )
					except:
						raise "Error: %s is not ECD format. Non-fload data are included." %self.theFileName
			aMatrixList.append( tuple(aDataList) )

		# checks matrix size
		aSizeOfLine = len(aMatrixList)
		aSizeOfColumn = 0
		if aSizeOfLine >= 1 :
			aSizeOfColumn = len(aMatrixList[0])

		if aSizeOfLine != self.getSizeOfLine():
			print "Warnig : %s, the line size (=%s) in 'SIZE:' line does not match that of matrix (=%s), so %s is applied to the line size of matrix" %(self.theFileName,self.getSizeOfLine(),aSizeOfLine,aSizeOfLine)

		if aSizeOfColumn != self.getSizeOfColumn():
			print "Warnig : %s, the column size (=%s) in 'SIZE:' line does not match that of matrix (=%s), so %s is applied to the column size of matrix" %(self.theFileName,self.getSizeOfColumn(),aSizeOfColumn,aSizeOfColumn)

		self.setMatrixData( tuple(aMatrixList) )

 		#close the file 
		aInputFile.close()

	# end of load


# end of ECDDataFile


if __name__ == "__main__":

	def main():
		a = ECDDataFile()
		a.setDataName('name')
		a.setLabel('testlabel')
		a.setNote('testnote')

		aMat = ((3,4),(10,20),(2000,111))

		a.setMatrixData( aMat )
		a.setFileName('hoge')
		#a.save()

		b = ECDDataFile()
		b.setFileName('hoge')
		b.load()
		b.setFileName('hoge1')
		print b.getHeaderString()
		b.save()

	main()


