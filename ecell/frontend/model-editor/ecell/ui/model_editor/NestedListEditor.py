#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2015 Keio University
#       Copyright (C) 2008-2015 RIKEN
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
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

import os
import os.path
import sys

import gtk
import gobject

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ModelEditor import *
from ecell.ui.model_editor.ViewComponent import *

class BadNestedList( Exception ):
    def __init__( self, badString ):
        self.args = "%s\n cannot be parsed as nestedlist!"%badString

class NestedListEditor(ViewComponent):

    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach ):
        self.theParentWindow = aParentWindow
        # call superclass
        ViewComponent.__init__( self,   pointOfAttach, 'attachment_box' )
        self.theNestedList = copyValue( self.theParentWindow.thePropertyValue )
        self.theTextView = self['textview']
        self.textBuffer = gtk.TextBuffer()
        self.theTextView.set_buffer( self.textBuffer )
        self.textBuffer.set_text( self.__nestedListToString( self.theNestedList,0 ) )

    def getValue( self ):
        
        aText = self.textBuffer.get_text( self.textBuffer.get_start_iter(), self.textBuffer.get_end_iter())
        try:
            aValue= self.__stringToNestedList( aText)
        except BadNestedList:
            self.theParentWindow.theModelEditor.printMessage( ''.join(sys.exc_value), ME_ERROR )
            aValue = None
        return aValue


    def __nestedListToString( self, aNestedList, level = 1 ):
        if type(aNestedList ) == type(''):
            return aNestedList
        stringList = []
        
        for aSubList in aNestedList:
            stringList.append( self.__nestedListToString( aSubList ) )
        if level == 0:
            separator = '\n,'
        else:
            separator = ', '
        return '( ' + separator.join( stringList ) + ' )  '



    def __stringToNestedList( self, aString ):
        # should return a nestedlist if string format is OK
        # should return None if string format is not OK, should display an error message in this case.
        aString=aString.strip()
        
        # decide whether list or string
        if aString.__contains__(',') or aString.__contains__('(') or aString.__contains__(')'):
            #must be list
            if not (aString.startswith('(') and aString.endswith(')') ):
                raise BadNestedList( aString )
            stringList = self.__split(aString[1:len(aString)-1].strip())
            parsedList = map( self.__stringToNestedList, stringList )
            if len(parsedList) == 1 and type( parsedList[0]) != type(parsedList ):
                return stringList[0]
            return parsedList

        else:
            return aString  


    def __split( self, aString ):
        openPara = 0
        returnList = []
        actualWord = ''
        for aChar in aString:
            if aChar == ',' and openPara == 0:
                returnList.append( actualWord )
                actualWord = ''
            elif aChar == '(':
                openPara +=1
                actualWord += aChar
            elif aChar == ')':
                openPara -=1
                actualWord += aChar
            else:
                actualWord += aChar
        
        if openPara!=0:
            raise BadNestedList( aString )
        returnList.append( actualWord )
        return returnList
