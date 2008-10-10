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
#'Design: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>'
#
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from ecell.ecssupport import *
from ConfirmWindow import *
import string
import re

regexForSplitCamelcasedName = re.compile(r'^[a-z]+|[A-Z]+$|[A-Z][a-z]+|(?:[A-Z]*[A-Z](?=[A-Z][a-z]))|[A-Z]')

def decodeAttribute(anAttribute):
    """
    decodeAttribute
    
    anAttribute : an attribute ( TRUE or FALSE )
    
    return -> '+' or '-'
    """
    if anAttribute == TRUE:
        return '+'
    else:
        return '-'

def convertStringToTuple(aString):
    """
    convertStringToTuple
    
    aString : a string as below
              "(a,b,c)"
    
    return -> '+' or '-'
    """
    aString = aString[1:-1]
    aList = string.split(aString,',')

    for anIndex in range(0,len(aList)):
        anElement = aList[anIndex]
        anElement = string.strip(anElement)
        try:
            anElement = string.atoi(anElement)
        except:
            try:
                anElement = string.atof(anElement)
            except:
                anElement = anElement[1:-1]

        aList[anIndex] = anElement

    return tuple(aList)

def shortenString( aValue, aNumber):
    """
    shortenString
    
    aValue : an original string
    aNumber : the length to cut original string
    
    return -> a shorten string
    """
    if len( str(aValue) ) > aNumber:
        return aValue[:aNumber] + '...'
    else:
        return aValue
     
def retrieveValueFromListStore( aListStore, anIndex, aColumnIndex = 0 ):
    anIter = aListStore.iter_nth_child( None, anIndex )
    if anIter == None:
        return None
    return aListStore.get_value( anIter, aColumnIndex )

def splitCamelcasedName( name ):
    return regexForSplitCamelcasedName.findall( name )

def showPopupMessage( aMode, aMessage, aTitle = 'Confirm' ):
    aDialog = ConfirmWindow( aMode, aMessage, aTitle )
    aDialog.show_all()
    aResponseCode = aDialog.run()
    aDialog.destroy()
    return aResponseCode
