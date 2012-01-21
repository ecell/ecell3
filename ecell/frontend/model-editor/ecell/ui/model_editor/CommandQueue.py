#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
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

class CommandQueue:
    def __init__(self, aLength ):
        self.__theLength = aLength
        self.__theCommandQueue = []
        self.__thePointer = 0

    def push( self, aCommandList):
        if self.isNext():
            self.__deleteForward()
        self.__theCommandQueue.append( aCommandList )
        self.__thePointer += 1
        if len(self.__theCommandQueue) > self.__theLength:
            self.__theCommandQueue.__delitem__(0)
            self.__thePointer -= 1

    def moveback( self ):
        if self.isPrevious():
            self.__thePointer-=1
            return self.__theCommandQueue[ self.__thePointer ]
        raise Exception("No way to move back back in CommandQueue!")

    def moveforward( self):
        if self.isNext():
            self.__thePointer+=1
            return self.__theCommandQueue[ self.__thePointer -1 ]
        raise Exception("No way to move forward in CommandQueue")

    def __deleteForward( self ):
        while self.isNext():
            self.__theCommandQueue.pop()

    def isPrevious( self ):
        return self.__thePointer > 0

    def isNext( self ):
        return self.__thePointer < len( self.__theCommandQueue)

    


