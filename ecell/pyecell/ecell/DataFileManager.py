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
# Design: Koichi Takahashi <shafi@e-cell.org>
# Design and Programming: Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from ecell.DataFile import DataFile

__all__ = (
    'DataFileManager'
    )

class DataFileManager:
    """
    DataFileManager (This is abstract class)
      - manages some DataFile objects
    """

    def __init__(self):
        """
        Constructor
        
        return  -> None
        """
        self.theFileMap = {}
        self.theRootDirectory = '.'

    def getFileMap(self):
        """
        getFileMap()
        
        return -> None
        """
        return self.theFileMap

    def setRootDirectory( self, aRootDirectory ):
        """
        setRootDirectory()
        
        aRootDirectory : directory to save files
        
        return -> None
        """
        self.theRootDirectory = aRootDirectory

    def getRootDirectory( self ):
        """
        getRootDirectory()
        
        return -> aRootDirectory(string) 
        """
        return self.theRootDirectory 

    def saveAll(self):
        """
        saveAll()
        
        return -> None
        """
        for aKey in self.theFileMap.keys():
            aFileName = self.theFileMap[aKey].getFileName()
            aFileName = split(aFileName,'/')[-1]
            aFileName = self.theRootDirectory + '/' + aFileName
            self.theFileMap[aKey].save(aFileName)

    def loadAll(self):
        """
        loadAll()
        
        return -> None
        """
        for aKey in self.theFileMap.keys():
            self.theFileMap[aKey].load()

if __name__ == "__main__":
    from ECDDataFile import *
    ecdFile = ECDDataFile()
    ecdFile.setFileName('hoge')

    dm = DataFileManager()
    dm.getFileMap()['file'] = ecdFile 
    dm.loadAll()

    dm.getFileMap()['file'].setFileName('hoge1')
    dm.saveAll()
