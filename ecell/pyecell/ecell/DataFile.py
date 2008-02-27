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

__all__ = (
    'DataFile'
    )

class DataFile:
    """
    DataFile (This is an abstract class)
      - manages one file object
      - has file name
    """
    def __init__(self):
        """
        Constructor
        
        return  -> None
        """
        self.theFileName = ''

    def setFileName(self, aFileName):
        """
        setFileName
        
        aFileName(string)  : a file name 
        
        return -> None
        """
        if type( aFileName ) != string:
            raise TypeError
        if len(aFileName) == 0:
            print 'Warning: %s, the length of filename is 0' %__name__

        self.theFileName = aFileName

    def getFileName(self):
        """
        getFileName
        
        return -> the file name (string)
        """
        return self.theFileName

    def save(self):
        """
        save ( abstract )
        
        If this method doesn't be implemented in sub class, 
        then throws NotImplementedError
        """
        raise NotImplementedError

    def load(self):
        """
        load
        
        If this method doesn't be implemented in sub class, 
        then throws NotImplementedError
        """
        raise NotImplementedError

if __name__ == "__main__":
    class SubClass1(DataFile):
        def setData(self, aData):
            print "setData"
        def theData(self):
            print "theData"
        def save(self):
            print "save"
        def load(self):
            print "load"

    class SubClass2(DataFile):
        def setData(self, aData):
            print "setData"
        def theData(self):
            print "theData"
        def save(self):
            print "save"
        def load(self):
            print "load"

    sub = SubClass1()
    sub.setData('hoge')
    sub.theData()
    sub.save()
    sub.load()

    sub = SubClass2()
    sub.setData('hoge')
    sub.theData()
    sub.save()
    sub.load()

    file = open('hoge','w')
    file.close()

    main()

