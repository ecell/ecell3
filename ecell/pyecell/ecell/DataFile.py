#!/usr/bin/env python
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
# Design: Koichi Takahashi <shafi@e-cell.org>
# Design and Programming: Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

# ------------------------------------------------------------------
# DataFile (This is abstract class)
#   - manages one file object
#   - has file name
# ------------------------------------------------------------------
class DataFile:

    # ------------------------------------------------------------------
    # Constructor
    #
    # return  -> None
    # ------------------------------------------------------------------
    def __init__(self):

        self.theFileName = ''

    # end of __init__


    # ------------------------------------------------------------------
    # setFileName
    #
    # aFileName(string)  : a file name 
    #
    # return -> None
    # ------------------------------------------------------------------
    def setFileName(self, aFileName):

        if( type(aFileName) != type('') ):
            raise TypeError("Error : aFileName is not list matrix.")

        if( len(aFileName) == 0 ):
            print 'Warning: %s, the length of filename is 0' %__name__

        self.theFileName = aFileName

    # end of setFileName


    # ------------------------------------------------------------------
    # getFileName
    #
    # return -> the file name (string)
    # ------------------------------------------------------------------
    def getFileName(self):

        return self.theFileName

    # end of theFileName


    # ------------------------------------------------------------------
    # save ( abstract )
    #
    # If this method doesn't be implemented in sub class, 
    # then throws NotImplementedError
    # ------------------------------------------------------------------
    def save(self):

        import inspect
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + 'must be implemented in subclass')

    # end of save


    # ------------------------------------------------------------------
    # load
    #
    # If this method doesn't be implemented in sub class, 
    # then throws NotImplementedError
    # ------------------------------------------------------------------
    def load(self):

        import inspect
        caller = inspect.getouterframes(inspect.currentframe())[0][3]
        raise NotImplementedError(caller + 'must be implemented in subclass')

    # end of load


# end of DataFile

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

    def main():
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

