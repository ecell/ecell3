#!/usr/bin/env python
"""TableIO.py
A module for reading ASCII tables into NumPy arrays (and lists). 

Copyright (C) 2000 Michael A. Miller <mmiller@debian.org>
Time-stamp: <2002-03-08 12:13:43 miller>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA.


"""

__version__ = "$Revision$"
# $Source$
   
import numpy
import _tableio
import types

def TableToArray(t):
    a = numpy.array(t['data'])
    a.shape = (t['rows'],t['columns'])
    return a

def readTable(file,commentChars):
    t = _tableio.readTable(file,commentChars)
    return t

def readTableAsArray(file,commentChars):
    t = _tableio.readTable(file,commentChars)
    a = TableToArray(t)
    return a

def readColumns(file, commentChars, columns='all'):
    t = _tableio.readTable(file,commentChars)
    return extractColumns(t, columns)

def extractColumns(t, columns='all'):
    a = numpy.array(t['data'])
    a.shape = (t['rows'],t['columns'])
    if columns == 'all':
        data = []
        for c in range(t['columns']):
            data.append(numpy.ravel(numpy.take(a, [c], 1)))
    else:
        if type(columns) == types.ListType \
           or type(columns) == types.TupleType:
            data = []
            for c in columns:
                data.append(numpy.ravel(numpy.take(a, [c], 1)))
                
    return data

def selectColumns(t, columns):
    a = numpy.array(t['data'])
    a.shape = (t['rows'],t['columns'])
    data = []
    for c in columns:
        data.append(numpy.ravel(numpy.take(a, [c], 1)))        
    return data

def writeArray(file, data, append=0 ):
    t = {}
    t['filename'] = file
    t['data'] = numpy.ravel(data).tolist()
    t['rows'] = data.shape[0]
    t['columns'] = data.shape[1]
    t['append'] = append
    status = writeTable( t )
    return status

def writeTable(t):
    if 'append' in t.keys():
        status = _tableio.writeTable(t['filename'], t['data'], t['rows'], t['columns'],t['append'])
    else:
        status = _tableio.writeTable(t['filename'], t['data'], t['rows'], t['columns'])
    return status

def TableFromColumns(list):
    t = {}
    for l in map(len,list):
        if l != len(list[0]):
            print 'Columns must all have the same length', map(len,list)
    else:
        t['filename'] = ''
        t['rows'] = len(list[0])
        t['columns'] = len(list)
        t['data'] = numpy.ravel(numpy.transpose(numpy.array(list))).tolist()
    return t

def test():
    # Make a table and write it to a file:
    file = 'TableIO.test-data'
    print 'Writing a table to', file
    t = {}
    t['columns'] = 3
    t['rows'] = 5
    t['filename'] = file
    t['data'] = [ 4,  4,  1.3,
                  8,  8,  1.7,
                  9,  9,  1.8,
                  10, 10, 1.9,
                  11, 11, 0.000000001,
                  11, 11, 1e10 ]
    writeTable(t)

    # Read it back in:
    print 'Reading it back as columns'   
    [x,y,z] = readColumns(file, '#!')
    [a, b] = readColumns(file, '#!', [0,2])
    print 'x =', x
    print 'y =', y
    print 'z =', z

    # Creat a table from columns and write it out
    print 'Creating a table from columns and writing it'
    u = TableFromColumns([x,y,z,x*x,y,z-y])
    u['filename'] = file + '2'
    writeTable(u)
    print 'Table is', u

    print 'Read it back as a table:'
    t = readTable(file,'#!')
    print t

    print '... and as an array'
    a = readTableAsArray(file,'!#')
    print type(a), a.shape
    print a

    print 'Writing array to file', file+'3'
    writeArray(file+'3', a)

    print 'Testing comments'
    f = open(file+'4','w')
    f.writelines(['# A comment line\n',
                  '1 2 3 4 5\n',
                  '1 2 3 4 5\n',
                  '# A comment line\n',
                  '1 2 3 4 5\n'])
    f.close()
    t = readTable(file+'4','#')

    f = open(file+'5','w')
    f.writelines(['# A comment line\n',
                  'X Also comment line\n',
                  '! Also comment line\n',
                  '? Also comment line\n',
                  '1 2 3 4 5\n',
                  '1 2 3 4 5\n',
                  '# A comment line\n',
                  '1 2 3 4 5\n'])
    f.close()
    t = readTable(file+'5','#X!?')

    
if __name__ == '__main__':
    test()
