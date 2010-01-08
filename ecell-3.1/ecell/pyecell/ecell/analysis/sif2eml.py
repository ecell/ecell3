#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
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

import ecell.eml


def make_processes(model, target, executors, w0, J):
    aid = 'Process:/:%s_A' % target
    iid = 'Process:/:%s_I' % target

    model.createEntity('BooleanProcess', aid)
    model.createEntity('BooleanProcess', iid)
    model.setEntityProperty(aid, 'J', ['%g' % J])
    model.setEntityProperty(iid, 'J', ['%g' % J])

    if w0 >= 0:
        model.setEntityProperty(aid, 'w0', ['%g' % w0])
        model.setEntityProperty(iid, 'w0', ['0.0'])
    else: # w0 < 0
        model.setEntityProperty(aid, 'w0', ['0.0'])
        model.setEntityProperty(iid, 'w0', ['%g' % -w0])

    areference = [[target, 'Variable:/:%s' % target, '+1']]
    aexpression = 'w0'
    ireference = [[target, 'Variable:/:%s' % target, '-1']]
    iexpression = 'w0'

    for executor, coef in executors:
        if coef > 0:
            model.setEntityProperty(aid, 'w%s' % executor, ['%g' % coef])
            areference.append([executor, 'Variable:/:%s' % executor, '0'])
            aexpression += ' + w%s * %s.Value' % (executor, executor)
        elif coef < 0:
            model.setEntityProperty(iid, 'w%s' % executor, ['%g' % -coef])
            ireference.append([executor, 'Variable:/:%s' % executor, '0'])
            iexpression += ' + w%s * %s.Value' % (executor, executor)
        
    model.setEntityProperty(aid, 'Expression', [aexpression])
    model.setEntityProperty(aid, 'VariableReferenceList', areference)
    model.setEntityProperty(iid, 'Expression', [iexpression])
    model.setEntityProperty(iid, 'VariableReferenceList', ireference)

def sif2eml(filename, w0=0.5, wdict={'inhibit': -1, 'activate': -1},
            J=0.1, stepper=0):
    model = ecell.eml.Eml()

    if stepper:
        model.createStepper('ODEStepper', 'BS01')
    else:
        model.createStepper('BooleanStepper', 'BS01')

    model.createEntity('System', 'System::/')
    model.setEntityProperty('System::/', 'StepperID', ['BS01'])

    interactions = {}

    inputFile = open(filename, 'r')
    while True:
        line = inputFile.readline()
        if line == '' or line is None:
            break

        executor, interaction, target = line.rstrip().split(' ') # FIX ME

        if executor == '0' or target == '0':
            raise InputError, 'Variable name [0] does not allowed.'

        if not target in interactions.keys():
            interactions[target] = []
        if not executor in interactions.keys():
            interactions[executor] = []

        if interaction in wdict.keys():
            interactions[target].append((executor, wdict[interaction]))
        else:
            raise InputError, \
                'Interaction type [%s] is undefined.' % interaction

    inputFile.close()

    for variable in interactions.keys():
        model.createEntity('Variable', 'Variable:/:%s' % variable)
        model.setEntityProperty('Variable:/:%s' % variable, 'Value', ['0.0'])

        make_processes(model, variable, interactions[variable], w0, J)

    return model


if __name__ == '__main__':

    import sys

    def main(filename):
        model = sif2eml(filename)
        print model.asString()


    main(sys.argv[1])
