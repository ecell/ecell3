TestDic = {
    'System': {'CELL':('P','M'), 'ENVIRONMENT':'C', 'CHAOS': ('MIYOSHI','NUMATA')}}


EcsTestDic = {
    'System':
    {'CELL':
     {'System':
      {'CYTOPLASM':
       {'System': {'CHAOS': {'Substance':('aa', 'hai', 'B')}, 'Reactor':('Rmt1','R2'), 'Property': {'Volume':1.4e-19}},
        'Substance':('A', 'B', 'C'),
        'Reactor': ('R1', 'R2', 'A'),
        'Property': {'Volume': 1.4e-15}},
       'MEMBRANE':{'Property': {'Potential': 20, 'OsmoPressure': 14},
                   'Reactor':('Transporter1', 'Transporter2')},
       'Property': {'pekin': 'chocobo'}}},
      'ENVIRONMENT': {'Property': {'Volume': 1.4e-12, 'Temparature': 289},
                      'Substance': ('Glucose', 'Na', 'K')}
     }
    }


