
CLASSNAME = 'DegradeProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','kd',0.0)]

PROTECTED_AUX = '''
  Connection S0;
'''

defineMethod( 'initialize', '''
  S0 = getConnection( "S0" );
''' )

defineMethod( 'react', '''
  Real S( S0.getVariable()->getConcentration() );
  Real V( kd * S * getSuperSystem()->getVolume() * N_A );

  process( V );
''' )

