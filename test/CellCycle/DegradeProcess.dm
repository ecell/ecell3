
CLASSNAME = 'DegradeProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','kd',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
'''

defineMethod( 'initialize', '''
  S0 = getVariableReference( "S0" );
''' )

defineMethod( 'process', '''
  Real S( S0.getVariable()->getConcentration() );
  Real V( kd * S * getSuperSystem()->getVolume() * N_A );

  setFlux( V );
''' )

