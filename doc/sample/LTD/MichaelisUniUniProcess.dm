CLASSNAME = 'MichaelisUniUniProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','Km',0.0), ('Real','Kcat',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  S0 = getVariableReference( "S0" );
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''

  const Real S( S0.getVariable()->getConcentration() );
  const Real E( C0.getVariable()->getValue() );
  Real velocity( (Kcat * E * S /( Km + S)) );
  setFlux( velocity );
''' )
