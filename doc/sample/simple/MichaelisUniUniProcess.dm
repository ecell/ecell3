CLASSNAME = 'MichaelisUniUniProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','KmS',0.0), ('Real','KcF',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  S0 = getVariableReference( "S0" );
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  Real velocity( KcF );

  velocity *= C0.getVariable()->getValue();
  const Real S( S0.getVariable()->getConcentration() );
  velocity *= S;
  velocity /= ( KmS + S );

  setFlux( velocity );
''' )
