
CLASSNAME = 'DegradeClb5Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0)]

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
  const Real E1( C0.getVariable()->getConcentration() );
  Real v( k1 + k2 * E1 );
  Real V( v * S );

  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

