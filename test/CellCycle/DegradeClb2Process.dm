
CLASSNAME = 'DegradeClb2Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','k3',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
'''

defineMethod( 'initialize', '''
 S0 = getVariableReference( "S0" );
 C0 = getVariableReference( "C0" );
 C1 = getVariableReference( "C1" );
 C2 = getVariableReference( "C2" );
''' )

defineMethod( 'process', '''
  const Real S( S0.getVariable()->getConcentration() );
  const Real E1( C0.getVariable()->getConcentration() );
  const Real E2( C1.getVariable()->getConcentration() );
  const Real E3( C2.getVariable()->getConcentration() );
  Real v( k1 * (E1 - E2) + k2 * E2 + k3 * E3);
  Real V( v * S );
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

