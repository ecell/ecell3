
CLASSNAME = 'Sic1Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  const Real E1( C0.getVariable()->getConcentration() );
  Real V( k1 + k2 * E1 );
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

