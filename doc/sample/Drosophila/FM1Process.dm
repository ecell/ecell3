CLASSNAME = 'FM1Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','vs',0.0), ('Real','KI',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getVariable()->getConcentration() );
  Real V( vs * KI );
  V /= KI + (E * E * E);
  V *= 1E-018 * N_A;
  setFlux( V );
''' )
