
CLASSNAME = 'MassCalculateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','m',0.0)]

PROTECTED_AUX = '''
  Connection C0;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
''' )

defineMethod( 'process', '''
  Real V( m );
  const Real E( C0.getVariable()->getConcentration() );
  V = m * E * getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

