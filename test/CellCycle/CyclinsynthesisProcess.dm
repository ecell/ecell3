
CLASSNAME = 'CyclinsynthesisProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0)]

PROTECTED_AUX = '''
  Connection C0;
  Connection C1;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
''' )

defineMethod( 'process', '''
  const Real E1( C0.getVariable()->getConcentration() ); 
  const Real E2( C1.getVariable()->getConcentration() ); 
  Real V( k1 + k2 * E1 );

  V *= E2;
  V *= getSuperSystem()->getVolume() * N_A;

  setFlux( V );
''' )

