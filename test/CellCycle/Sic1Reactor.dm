
CLASSNAME = 'Sic1Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getSubstance()->getConcentration() );
  Real V( k1 + k2 * E1 );
  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

