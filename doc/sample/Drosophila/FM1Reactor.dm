CLASSNAME = 'FM1Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','vs',0.0), ('Real','KI',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getSubstance()->getConcentration() );
  Real V( vs * KI );
  V /= KI + (E * E * E);
  V *= 1E-018 * N_A;
  process( V );
''' )
