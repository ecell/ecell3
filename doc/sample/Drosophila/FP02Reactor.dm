CLASSNAME = 'FP02Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','V1',0.0),('Real','K1',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getSubstance()->getConcentration() );

  Real V( -1 * V1 * E );
  V /= K1 + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
