
CLASSNAME = 'MassCalculateReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','m',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real V( m );
  const Real E( C0.getSubstance()->getConcentration() );
  V = m * E * getSuperSystem()->getVolume() * N_A;

  process ( V );
''' )

