
CLASSNAME = 'DegradeReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','kd',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
'''

defineMethod( 'initialize', '''
  S0 = getReactant( "S0" );
''' )

defineMethod( 'react', '''
  Real S( S0.getSubstance()->getConcentration() );
  Real V( kd * S * getSuperSystem()->getVolume() * N_A );

  process( V );
''' )

