CLASSNAME = 'BUDReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k',0.0),('Real','e',0.0),]

PROTECTED_AUX = '''
  Reactant C0;
  Reactant C1;
  Reactant C2;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
  C2 = getReactant( "C2" );
''' )

defineMethod( 'react', '''
  Real E1 = C0.getSubstance()->getConcentration();
  Real E2 = C1.getSubstance()->getConcentration();
  Real E3 = C2.getSubstance()->getConcentration();
  Real V = E1 + E2 + e*E3;
  V *= k;
  V *= getSuperSystem()->getVolume() * N_A;

  process(V);
''' )
