CLASSNAME = 'Mcm1calcPReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','J1',0.0),('Real','J2',0.0),]

PROTECTED_AUX = '''
  Reactant C0;
  Reactant P0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  P0 = getReactant( "P0" );
''' )

defineMethod( 'react', '''
  const Real E1 = C0.getSubstance()->getConcentration(); // const?
  Real Va = k1 * E1;
 
  Real a = k2 - Va;
  Real b = a + Va*J2 + k2*J1;
  Real g = Va * J2;

  Real F = 2*g;

  Real x = b*b;
  x -= 4*a*g;
  F /= b + sqrt(x);

  F *= getSuperSystem()->getVolume() * N_A;
  P0.getSubstance()->setQuantity(F);
''' )
