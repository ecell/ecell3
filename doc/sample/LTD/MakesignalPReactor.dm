CLASSNAME = 'MakesignalPReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','add',0.0)]

PROTECTED_AUX ='''
  Reactant S0;
  Reactant P0;
  Int i;
  Int k;
'''

defineMethod('initialize','''
  S0 = getReactant( "S0" );
  P0 = getReactant( "P0" );
  i = 0;
  k = 0;
''')

defineMethod('react',
'''
  Real p0( P0.getSubstance()->getQuantity() );

  i += 1;

  if( k < 30000)
    {
    Int ii( i % 10 );
    if( ii == 0 )
      {
      p0 = add * getSuperSystem()->getVolume() * N_A;
      k = k + 1;
      }
    else
      {
      p0 = add * getSuperSystem()->getVolume() * N_A / (2 * ii);
      }
    }
    else
      {
      p0 = 0.00000000000000000001 * getSuperSystem()->getVolume() * N_A;
      }					
  P0.getSubstance()->setQuantity(p0);
''')
