CLASSNAME = 'MakesignalPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','add',0.0)]

PROTECTED_AUX ='''
  VariableReference S0;
  VariableReference P0;  
  Int i;
  Int k;
'''

defineMethod( 'initialize','''
  S0 = getVariableReference( "S0" );
  P0 = getVariableReference( "P0" );
  i = 0;
  k = 0;
''')

defineMethod( 'process',
'''
  Real p0( P0.getVariable()->getValue() );

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
  P0.getVariable()->setValue(p0);
''')
