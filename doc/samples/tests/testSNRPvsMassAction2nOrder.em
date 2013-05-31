Stepper SpatiocyteStepper(SS) { VoxelRadius 6e-9; }
Stepper ODEStepper(DE) { MaxStepInterval 1e-5; }

System System(/)
{
  StepperID       SS;
  Variable Variable(GEOMETRY)
    {
      Value 0;
    }
  Variable Variable(LENGTHX)
    {
      Value 1e-6;
    }
  Variable Variable(LENGTHY)
    {
      Value 1e-6;      
    }
  Variable Variable(LENGTHZ)
    {
      Value 1e-6;     
    }
  Variable Variable(VACANT)
    {
      Value 0;
    }
    Variable Variable( A )
    {
        Name HD;
        Value   10000;
    }
    Variable Variable( B )
    {
        Name HD;
        Value   10000;
    }
    Variable Variable( C )
    {
        Name HD;
        Value  0;  
    }
    Variable Variable( Am )
    {
        Name HD;
        Value   10000;
    }
    Variable Variable( Bm )
    {
        Name HD;
        Value   10000;
    }
    Variable Variable( Cm )
    {
        Name HD;
        Value  0;  
    }
    Variable Variable( At )
    {
        Name HD;
        Value   10000;
    }
    Variable Variable( Bt )
    {
        Name HD;
        Value   10000;
    }
    Variable Variable( Ct )
    {
        Name HD;
        Value  0;  
    }
    Variable Variable( An )
    {
        Name HD;
        Value   1000;
    }
    Variable Variable( Bn )
    {
        Name HD;
        Value   0;
    }
    Variable Variable( Cn )
    {
        Name HD;
        Value  0;  
    }
    Variable Variable( Ans )
    {
        Name HD;
        Value   1000;
    }
    Variable Variable( Bns )
    {
        Name HD;
        Value   0;
    }
    Variable Variable( Cns )
    {
        Name HD;
        Value  0;  
    }
    Variable Variable( Ano )
    {
        Name HD;
        Value   1000;
    }
    Variable Variable( Bno )
    {
        Name HD;
        Value   0;
    }
    Variable Variable( Cno )
    {
        Name HD;
        Value  0;  
    }
   Process SpatiocyteNextReactionProcess( reaction )
    {
      VariableReferenceList [_ Variable:/:A -2]   
                            [_ Variable:/:B -1]   
                            [_ Variable:/:C 1];    
      k                     1e-45;
    }
   Process SpatiocyteNextReactionProcess( reaction_ )
    {
      VariableReferenceList [_ Variable:/:C -1]   
                            [_ Variable:/:A 2]   
                            [_ Variable:/:B 1];    
      k                     1;
    }
   Process MassActionProcess( reaction2 )
    {
      StepperID       DE;
      VariableReferenceList [_ Variable:/:Am -2]   
                            [_ Variable:/:Bm -1]   
                            [_ Variable:/:Cm 1];    
      k                     1e-45;
    }
   Process MassActionProcess( reaction2_ )
    {
      StepperID       DE;
      VariableReferenceList [_ Variable:/:Cm -1]   
                            [_ Variable:/:Am 2]   
                            [_ Variable:/:Bm 1];    
      k                     1;
    }
   Process SpatiocyteTauLeapProcess( reaction3 )
    {
      VariableReferenceList [_ Variable:/:At -2]   
                            [_ Variable:/:Bt -1]   
                            [_ Variable:/:Ct 1];    
      k                     1e-45;
    }
   Process SpatiocyteTauLeapProcess( reaction3_ )
    {
      VariableReferenceList [_ Variable:/:Ct -1]   
                            [_ Variable:/:At 2]   
                            [_ Variable:/:Bt 1];    
      k                     1;
    }
   Process SpatiocyteTauLeapProcess( reaction4 )
    {
      VariableReferenceList [_ Variable:/:An -1]   
                            [_ Variable:/:Bn 1];    
      k                     0.1;
    }
   Process SpatiocyteTauLeapProcess( reaction5 )
    {
      VariableReferenceList [_ Variable:/:Bn -1]   
                            [_ Variable:/:Cn 1];    
      k                     0.025;
    }
   Process SpatiocyteNextReactionProcess( reaction6 )
    {
      VariableReferenceList [_ Variable:/:Ans -1]   
                            [_ Variable:/:Bns 1];    
      k                     0.1;
    }
   Process SpatiocyteNextReactionProcess( reaction7 )
    {
      VariableReferenceList [_ Variable:/:Bns -1]   
                            [_ Variable:/:Cns 1];    
      k                     0.025;
    }
   Process MassActionProcess( reaction8 )
    {
      StepperID       DE;
      VariableReferenceList [_ Variable:/:Ano -1]   
                            [_ Variable:/:Bno 1];    
      k                     0.1;
    }
   Process MassActionProcess( reaction9 )
    {
      StepperID       DE;
      VariableReferenceList [_ Variable:/:Bno -1]   
                            [_ Variable:/:Cno 1];    
      k                     0.025;
    }
   Process IteratingLogProcess(logiter)
    {
      VariableReferenceList [_ Variable:/:Am]
                            [_ Variable:/:Bm]
                            [_ Variable:/:Cm]
                            [_ Variable:/:A]
                            [_ Variable:/:B]
                            [_ Variable:/:C];
      LogInterval 1;
      LogEnd 10;
    }
}
