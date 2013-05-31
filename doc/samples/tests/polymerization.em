Stepper SpatiocyteStepper(SS) { VoxelRadius 3e-8; }   # m
System System(/)
{
  StepperID       SS; 
  Variable Variable(GEOMETRY)
    {
      Value 3;         # { 0: Cuboid (uses LENGTHX, LENGTHY, LENGTHZ)
                       #   1: Ellipsoid (uses LENGTHX, LENGTHY, LENGTHZ) }
                       #   2: Cylinder (uses LENGTHX, LENGTHY=2*radius)
                       #   3: Rod (uses LENGTHX, LENGTHY=2*radius)
                       #   4: Torus (uses LENGTHX, LENGTHY, LENGTHZ) }
                       #   5: Pyramid (uses LENGTHX, LENGTHY, LENGTHZ) }
    } 
  Variable Variable(LENGTHX)
    {
      Value 4.5e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 1e-6;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 0; 
    } 
  Process VisualizationLogProcess(visualize)
    {
      VariableReferenceList [_ Variable:/Surface:A]
                            [_ Variable:/Surface:B];
    }
  Process MoleculePopulateProcess(populate)
    {
      VariableReferenceList [_ Variable:/Surface:A ]
                            [_ Variable:/Surface:B ];
    }
}

System System(/Surface)
{
  StepperID SS;
  Variable Variable(DIMENSION)
    {
      Value 2;    
    } 
  Variable Variable(VACANT)
    {
      Value 1;
    } 
  Variable Variable(A)
    {
      Value 300;         # molecule number 
    } 
  Variable Variable(B)
    {
      Value 0;         # molecule number 
    }
  Process DiffusionProcess(diffuseA)
    {
      VariableReferenceList [_ Variable:/Surface:A];
      D 0.1e-12;
    }
  Process PolymerizationParameterProcess(param)
    {
      VariableReferenceList [_ Variable:/Surface:B];
      BendAngles [0.78];
    }
# A + A -> B + B
  Process PolymerizationProcess(dimerize)
    {
      VariableReferenceList [_ Variable:/Surface:A -1]
                            [_ Variable:/Surface:A -1]
                            [_ Variable:/Surface:B 1]
                            [_ Variable:/Surface:B 1];
      p 1;
    }
# A + B -> B + B
  Process PolymerizationProcess(polymerize)
    {
      VariableReferenceList [_ Variable:/Surface:B -1]
                            [_ Variable:/Surface:A -1]
                            [_ Variable:/Surface:B 1]
                            [_ Variable:/Surface:B 1];
      p 1;
    }
}

