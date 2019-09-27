Merge "cas_delicat2.step";
l1 = 3e-4;
l2 = 0.25*l1;
l3 = 0.5*l1;
l4 = 8*l1;
For i In {1:19}
  Characteristic Length {i} = l1;
EndFor
Characteristic Length {11} = l2;
Characteristic Length {17} = l3;
Characteristic Length {19} = l3;
Characteristic Length {5} = l4;
Characteristic Length {2} = l4;
Characteristic Length {8} = l4;
Characteristic Length {14} = l4;

