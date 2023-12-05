// Gmsh project created on Tue Dec  5 23:42:05 2023
//+
Point(1) = {0, 0.75, 0, 1.0};
//+
Point(2) = {0, 0.25, 0, 1.0};
//+
Point(3) = {1, 0.35, 0, 1.0};
//+
Point(4) = {1, 0.75, 0, 1.0};
//+
Line(1) = {3, 2};
//+
Line(2) = {2, 1};
//+
Line(3) = {1, 4};
//+
Line(4) = {4, 3};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Plane Surface(1) = {1};
//+
Physical Surface(5) = {1};
//+
MeshSize {1, 4, 3, 2} = 0.01;
