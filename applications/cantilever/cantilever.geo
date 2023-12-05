// Gmsh project created on Tue Dec  5 15:07:55 2023
//+
Point(1) = {-0, 0.4, 0, 1.0};
//+
Point(2) = {0, 0.6, 0, 1.0};
//+
Point(3) = {1, 0.4, 0, 1.0};
//+
Point(4) = {1, 0.6, 0, 1.0};
//+
Line(1) = {1, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 2};
//+
Line(4) = {2, 1};
//+
Curve Loop(1) = {3, 4, 1, 2};
//+
Plane Surface(1) = {1};
//+
Physical Surface(5) = {1};
//+
MeshSize {2, 4, 3, 1} = 0.01;
//+
MeshSize {2} = 0.01;
