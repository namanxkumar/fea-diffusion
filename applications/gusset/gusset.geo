// Gmsh project created on Fri Jan  5 23:06:34 2024
SetFactory("OpenCASCADE");
//+
Point(1) = {-0, -0, 0, 1.0};
//+
Point(2) = {0, 1, 0, 1.0};
//+
Point(3) = {0.5, 1, 0, 1.0};
//+
Point(4) = {1, 0.5, 0, 1.0};
//+
Point(5) = {1, -0, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 1};
//+
MeshSize {3, 2, 1, 5, 4} = 0.01;
//+
Curve Loop(1) = {2, 3, 4, 5, 1};
//+
Plane Surface(1) = {1};
//+
Physical Surface(6) = {1};
