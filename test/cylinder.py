from rhino3dm import *

pt = Point3d(1, 1, 1)
circle = Circle(pt, 5)
circle_curve = circle.ToNurbsCurve()
extrusion = Extrusion.Create(circle_curve, 10, True)
mesh_type = MeshType(0) # idk what 0 means, no documentation
mesh = extrusion.GetMesh(mesh_type)
file_rhino = File3dm()
file_rhino.Objects.AddExtrusion(extrusion)
export_result = file_rhino.Write('cylinder.3dm', 6)

print(export_result)

