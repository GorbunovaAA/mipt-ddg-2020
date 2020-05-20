from pythreejs import *
from IPython.display import display


def obj_read(filename):
    with open(filename, 'r') as obj:
        lines = [[f for f in s.split(' ') if len(f) > 0] for s in obj.read().split('\n')]
    vertices = [[float(coord) for coord in l[1:4]] for l in lines if len(l) > 3 and l[0] == 'v']
    faces = [[int(coord.split('/')[0]) - 1 for coord in l[1:4]] for l in lines if len(l) > 3 and l[0] == 'f']
    return faces, vertices


def obj_write(filename, mesh):
    with open(filename, 'w+') as obj:
        if mesh.color is None:
            for v in mesh.coordinates:
                obj.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
        else:
            for i in range(mesh.n):
                v = mesh.coordinates[i]
                c = mesh.color[i]
                obj.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) +
                          " " + str(c[0]) + " " + str(c[1]) + " " + str(c[2]) + "\n")
        for f in mesh.faces:
            obj.write("f " + str(f[0]+1) + " " + str(f[1]+1) + " " + str(f[2]+1) + "\n")


def draw(faces, vertices):
    # Create the geometry:
    vertexcolors = ['#0000ff' for v in vertices]
    faces = [f + [None, [vertexcolors[i] for i in f], None] for f in faces]
    geometry = Geometry(faces=faces, vertices=vertices, colors=vertexcolors)
    # Calculate normals per face, for nice crisp edges:
    geometry.exec_three_obj_method('computeFaceNormals')

    object1 = Mesh(
        geometry=geometry,
        material=MeshLambertMaterial(color="brown", side="FrontSide"),
    )

    object2 = Mesh(
        geometry=geometry,
        material=MeshLambertMaterial(color="black", side="BackSide"),
    )

    # Set up a scene and render it:
    camera = PerspectiveCamera(
        position=[2 * max(v[0] for v in vertices), 2 * max(v[1] for v in vertices), 2 * max(v[2] for v in vertices)],
        fov=40,
        children=[DirectionalLight(color='#cccccc', position=[-3, 5, 1], intensity=0.5)])
    scene = Scene(children=[object1, object2, camera, AmbientLight(color='#dddddd')])

    renderer = Renderer(camera=camera, background='black', background_opacity=1,
                        scene=scene, controls=[OrbitControls(controlling=camera)])

    display(renderer)
