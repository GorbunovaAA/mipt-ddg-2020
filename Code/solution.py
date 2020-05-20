import numpy as np
import math
import draw
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import inv
from scipy.sparse import identity
from scipy.sparse import vstack
import time


class Mesh:
    def __init__(self, faces, coordinates=None):
        self.faces = np.array(faces)
        self.color = None
        vertices = set(i for f in faces for i in f)
        self.n = max(vertices) + 1
        if coordinates is not None:
            self.coordinates = np.array(coordinates)

        assert set(range(self.n)) == vertices
        # self.adj = np.full((self.n, self.n), -1)
        self.adj = lil_matrix((self.n, self.n), dtype=np.int64)
        for i in range(len(faces)):
            f = self.faces[i, :]
            assert len(f) == 3
            for j in range(3):
                self.adj[f[j], f[(j + 1) % 3]] = i + 1
        if coordinates is not None:
            assert self.n == len(coordinates)
            for c in coordinates:
                assert len(c) == 3

    @classmethod
    def fromobj(cls, filename):
        faces, vertices = draw.obj_read(filename)
        return cls(faces, vertices)

    def draw(self):
        draw.draw(self.faces, self.coordinates.tolist())

    def save(self, filename):
        draw.obj_write(filename, self)

    def cotan(self, vertex1, vertex2):
        """
        returns the cotangent of the edge vertex1 vertex2
        """
        aux = np.array([vertex1, vertex2])
        ind = self.adj[vertex1, vertex2] - 1
        vertex3 = np.setdiff1d(self.faces[ind], aux)[0]
        ind = self.adj[vertex2, vertex1] - 1
        vertex4 = np.setdiff1d(self.faces[ind], aux)[0]

        c1 = self.coordinates[vertex1]
        c2 = self.coordinates[vertex2]
        c3 = self.coordinates[vertex3]
        c4 = self.coordinates[vertex4]
        v31 = c1 - c3
        v32 = c2 - c3
        cotan1 = np.dot(v31, v32) / np.linalg.norm(np.cross(v31, v32))
        v41 = c1 - c4
        v42 = c2 - c4
        cotan2 = np.dot(v41, v42) / np.linalg.norm(np.cross(v41, v42))
        return cotan1 + cotan2

    def dual_area(self, vertex):
        """
        returns the dual area of vertex
        """
        c1 = self.coordinates[vertex]
        indices = np.argwhere(self.faces == vertex)
        # indices = self.adj[vertex, :][self.adj[vertex, :] > -1]
        sum = 0.0
        for i in indices:
            vertex2 = self.faces[i[0], (i[1] + 1) % 3]
            vertex3 = self.faces[i[0], (i[1] - 1) % 3]
            c2 = self.coordinates[vertex2]
            c3 = self.coordinates[vertex3]

            l12 = np.linalg.norm(c2 - c1) ** 2
            l13 = np.linalg.norm(c3 - c1) ** 2

            v1 = (c1 - c2)
            v2 = (c3 - c2)
            cotan_2 = np.dot(v1, v2) / np.linalg.norm(np.cross(v1, v2))
            v1 = (c1 - c3)
            v2 = -v2
            cotan_3 = np.dot(v1, v2) / np.linalg.norm(np.cross(v1, v2))
            sum += (l12 * cotan_3 + l13 * cotan_2)
        return 1 / 8 * sum

    def weak_laplace_operator(self):
        """
        returns a weak Laplacian matrix C that consists of cotangents
        """
        C = lil_matrix((self.n, self.n), dtype=np.float64)
        for i in range(self.n):
            indices = (self.adj[i, :][self.adj[i, :] > 0]).toarray().flatten() - 1
            for j in indices:
                v = np.setdiff1d(self.faces[j, :], np.array([i]))
                valA = 1 / 2 * self.cotan(i, v[0])
                valB = 1 / 2 * self.cotan(i, v[1])
                C[i, v[0]] = valA
                C[i, v[1]] = valB
            C[i, i] = -C[i, :].sum()
        return C.tocsc()

    def laplace_operator(self):
        """
        returns the Laplacian Operator L
        """
        C = self.weak_laplace_operator()
        M = coo_matrix((self.n, self.n), dtype=np.float64)
        values = np.zeros(self.n)
        for i in range(self.n):
            values[i] = abs(self.dual_area(i))
        M.setdiag(values)
        M = M.tocsc()
        res = inv(M) * C
        # print(f'Laplacian operator form = {res.shape}')
        return res

    def heat_flow(self, func, step=1 / 1000, nsteps=1000, implicit=False):
        """
        this method integrates the function 'func' via the heat equation
        func is an array of function values at vertices,
        step is the length of one step (also called h, or epsilon),
        nsteps is the number of steps to perform.
        implicit is an indication whether we should use implicit integration (backward Euler) or explicit (forward Euler)
        """
        u = np.array(func)
        L = self.laplace_operator()
        if implicit:
            A = inv(identity(self.n, format='csc') - step * L)
            for i in range(nsteps):
                u = A.dot(u)
        else:
            for i in range(nsteps):
                u = u + step * L.dot(u)
        return u

    def reconstruct(self, anchors, anchor_coordinates, edit, edit_coordinates, anchor_weight=1.):
        """
        anchors is a list of vertex indices,
        anchor_coordinates is a list of same length of vertex coordinates (arrays of length 3),
        anchor_weight is a positive number
        """
        L = self.weak_laplace_operator()

        size_A2 = len(anchors) + len(edit)

        b1 = L.dot(self.coordinates)

        A2 = lil_matrix((size_A2, self.n))
        b2 = lil_matrix((size_A2, 3))
        for i in range(len(anchors)):
            A2[i, anchors[i]] = 1
            b2[i, :] = anchor_coordinates[i, :]

        for i in range(len(edit)):
            A2[-(i + 1), edit[i]] = 1
            b2[-(i + 1), :] = edit_coordinates[i, :]

        A = vstack([L, A2.tocsc()])
        b = vstack([b1, b2.tocsc()])

        a = A.transpose().dot(A)
        b = csc_matrix(A.transpose().dot(b))
        return spsolve(a, b)


def perform():
    # Uses reconstruct to do something with teddy for example
    mesh = Mesh.fromobj("../Data/teddy.obj")
    anchors = [913, 0, 795, 982, 590, 338]
    anchor_coordinates = []
    for v in anchors:
        anchor_coordinates.append(mesh.coordinates[v, :])
    anchor_coordinates = np.array(anchor_coordinates)
    edit = [680, 585]
    edit_coordinates = np.zeros((2, 3))
    edit_coordinates[0, :] = mesh.coordinates[edit[0], :] + np.array([0.0, 5.0, 0.0])
    edit_coordinates[1, :] = mesh.coordinates[edit[1], :] + np.array([-2.0, -5.0, 0.0])
    mesh.coordinates = mesh.reconstruct(anchors, anchor_coordinates, edit, edit_coordinates).toarray()
    mesh.save("../Data/new_teddy.obj")


def dragon():
    mesh = Mesh.fromobj("../Data/dragon.obj")
    anchors = [43670, 13341, 6553, 19814, 8027]
    anchor_coordinates = []
    for v in anchors:
        anchor_coordinates.append(mesh.coordinates[v, :])
    anchor_coordinates = np.array(anchor_coordinates)
    edit = [485]
    edit_coordinates = np.zeros((1, 3))
    edit_coordinates[0, :] = mesh.coordinates[edit[0], :] + np.array([0.0, 2.0, 0.0])
    # edit_coordinates[1, :] = mesh.coordinates[edit[1], :] + np.array([-2.0, -5.0, 0.0])
    mesh.coordinates = mesh.reconstruct(anchors, anchor_coordinates, edit, edit_coordinates).toarray()
    mesh.save("../Data/new_dragon.obj")


if __name__ == "__main__":
    start = time.time()
    dragon()
    dragon_time = time.time() - start
    print(f'Dragon time = {dragon_time // 60}m {dragon_time % 60}sec')
    perform()
    teddy_time = time.time() - dragon_time
    print(f'Teddy time = {teddy_time // 60}m {teddy_time % 60}sec')
