import argparse

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time
from cvxopt import solvers, matrix
import cvxpy as cp
from solution import Mesh
import scipy.optimize as spo
from solution_by_Perepechko import Variety
import scipy.sparse as scr

Vertex = Tuple[float, float, float]
Edge = Tuple[int, int]
Face = Tuple[int, int, int]

EPS = 1e-9


class Surface:
    def __init__(self,
                 vertices: List[Vertex],
                 faces: List[Face],
                 triangles: Dict[int, List[int]],
                 measure: Optional[List[float]] = None,
                 p0: Optional[List[float]] = None,
                 p1: Optional[List[float]] = None):

        assert len(triangles.keys()) == len(vertices)

        # required params
        self.n = len(vertices)
        self.vertices = vertices
        self.faces = faces
        self.triangles = triangles

        # optional params
        self.measure = measure if measure is not None else np.full(self.n, 1 / self.n)
        self.p0 = p0 if p0 is not None else np.full(self.n, 1 / self.n)
        self.p1 = p1 if p1 is not None else np.full(self.n, 1 / self.n)

        # self.p0 = np.array([0, 1] + [0] * (self.n - 2), dtype=float)
        # self.p1 = np.array([0, 0, 1] + [0] * (self.n - 3), dtype=float)

        assert len(self.measure) == self.n
        assert len(self.p0) == self.n
        assert len(self.p1) == self.n

        assert np.abs(np.sum(self.measure) - 1) < EPS
        assert np.abs(np.sum(self.p0) - 1) < EPS
        assert np.abs(np.sum(self.p1) - 1) < EPS

        self.change_orientation()
        self.set_additional_values()
        self.edges = {}
        self.centres = []
        for k, face in enumerate(self.faces):
            a = np.linalg.norm(self.vertices[face[1]] -
                               self.vertices[face[2]]) ** 2
            b = np.linalg.norm(self.vertices[face[2]] -
                               self.vertices[face[0]]) ** 2
            c = np.linalg.norm(self.vertices[face[0]] -
                               self.vertices[face[1]]) ** 2

            weights = (a * (b + c - a), b * (c + a - b), c * (a + b - c))
            self.centres.append(np.average(self.vertices[face], weights=weights, axis=0))

            for i in range(3):
                j = (i + 1) % 3
                key = (min(face[i], face[j]), max(face[i], face[j]))
                if face[i] < face[j]:
                    self.edges.setdefault(key, [-1, -1])[0] = k
                else:
                    self.edges.setdefault(key, [-1, -1])[1] = k

    def set_additional_values(self) -> None:

        areas = []

        for i in range(len(self.faces)):
            face = self.faces[i]

            areas.append(self.calculate_area(face))

        self.areas = np.array(areas)
        self.total_area = np.sum(self.areas)

        dual_areas = []

        for i in range(self.n):
            faces_indices = self.triangles[i]
            dual_areas.append((1 / 3) * np.sum(self.areas[faces_indices]))

        assert np.abs(self.total_area - np.sum(dual_areas)) < EPS

        self.dual_areas = np.array(dual_areas)

    def change_orientation(self) -> None:
        variety = Variety(self.faces.tolist())
        res = variety.isOrientable()

        assert res != False

        signs = res[1]
        assert len(signs) == len(self.faces)
        for i, face in enumerate(self.faces):
            if signs[i] == -1:
                face[0], face[1] = face[1], face[0]

    def calculate_area(self, face: Face) -> float:
        ind_0, ind_1, ind_2 = face

        x = self.vertices[ind_0] - self.vertices[ind_1]
        y = self.vertices[ind_0] - self.vertices[ind_2]

        area = 0.5 * np.linalg.norm(np.cross(x, y))

        return area

    def calculate_G(self) -> List[List[float]]:
        Grad = np.zeros((3 * len(self.faces), self.n), dtype=float)

        for i, face in enumerate(self.faces):
            coordinates = self.vertices[face]
            A = np.linalg.inv(coordinates)
            Grad[i * 3: (i + 1) * 3, face] = A

        return Grad

    def calculate_sparse_G(self):
        row = []
        col = []
        data = []
        for i, face in enumerate(self.faces):
            coordinates = self.vertices[face]
            row += [3 * i] * 3 + [3 * i + 1] * 3 + [3 * i + 2] * 3
            col += [*face] * 3
            data += [*np.linalg.inv(coordinates).ravel()]

        res = scr.coo_matrix((data, (row, col)), shape=(3 * len(self.faces), self.n))
        try:
            assert np.array_equal(res.toarray(), self.calculate_G())
        except AssertionError:
            print(res.toarray(), self.calculate_G(), sep='\n\n')
            raise
        return res

    def calculate_D(self, N: Optional[int] = 10) -> List[List[int]]:
        Diff = (np.diagflat(np.full(N + 1, -N)) + np.diagflat(np.full(N, N), 1))[:-1]
        return Diff

    def calculate_sparse_D(self, N: Optional[int] = 10):
        row = np.concatenate((np.arange(N), np.arange(N)))
        col = np.concatenate((np.arange(N), np.arange(N) + 1))
        data = np.concatenate((np.full(N, -N, dtype=float), np.full(N, N)))
        diff = scr.coo_matrix((data, (row, col)), shape=(N, N + 1))
        assert np.array_equal(diff.toarray(), self.calculate_D())
        return diff

    def calculate_measure(self, N: int) -> Tuple[List[float], List[float]]:
        measure = np.linspace(self.p0, self.p1, num=2 * N + 1)
        st_measure = measure[::2]
        c_measure = measure[1::2]

        assert len(st_measure) == N + 1
        assert len(c_measure) == N

        return st_measure, c_measure

    def get_basis(self) -> List[Tuple[Vertex, Vertex]]:
        basis = []
        for i, face in enumerate(self.faces):
            e1 = self.vertices[face[2]] - self.vertices[face[0]]
            e2 = self.vertices[face[1]] - self.vertices[face[0]]
            e2 = e2 - (e2 @ e1) / (e1 @ e1) * e1
            normed_e1, normed_e2 = e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
            e3 = np.cross(normed_e1, normed_e2)

            basis.append((normed_e1, normed_e2, e3))

        return basis

    def find_optimal_solution(self, N: int = 10) -> float:
        Grad = self.calculate_G()
        Diff = self.calculate_D(N)

        soc_constraints = []
        x = cp.Variable((N + 1) * self.n)

        for time in range(N):
            for vertex in range(self.n):
                x_neg = np.hstack((np.diagflat(np.ones(self.n), time * self.n)[:self.n],
                                   np.zeros((self.n, (N - time) * self.n))))
                x_pos = np.hstack((np.diagflat(np.ones(self.n), (time + 1) * self.n)[:self.n],
                                   np.zeros((self.n, (N - time - 1) * self.n))))

                faces_indices = self.triangles[vertex]
                faces_weights = np.zeros(len(self.faces))
                faces_weights[faces_indices] = self.areas[faces_indices]
                faces_weights = np.repeat(np.sqrt(faces_weights), 3)[:, None]

                neg_weighted_grad = faces_weights * (Grad @ x_neg)
                pos_weighted_grad = faces_weights * (Grad @ x_pos)
                weighted_grad = np.vstack((neg_weighted_grad, pos_weighted_grad))

                coef = 1 / (2 * 2 * 3 * self.dual_areas[vertex])
                E = weighted_grad * np.sqrt(coef)

                Z = np.zeros((N + 1, (N + 1) * self.n))
                for i in range(N + 1):
                    Z[i][vertex + i * self.n] = 1

                F = Diff[time] @ Z
                soc_constraints.append(F @ x + cp.square(cp.norm(E @ x)) <= 0)

        X_0 = np.hstack((np.diagflat(np.ones(self.n), 0)[:self.n], np.zeros((self.n, N * self.n))))
        X_1 = np.diagflat(np.ones(self.n), N * self.n)[:self.n]
        c = (self.p1 @ np.diagflat(self.dual_areas) @ X_1) - (self.p0 @ np.diagflat(self.dual_areas) @ X_0)

        prob = cp.Problem(cp.Maximize(c @ x), soc_constraints)
        prob.solve()

        return prob.value

    def find_sparse_distance(self, N: int = 10) -> float:
        Grad = self.calculate_sparse_G()  # shape=(3 * len(self.faces), self.n)
        Diff = self.calculate_sparse_D(N)  # shape=(N, N + 1)

        soc_constraints = []
        x = cp.Variable((N + 1) * self.n)

        for time in range(N):
            for vertex in range(self.n):
                # print(f'time = {time}, vertex = {vertex}')
                x_neg = scr.coo_matrix(
                    (np.ones(self.n, dtype=float), (np.arange(self.n), np.arange(self.n) + time * self.n)),
                    shape=(self.n, (N + 1) * self.n))
                x_pos = scr.coo_matrix(
                    (np.ones(self.n, dtype=float), (np.arange(self.n), np.arange(self.n) + (time + 1) * self.n)),
                    shape=(self.n, (N + 1) * self.n))

                faces_indices = self.triangles[vertex]

                col = [3 * i for i in faces_indices] + [3 * i + 1 for i in faces_indices] + \
                      [3 * i + 2 for i in faces_indices]
                data = [*np.sqrt(self.areas[faces_indices])] * 3

                sparse_faces_weights = scr.coo_matrix((data, (col, col)),
                                                      shape=(3 * len(self.faces), 3 * len(self.faces)))

                neg_weighted_grad = sparse_faces_weights @ (Grad @ x_neg)
                pos_weighted_grad = sparse_faces_weights @ (Grad @ x_pos)
                weighted_grad = scr.vstack((neg_weighted_grad, pos_weighted_grad))

                coef = 1 / (2 * 2 * 3 * self.dual_areas[vertex])
                E = weighted_grad * np.sqrt(coef)

                Z = scr.coo_matrix(([1] * (N + 1), (range(N + 1), [vertex + i * self.n for i in range(N + 1)])),
                                   shape=(N + 1, (N + 1) * self.n))

                F = Diff.getrow(time) @ Z
                soc_constraints.append(F @ x + cp.square(cp.norm(E @ x)) <= 0)
                del x_pos
                del x_neg
                del sparse_faces_weights
                del neg_weighted_grad
                del pos_weighted_grad
                del weighted_grad
                del E
                del Z
                del F

        x0 = scr.coo_matrix((np.ones(self.n), (np.arange(self.n), np.arange(self.n))),
                            shape=(self.n, (N + 1) * self.n))
        x1 = scr.coo_matrix((np.ones(self.n), (np.arange(self.n), np.arange(self.n) + N * self.n)),
                            shape=(self.n, (N + 1) * self.n))
        c = (self.p1 * self.dual_areas @ x1) - (self.p0 * self.dual_areas @ x0)
        del x0
        del x1

        prob = cp.Problem(cp.Maximize(c @ x), soc_constraints)
        prob.solve()

        return prob.value

    def build_mesh(self, filename) -> None:
        self.mesh = Mesh.fromobj(filename)

    def get_laplacian(self) -> np.ndarray:
        laplacian = self.mesh.laplace_operator()
        shape = laplacian.shape
        return laplacian.toarray().reshape(shape)

    def get_f(self) -> np.ndarray:
        laplacian = self.get_laplacian()
        n = laplacian.shape[0]

        new_laplacian = np.pad(laplacian, pad_width=((0, 1), (0, 1)))
        new_laplacian[0][-1] = 1
        new_laplacian[-1][0] = 1

        f = np.linalg.solve(a=new_laplacian,
                            b=np.pad(self.p1 - self.p0, (0, 1))
                            )
        return f[:-1]

    def get_rotation(self):
        rotate_matrix = np.zeros((3 * len(self.faces), 3 * len(self.faces)))
        basis = self.get_basis()
        for i, face in enumerate(self.faces):
            e1, e2, e3 = basis[i]
            e1, e2, e3 = e1[:, None], e2[:, None], e3[:, None]
            a = np.hstack((e1, e2, e3))
            b = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            c = np.vstack((e1.T, e2.T, e3.T))
            right_matrix = a @ b @ c
            rotate_matrix[3 * i: 3 * (i + 1), 3 * i: 3 * (i + 1)] = right_matrix

        return rotate_matrix

    def get_q(self):
        f = self.get_f()
        q = []
        for edge in self.edges:
            q.append(f[edge[0]] - f[edge[1]])
        return np.array(q)

    def get_P(self):
        P = np.zeros((len(self.edges), len(self.faces)))
        for k, edge in enumerate(self.edges):
            ind_centre_0, ind_centre_1 = self.edges[edge]
            weight = (np.linalg.norm(self.vertices[edge[0]] - self.vertices[edge[1]]) /
                      np.linalg.norm(self.centres[ind_centre_0] - self.centres[ind_centre_1]))
            P[k][ind_centre_0] = +weight
            P[k][ind_centre_1] = -weight

        return P

    def hit_method(self):
        q = self.get_q()
        P = self.get_P()
        u = cp.Variable(len(self.faces))
        problem = cp.Problem(cp.Minimize(cp.norm1(P @ u - q)))
        problem.solve()

        return np.linalg.norm(P @ u.value - q)


def read_input() -> Surface:
    v_num, f_num = map(int, input().split())

    triangles = {}

    vertices = []
    for i in range(v_num):
        x, y, z = map(float, input().split())
        vertices.append(np.array([x, y, z]))

    faces = []
    for i in range(f_num):
        v_1, v_2, v_3 = map(int, input().split())
        faces.append([v_1, v_2, v_3])

        triangles.setdefault(v_1, []).append(i)
        triangles.setdefault(v_2, []).append(i)
        triangles.setdefault(v_3, []).append(i)

    p0 = np.array(list(map(float, input().split())))
    p1 = np.array(list(map(float, input().split())))

    polygon = Surface(vertices=np.array(vertices),
                      faces=np.array(faces),
                      triangles=triangles,
                      p0=p0,
                      p1=p1,
                      )

    return polygon


def read_file(surface_filename, distr_filename):
    with open(surface_filename, 'r') as obj:
        lines = [[f for f in s.split(' ') if len(f) > 0] for s in obj.read().split('\n')]

    vertices = [[float(coord) for coord in l[1:4]] for l in lines if len(l) > 3 and l[0] == 'v']
    faces = [[int(coord.split('/')[0]) - 1 for coord in l[1:4]] for l in lines if len(l) > 3 and l[0] == 'f']

    triangles = {}
    for i, face in enumerate(faces):
        assert len(face) == 3

        triangles.setdefault(face[0], []).append(i)
        triangles.setdefault(face[1], []).append(i)
        triangles.setdefault(face[2], []).append(i)

    with open(distr_filename, 'r') as file:
        p0 = [float(x) for x in file.readline().split()]
        p1 = [float(x) for x in file.readline().split()]

    polygon = Surface(np.array(vertices), np.array(faces), triangles, p0=np.array(p0), p1=np.array(p1))

    return polygon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', required=True)
    parser.add_argument('--distributions', '-d', dest='distr', required=True)

    args = parser.parse_args()

    start_time = time.time()
    surface = read_file(args.filename, args.distr)

    read_time = time.time()
    print(f'Reading time: {np.around(read_time - start_time, 3)}')
    ############# SOCP ##############
    surface.build_mesh(args.filename)
    start_time = time.time()
    print(f'Distance from hit method: {np.around(surface.hit_method(), 3)}')
    print(f'Search time: {np.around(time.time() - start_time, 3)}')
    ############# HIT ###############
    if surface.n < 500:
        start_time = time.time()
        print(f'Distance from SOCP problem = {np.around(surface.find_sparse_distance(), 3)}')
        print(f'Search time: {np.around(time.time() - start_time, 3)}')
    else:
        print('Too much vertices for SOCP problem')


if __name__ == '__main__':
    np.random.seed(42)
    main()
