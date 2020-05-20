class Variety():
    def __init__(self, faces):
        self.faces = faces
        vertices = set(i for f in faces for i in f)
        self.n = max(vertices) + 1
        assert set(range(self.n)) == vertices
        for f in faces:
            assert len(f) == 3

    @property
    def num_vertices(self):
        return self.n

    def check(self):
        return self.isComplex() and self.isSurface() and self.isOriented() and self.isConnected()

    def isComplex(self):
        # сортируем вершины в каждой грани и удаляем дубликаты
        s = set(tuple(sorted(list(f))) for f in self.faces)

        # проверяем, были ли дубликаты
        if len(s) != len(self.faces):
            return False
        return True

    def isSurface(self):
        if not self.isComplex():
            return False
        # для каждой вершины проверим, что окрестность вершины - диск
        return all(self.neighbourhoodIsDisc(vertex) for vertex in range(self.n))

    # проверяем, что окрестность вершины - диск
    def neighbourhoodIsDisc(self, vertex):
        # функция link возвращает нам максимальный по включению справа путь на смежных вершинах
        link = self.link(vertex)
        # выпишем рёбра линка
        link_edges = [[v for v in f if v != vertex] for f in self.faces if vertex in f]
        # достаточно проверить, что каждая вершина линка содержится в нашем пути и участвует ровно в двух рёбрах
        return set(link) == set(v for edge in link_edges for v in edge) and all(
            len([1 for edge in link_edges if v in edge]) == 2 for v in link)

    def link(self, vertex):
        # выпишем рёбра линка
        link_edges = [[v for v in f if v != vertex] for f in self.faces if vertex in f]
        # начнём строить линк с одного из рёбер
        link = list(link_edges.pop())
        # путешествуем по линку
        while True:
            # записываем номера рёбер из оставшихся, содержащие последнюю вершину
            incident_edges = [enumerated_edge[0] for enumerated_edge in enumerate(link_edges) if
                              link[-1] in enumerated_edge[1]]
            # если некуда идти - заканчиваем
            if len(incident_edges) == 0:
                return link
            # добавляем ребро к линку, удаляя из списка рёбер
            new_edge = link_edges.pop(incident_edges[0])
            # находим новую вершину
            new_vertex = [v for v in new_edge if v != link[-1]][0]
            # если новая вершина уже была - заканчиваем
            if new_vertex in link:
                return link
            # добавляем в линк
            link.append(new_vertex)
        print(link_edges)

    # возращает список всех рёбер всех граней согласно их ориентации
    def edges(self):
        edge_triples = [[(f[0], f[1]), (f[1], f[2]), (f[2], f[0]), ] for f in self.faces]
        return [edge for edges in edge_triples for edge in edges]

    def isConnected(self):
        if not self.isSurface():
            return False
        # будем искать компоненты связности, объединяя их согласно рёбрам
        connected_components = [[i] for i in range(self.n)]

        def find_component(vertex):
            for index, component in enumerate(connected_components):
                if vertex in component:
                    return index

        for edge in self.edges():
            first_component = find_component(edge[0])
            second_component = find_component(edge[1])
            if first_component != second_component:
                connected_components[first_component] = connected_components[first_component] + connected_components[
                    second_component]
                connected_components.pop(second_component)

        return len(connected_components) == 1

    def isOriented(self):
        if not self.isSurface():
            return False
        edges = self.edges()
        return all((edge[1], edge[0]) in edges for edge in edges)

    def isOrientable(self):
        if not self.isSurface():
            return False
        # в signs будем хранить, какую ориентацию следует предписать грани для корректности
        signs = [1 for i in range(len(self.faces))]
        # в components будем хранить ориентированные куски поверхности
        components = [[i] for i in range(len(self.faces))]

        def find_component(face):
            for index, component in enumerate(components):
                if face in component:
                    return index

        def signs_consistent(face_index1, face_index2):
            faces = [self.faces[face_index1], self.faces[face_index2]]
            edge = list(set(faces[0]).intersection(set(faces[1])))
            orientations = [(f.index(edge[0]) - f.index(edge[1])) % 3 for f in faces]
            orientations = [(1 if o == 1 else -1) for o in orientations]
            return orientations[0] * signs[face_index1] == - orientations[1] * signs[face_index2]

        facesWithEdge = lambda edge: [face_index for face_index in range(len(self.faces)) if
                                      all(e in self.faces[face_index] for e in edge)]

        for edge in self.edges():
            incident_faces = facesWithEdge(edge)
            incident_components = [find_component(f) for f in incident_faces]
            if incident_components[0] == incident_components[1]:
                if not signs_consistent(incident_faces[0], incident_faces[1]):
                    return False
            else:
                if not signs_consistent(incident_faces[0], incident_faces[1]):
                    for face in components[incident_components[1]]:
                        signs[face] = -signs[face]
                components[incident_components[0]] = components[incident_components[0]] + components[
                    incident_components[1]]
                components.pop(incident_components[1])
        return True, signs

    def Euler(self):
        return self.n + len(self.faces) - len(self.edges()) // 2
