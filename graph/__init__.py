# faces: dict[face_id, Face()]
def bfs(faces):
    if len(faces) == 0:
        return []

    # candidate
    nodes = set([f.face_id for f in faces.values()])
    visited = set()

    connect_parts = []

    first_src = nodes.pop()
    queue = [first_src]  # randomly get one
    visited.add(first_src)
    current_connect_sub_graph = [first_src]

    while queue:
        top_id = queue.pop()
        for n in faces[top_id].neighbors:
            if n in visited:
                continue
            elif n not in faces:
                # we don't have this n
                continue
            else:
                queue.append(n)
                visited.add(n)
                nodes.remove(n)
                current_connect_sub_graph.append(n)

        if len(queue) == 0:
            # we reach a dead end
            if len(current_connect_sub_graph) > 0:
                connect_parts.append(current_connect_sub_graph)
                current_connect_sub_graph = []

            if len(nodes) > 0:
                next_src = nodes.pop()
                current_connect_sub_graph.append(next_src)
                queue.append(next_src)
                visited.add(next_src)
            else:
                break
    return connect_parts


class Solver(object):
    def bitfield(self, n):
        bit_str = ('{:0%db}' % self.n_bits).format(n)
        return [int(digit) for digit in bit_str]

    def __init__(self,
                 id2edge,
                 graph):
        self.n_bits = len(id2edge)


# edge
class Edge(object):
    def __init__(self, edge_id, binomial):
        self.edge_id = edge_id
        self.is_break = True
        self.binomial = binomial

    def broke(self):
        pass

    def prob(self):
        if self.is_break:
            return self.binomial
        else:
            return 1.0 - self.binomial


class Face(object):
    def __init__(self, face_id, edges, neighbors, size=1):
        self.face_id = face_id
        self.edges = edges
        self.neighbors = neighbors  # neighbors id
        self.size = size

    def is_complete(self):
        for e in self.edges:
            if e.is_break:
                return False
        return True


def bitfield(n):
    bit_str = '{:017b}'.format(n)
    return [int(digit) for digit in bit_str]


def permutation(begin, end):
    for i in range(begin, end):
        bits = bitfield(i)
        yield bits
