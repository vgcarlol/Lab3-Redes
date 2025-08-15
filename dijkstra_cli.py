import argparse, json, sys, csv, heapq
from collections import defaultdict
from typing import Dict, Iterable, Tuple, Any, Optional, List


# Grafo (lista de adyacencia)
class Graph:
    """Grafo ponderado simple. Por defecto es no dirigido."""

    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adj: Dict[Any, Dict[Any, float]] = defaultdict(dict)

    def add_edge(self, u, v, w: float):
        w = float(w)
        if w < 0:
            raise ValueError("Dijkstra requiere pesos no negativos")
        self.adj[u][v] = w
        if not self.directed:
            self.adj[v][u] = w

    def neighbors(self, u) -> Dict[Any, float]:
        return self.adj.get(u, {})

    @classmethod
    def from_edge_list(
        cls, edges: Iterable[Tuple[Any, Any, float]], directed: bool = False
    ):
        g = cls(directed=directed)
        for u, v, w in edges:
            g.add_edge(u, v, w)
        return g

    @classmethod
    def read_csv(cls, path: str, directed: bool = False, delimiter: str = ","):
        """CSV con columnas: u{delim}v{delim}weight. Las líneas que empiezan con # son comentarios."""
        edges = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=delimiter):
                if not row:
                    continue
                if str(row[0]).strip().startswith("#"):
                    continue
                if len(row) < 3:
                    raise ValueError(f"Fila inválida, se esperan 3 campos: {row}")
                u, v, w = row[0].strip(), row[1].strip(), float(row[2])
                edges.append((u, v, w))
        return cls.from_edge_list(edges, directed=directed)

    @classmethod
    def read_json(cls, path: str):
        """JSON:
        {
          "directed": false,
          "edges": [["A","B",1], ["B","C",2]]
        }"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        directed = bool(data.get("directed", False))
        edges = data.get("edges", [])
        return cls.from_edge_list(edges, directed=directed)


# Dijkstra + utilidades
def dijkstra(graph: Graph, source: Any):
    """Devuelve (dist, prev):
    dist[nodo] = distancia mínima desde source
    prev[nodo] = predecesor inmediato en el camino mínimo (None para source)
    """
    dist: Dict[Any, float] = {}
    prev: Dict[Any, Optional[Any]] = {}
    pq: List[Tuple[float, Any]] = []

    # Inicialización
    for node in graph.adj.keys():
        dist[node] = float("inf")
        prev[node] = None
    if source not in graph.adj:
        graph.adj.setdefault(source, {})  # permite origen aislado
    dist[source] = 0.0
    heapq.heappush(pq, (0.0, source))
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if d > dist[u]:
            continue
        for v, w in graph.neighbors(u).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev: Dict[Any, Optional[Any]], target: Any):
    path = []
    cur = target
    while cur is not None and cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def next_hop_table(prev: Dict[Any, Optional[Any]], source: Any):
    """Construye la tabla de próximo salto a partir del árbol de predecesores."""
    table: Dict[Any, Any] = {}
    for node, parent in prev.items():
        if node == source or parent is None:
            continue
        cur = node
        # retrocede hasta el vecino directo del origen
        while prev.get(cur) is not None and prev[cur] != source:
            cur = prev[cur]
        nh = (
            cur if prev.get(cur) == source else node
        )  # si no hay ruta directa, queda como sí mismo
        table[node] = nh
    return table


# CLI
def build_argparser():
    p = argparse.ArgumentParser(
        prog="dijkstra", description="Dijkstra: caminos mínimos (SSSP)"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Ruta a CSV: u,v,weight")
    src.add_argument("--json", help="Ruta a JSON con edges")
    p.add_argument("--delimiter", default=",", help="Delimitador CSV (por defecto ',')")
    p.add_argument(
        "--directed",
        action="store_true",
        help="Tratar grafo como dirigido (por defecto no dirigido)",
    )
    p.add_argument("--source", required=True, help="Nodo origen")
    p.add_argument("--target", help="Nodo destino (opcional; imprime camino y costo)")
    p.add_argument(
        "--dump-spt",
        action="store_true",
        help="Imprimir aristas del árbol de caminos mínimos (SPT)",
    )
    p.add_argument(
        "--print-next-hop", action="store_true", help="Imprimir tabla de próximo salto"
    )
    p.add_argument("--as-json", action="store_true", help="Emitir resultados en JSON")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.csv:
        g = Graph.read_csv(args.csv, directed=args.directed, delimiter=args.delimiter)
    else:
        g = Graph.read_json(args.json)
        if args.directed:
            g.directed = True

    dist, prev = dijkstra(g, args.source)
    result = {"directed": g.directed, "source": args.source, "distances": dist}
    if args.target:
        path = reconstruct_path(prev, args.target)
        result["target"] = args.target
        result["path"] = path
        result["cost"] = dist.get(args.target, float("inf"))
    if args.dump_spt:
        spt = []
        for v, u in prev.items():
            if u is not None:
                spt.append([u, v])
        result["spt_edges"] = spt
    if args.print_next_hop:
        result["next_hop"] = next_hop_table(prev, args.source)

    if args.as_json:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(f"Source: {args.source}")
        print("Distances:")
        for k in sorted(result["distances"]):
            print(f"  {k}: {result['distances'][k]}")
        if args.target:
            print(
                f"\nShortest path {args.source} -> {args.target}: {result.get('path', [])} (cost={result.get('cost')})"
            )
        if args.dump_spt:
            print("\nSPT edges:")
            for u, v in result.get("spt_edges", []):
                print(f"  {u} -> {v}")
        if args.print_next_hop:
            print("\nNext-hop table:")
            for dst, nh in result.get("next_hop", {}).items():
                print(f"  to {dst}: next hop {nh}")


if __name__ == "__main__":
    main()
