from collections import Counter

from .graph import Graph


def merge_request(graph: Graph) -> tuple[int, int] | None:
    runs = [[i] for i in range(graph.count)]
    while len(runs) > 1:
        next_runs = []
        for offset in range(0, len(runs), 2):
            if offset + 1 == len(runs):
                next_runs.append(runs[offset])
                continue
            left, right = runs[offset:offset + 2]
            merged, a, b = [], 0, 0
            while a < len(left) and b < len(right):
                winner = graph.known(left[a], right[b])
                if winner is None:
                    return tuple(sorted((left[a], right[b])))
                merged.append(winner)
                if winner == left[a]:
                    a += 1
                else:
                    b += 1
            next_runs.append(merged + left[a:] + right[b:])
        runs = next_runs
    return None


def schedule(graphs: list[Graph], rankings: list[dict],
             exposure: list[int], precheck: int) -> tuple[list[int] | None, int]:
    if all(r["certified"] for r in rankings):
        return None, precheck
    n = len(exposure)
    while precheck < n - 1:
        a, b = precheck, precheck + 1
        precheck += 1
        if any(g.known(a, b) is None for g in graphs):
            return [a, b], precheck
    requests = Counter()
    for graph, ranking in zip(graphs, rankings):
        if not ranking["certified"]:
            pair = merge_request(graph)
            if pair is not None:
                requests[pair] += 1
    if not requests:
        raise RuntimeError("Incomplete ranking has no pending comparisons")
    pair = min(requests, key=lambda p: (max(exposure[i] for i in p),
               -requests[p], sum(exposure[i] for i in p), p))
    return list(pair), precheck
