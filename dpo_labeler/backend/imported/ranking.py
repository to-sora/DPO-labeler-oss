import heapq

from .graph import Graph


def certificate(intervals: list[tuple[int, int]], cuts: list[int],
                tolerance: int) -> list[list[int]] | None:
    jobs = []
    for i, (lower, upper) in enumerate(intervals):
        legal = [b for b in range(5) if lower > cuts[b] - tolerance
                 and upper <= cuts[b + 1] + tolerance]
        if not legal:
            return None
        jobs.append((min(legal), max(legal), i))
    jobs.sort()
    heap, groups, cursor = [], [], 0
    for b in range(5):
        while cursor < len(jobs) and jobs[cursor][0] <= b:
            _, last, i = jobs[cursor]
            heapq.heappush(heap, (last, i))
            cursor += 1
        group = []
        for _ in range(cuts[b + 1] - cuts[b]):
            if not heap:
                return None
            last, i = heapq.heappop(heap)
            if last < b:
                return None
            group.append(i)
        groups.append(group)
    return groups


def rank(graph: Graph) -> dict:
    n = graph.count
    cuts = [j * n // 5 for j in range(6)]
    tolerance = n // 20
    intervals = graph.intervals()
    groups = certificate(intervals, cuts, tolerance)
    certified = groups is not None
    if groups is None:
        order = sorted(range(n), key=lambda i: (sum(intervals[i]), i))
        groups = [order[cuts[b]:cuts[b + 1]] for b in range(5)]
    cutoffs, selected = {}, []
    for j in range(1, 5):
        selected = selected + groups[j - 1]
        chosen = set(selected)
        proven = all(upper <= cuts[j] + tolerance if i in chosen
                     else lower > cuts[j] - tolerance
                     for i, (lower, upper) in enumerate(intervals))
        cutoffs[f"{j / 5:.1f}"] = {"image_ids": selected, "certified": proven}
    return {"certified": certified, "tolerance_ranks": tolerance,
            "rank_intervals": [{"image_id": i, "lower": lo, "upper": hi}
                               for i, (lo, hi) in enumerate(intervals)],
            "groups": groups, "cutoffs": cutoffs}
