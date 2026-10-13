import uuid

from .graph import Graph
from .ranking import rank
from .scheduler import schedule


def graphs_from(state: dict) -> list[Graph]:
    return [Graph(len(state["exposure"]), **g) for g in state["graphs"]]


def refresh(state: dict, graphs: list[Graph]) -> None:
    state["graphs"] = [g.dump() for g in graphs]
    state["rankings"] = [rank(g) for g in graphs]


def new_state(snapshot: dict) -> dict:
    n = len(snapshot["images"])
    state = {"exposure": [0] * n, "comparisons": 0, "precheck": 0,
             "pending": None}
    refresh(state, [Graph(n) for _ in snapshot["dimensions"]])
    return state


def prepare_pair(state: dict) -> None:
    if state["pending"] is not None:
        return
    graphs = graphs_from(state)
    pair, state["precheck"] = schedule(
        graphs, state["rankings"], state["exposure"], state["precheck"])
    if pair is not None:
        state["pending"] = {"comparison_id": uuid.uuid4().hex, "image_ids": pair}


def public_state(snapshot: dict, state: dict) -> dict:
    result = {**snapshot, "comparisons": state["comparisons"],
              "exposure": state["exposure"], "rankings": state["rankings"],
              "complete": all(r["certified"] for r in state["rankings"]),
              "pair": None}
    if state["pending"] is not None:
        pair = state["pending"]
        a, b = pair["image_ids"]
        locked = {}
        for name, graph in zip(snapshot["dimensions"], graphs_from(state)):
            winner = graph.known(a, b)
            if winner is not None:
                locked[name] = "a_good" if winner == a else "b_good"
        result["pair"] = {**pair, "locked": locked,
                          "images": [snapshot["images"][a], snapshot["images"][b]]}
    return result
