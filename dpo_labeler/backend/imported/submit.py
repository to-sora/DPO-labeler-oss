import json
import sqlite3

from .database import encode, read, write
from .events import Conflict, make_events, submission
from .state import graphs_from, refresh


def accept(db: sqlite3.Connection, payload: dict) -> dict:
    request = submission(payload)
    previous = db.execute("SELECT request, events FROM batches WHERE id=?",
                          (request["comparison_id"],)).fetchone()
    if previous:
        if json.loads(previous[0]) != request:
            raise Conflict("This comparison was already saved; refresh to continue")
        return {"events": json.loads(previous[1]), "replayed": True}
    snapshot, state = read(db, "snapshot"), read(db, "state")
    pair = state["pending"]
    if pair is None or pair["comparison_id"] != request["comparison_id"]:
        raise Conflict("Comparison is no longer active; refresh to continue")
    dims = snapshot["dimensions"]
    if set(request["choices"]) - set(dims):
        raise ValueError("Unknown dimension in choices")
    a, b = pair["image_ids"]
    graphs, decisions = graphs_from(state), []
    for dim, graph in zip(dims, graphs):
        known = graph.known(a, b)
        supplied = request["choices"].get(dim)
        inferred = "a_good" if known == a else "b_good" if known == b else None
        if inferred and supplied is not None and supplied != inferred:
            raise Conflict(f"{dim}: preference is already implied; refresh to continue")
        decision = inferred or supplied
        if decision not in ("a_good", "b_good"):
            raise ValueError(f"{dim}: choose A or B")
        decisions.append((decision, inferred is not None))
    for graph, (decision, _) in zip(graphs, decisions):
        graph.add(a if decision == "a_good" else b, b if decision == "a_good" else a)
    events = make_events(snapshot, pair, request, decisions)
    state["exposure"][a] += 1
    state["exposure"][b] += 1
    state["comparisons"] += 1
    state["pending"] = None
    refresh(state, graphs)
    db.execute("INSERT INTO batches VALUES (?, ?, ?)",
               (request["comparison_id"], encode(request), encode(events)))
    write(db, "state", state)
    return {"events": events, "replayed": False}
