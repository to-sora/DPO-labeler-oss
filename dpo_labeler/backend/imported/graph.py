class Graph:
    def __init__(self, count: int, down: list[int] | None = None,
                 up: list[int] | None = None) -> None:
        self.count = count
        self.down = list(down) if down is not None else [0] * count
        self.up = list(up) if up is not None else [0] * count

    def known(self, a: int, b: int) -> int | None:
        if self.down[a] & (1 << b):
            return a
        if self.down[b] & (1 << a):
            return b
        return None

    def add(self, winner: int, loser: int) -> None:
        if winner == loser or self.known(winner, loser) == loser:
            raise ValueError("Comparison contradicts accepted preferences")
        ancestors = self.up[winner] | (1 << winner)
        descendants = self.down[loser] | (1 << loser)
        for i in range(self.count):
            if ancestors & (1 << i):
                self.down[i] |= descendants
            if descendants & (1 << i):
                self.up[i] |= ancestors

    def intervals(self) -> list[tuple[int, int]]:
        return [(1 + self.up[i].bit_count(), self.count - self.down[i].bit_count())
                for i in range(self.count)]

    def dump(self) -> dict:
        return {"down": self.down, "up": self.up}
