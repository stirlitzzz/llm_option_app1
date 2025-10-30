from collections import defaultdict
from typing import Callable, Dict, List, Any


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        self._subs[topic].append(handler)

    def publish(self, topic: str, payload: Any) -> None:
        for h in list(self._subs.get(topic, [])):
            try:
                h(payload)
            except Exception:
                # Best-effort: keep other handlers alive
                pass


# Simple global bus if desired
BUS = EventBus()


