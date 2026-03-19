from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Tile:
    number: Optional[int] = None          # 1-13, None for joker
    color: Optional[str] = None           # 'red', 'blue', 'black', 'orange', None for joker
    is_joker: bool = False
    position: Tuple[int, int] = (0, 0)    # (x, y) pixel position in image
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h) bounding box
    confidence: float = 0.0
    region: str = 'unknown'               # 'board' or 'rack'

    def __repr__(self) -> str:
        if self.is_joker:
            return "Tile(JOKER)"
        return f"Tile({self.color} {self.number})"

    def to_dict(self) -> dict:
        return {
            'number': self.number,
            'color': self.color,
            'is_joker': self.is_joker,
            'position': self.position,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'region': self.region,
        }
