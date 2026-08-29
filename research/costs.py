from dataclasses import dataclass


@dataclass
class CostModel:
    taker_fee: float = 0.0004
    slippage_bps: float = 1.0
    use_funding: bool = True

    def slippage_frac(self) -> float:
        return self.slippage_bps / 10000.0

    def fill_price(self, raw_price: float, side: int) -> float:
        slip = self.slippage_frac()
        if side > 0:
            return raw_price * (1.0 + slip)
        if side < 0:
            return raw_price * (1.0 - slip)
        return raw_price

    def round_trip_fee(self) -> float:
        return 2.0 * self.taker_fee
