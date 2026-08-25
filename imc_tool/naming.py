import re

# "{alloy}_{aging_hours}_{idx}.jpg", optionally with a literal "f" tag before idx,
# e.g. "100_1000_2.jpg", "0_0_20.jpg", "50_200_f_16.jpg".
_PATTERN = re.compile(r"^(?P<alloy>\d+)_(?P<aging>\d+)_(?:f_)?(?P<idx>\d+)$")


def parse_condition(stem):
    m = _PATTERN.match(stem)
    if not m:
        return {"alloy": None, "aging_hours": None, "idx": None}
    return {
        "alloy": m.group("alloy"),
        "aging_hours": int(m.group("aging")),
        "idx": int(m.group("idx")),
    }
