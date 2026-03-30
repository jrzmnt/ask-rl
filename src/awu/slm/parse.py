ACTION_TO_INT = {
    "LEFT": 0,
    "DOWN": 1,
    "RIGHT": 2,
    "UP": 3,
}


def parse_action(text: str, strategy: str = "plain") -> int | None:
    if not text:
        return None

    text = text.strip().upper()

    if text in {"UP", "DOWN", "LEFT", "RIGHT"}:
        return ACTION_TO_INT[text]

    return None

