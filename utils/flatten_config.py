from typing import Iterator


def flatten(item, name="", path="", is_root=True) -> Iterator[tuple[str, str]]:
    """Convert dataclass to flatten dict."""
    if isinstance(item, dict):
        for k, v in item.items():
            yield from flatten(v, k, (path + name + "_") if not is_root else name, False)
    elif isinstance(item, list):
        for i, v in enumerate(item):
            yield from flatten(v, str(i), path + name + "_", False)
    else:
        yield path + name, item
