"""Find and capture a specific app window on macOS."""
import Quartz


def list_windows(keyword=None):
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID
    )
    for w in windows:
        name = w.get("kCGWindowOwnerName", "")
        title = w.get("kCGWindowName", "")
        bounds = w.get("kCGWindowBounds", {})
        layer = w.get("kCGWindowLayer", 0)

        if keyword:
            if keyword.lower() not in name.lower() and keyword.lower() not in str(title).lower():
                continue

        x = bounds.get("X", 0)
        y = bounds.get("Y", 0)
        width = bounds.get("Width", 0)
        height = bounds.get("Height", 0)

        print(f"[{name}] '{title}' | layer={layer} | x={x}, y={y}, w={width}, h={height}")


if __name__ == "__main__":
    print("=== iPhone Mirroring windows ===")
    list_windows("iPhone")
    print("\n=== All windows (first 20) ===")
    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID
    )
    for i, w in enumerate(windows[:20]):
        name = w.get("kCGWindowOwnerName", "")
        title = w.get("kCGWindowName", "")
        bounds = w.get("kCGWindowBounds", {})
        wid = w.get("kCGWindowNumber", 0)
        x = bounds.get("X", 0)
        y = bounds.get("Y", 0)
        width = bounds.get("Width", 0)
        height = bounds.get("Height", 0)
        print(f"  [{name}] '{title}' id={wid} | {width}x{height} at ({x},{y})")
