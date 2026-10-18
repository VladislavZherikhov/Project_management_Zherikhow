"""
Coordinate finder: click on the template to get pixel coordinates.

Usage:
    python coordinate_finder.py templates/passport_template.jpg

Left click  - print (x, y) and mark the point
Right click - remove the last point
Close the window - all points are printed and saved to coordinates.txt
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "templates/passport_template.jpg")
    img = Image.open(path)
    print(f"Template: {path}  size = {img.width} x {img.height}")

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img)
    ax.set_title("Left click: add point | Right click: undo | Close: save")
    points, marks = [], []

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        if event.button == 1:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            points.append((x, y))
            dot, = ax.plot(x, y, "r+", markersize=12)
            label = ax.text(x + 4, y - 4, f"{len(points)}: ({x}, {y})",
                            color="red", fontsize=8)
            marks.append((dot, label))
            print(f"#{len(points)}: ({x}, {y})")
        elif event.button == 3 and points:
            points.pop()
            for artist in marks.pop():
                artist.remove()
            print("Last point removed")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    out = Path("coordinates.txt")
    out.write_text("\n".join(f"{i}: {x}, {y}" for i, (x, y) in enumerate(points, 1)),
                   encoding="utf-8")
    print(f"\n{len(points)} points saved to {out}")


if __name__ == "__main__":
    main()
