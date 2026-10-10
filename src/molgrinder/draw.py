from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def _rgb01_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(round(r * 255)))),
        max(0, min(255, int(round(g * 255)))),
        max(0, min(255, int(round(b * 255))))
    )


def _text_color_for_bg(rgb):
    # simple luminance heuristic
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if lum > 0.6 else "#ffffff"


def draw_2d(
    mol: Chem.Mol,
    groups: list[list[int]],
    bead_types: list[str],
    size: tuple[int, int] = (500, 500),
    palette: list[tuple[float, float, float]] | None = None,
    highlight_bonds: bool = True,
    remove_hs: bool = True,
    label_font_size_px: int = 16,
    label_pad_px: int = 4,
    label_margin_px: int = 6,          # distance above the group
    label_bg_opacity: float = 0.85,
    label_border_px: int = 1,
):
    """
    Draw 2D molecule + group highlights, then add per-group labels as colored SVG rectangles
    placed ABOVE the group (near the group's top), clamped inside the canvas.

    Returns SVG string.
    """
    if len(groups) != len(bead_types):
        raise ValueError(f"len(groups)={len(groups)} must equal len(bead_types)={len(bead_types)}")

    dm = Chem.Mol(mol)
    if remove_hs:
        dm = Chem.RemoveHs(dm)

    # Better 2D coords if CoordGen is available
    try:
        rdDepictor.SetPreferCoordGen(True)
    except Exception:
        pass
    rdDepictor.Compute2DCoords(dm)

    if palette is None:
        palette = [
            (0.90, 0.10, 0.10),
            (0.10, 0.55, 0.95),
            (0.10, 0.75, 0.25),
            (0.85, 0.55, 0.10),
            (0.60, 0.20, 0.80),
            (0.10, 0.75, 0.75),
            (0.55, 0.55, 0.55),
            (0.95, 0.25, 0.65),
        ]

    w, h = size
    d = rdMolDraw2D.MolDraw2DSVG(w, h)

    n_atoms = dm.GetNumAtoms()

    # atom colors / highlights
    atom_colors = {}
    atoms_to_highlight = set()
    for gi, grp in enumerate(groups):
        color = palette[gi % len(palette)]
        for a in grp:
            if not (0 <= a < n_atoms):
                raise IndexError(f"Atom index {a} out of range (0..{n_atoms-1}) for drawn molecule")
            atom_colors[a] = color
            atoms_to_highlight.add(a)

    # bond highlights (within each group)
    bonds_to_highlight = []
    bond_colors = {}
    if highlight_bonds:
        for gi, grp in enumerate(groups):
            color = palette[gi % len(palette)]
            grp_set = set(grp)
            for bnd in dm.GetBonds():
                a1, a2 = bnd.GetBeginAtomIdx(), bnd.GetEndAtomIdx()
                if a1 in grp_set and a2 in grp_set:
                    bid = bnd.GetIdx()
                    bonds_to_highlight.append(bid)
                    bond_colors[bid] = color

    # Draw the structure
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        dm,
        highlightAtoms=list(atoms_to_highlight),
        highlightAtomColors=atom_colors,
        highlightBonds=bonds_to_highlight,
        highlightBondColors=bond_colors,
    )

    # --- compute label placements in DRAW (pixel) coordinates ---
    # In SVG: y increases downward. "Above group" => smaller y than the group's top.
    placed_rects = []  # list of (x0,y0,x1,y1) for overlap avoidance
    label_svgs = []

    for gi, (grp, label) in enumerate(zip(groups, bead_types)):
        if not grp:
            continue

        # compute centroid x and top y in draw coords
        xs, ys = [], []
        for a in grp:
            p = d.GetDrawCoords(a)  # pixel coords
            xs.append(float(p.x))
            ys.append(float(p.y))

        x_c = sum(xs) / len(xs)
        y_top = min(ys)

        label = str(label)

        # approximate text box size in px (good enough for most fonts)
        text_w = int(round(label_font_size_px * 0.62 * len(label)))
        text_h = int(round(label_font_size_px * 1.10))
        rect_w = text_w + 2 * label_pad_px
        rect_h = text_h + 2 * label_pad_px

        # initial position: centered above the top of the group
        x0 = x_c - rect_w / 2
        y0 = y_top - label_margin_px - rect_h

        # clamp within canvas horizontally
        if x0 < 2:
            x0 = 2
        if x0 + rect_w > w - 2:
            x0 = (w - 2) - rect_w

        # clamp within canvas vertically (stay inside)
        if y0 < 2:
            y0 = 2

        rect = (x0, y0, x0 + rect_w, y0 + rect_h)
        placed_rects.append(rect)

        bg_rgb = palette[gi % len(palette)]
        bg_hex = _rgb01_to_hex(bg_rgb)
        txt_hex = _text_color_for_bg(bg_rgb)

        # SVG: text baseline needs adjustment -> use dominant-baseline="middle"
        # place text at rect center
        tx = rect[0] + rect_w / 2
        ty = rect[1] + rect_h / 2

        label_svgs.append(
            f'<rect x="{rect[0]:.2f}" y="{rect[1]:.2f}" width="{rect_w:.2f}" height="{rect_h:.2f}" '
            f'rx="3" ry="3" fill="{bg_hex}" fill-opacity="{label_bg_opacity}" '
            f'stroke="#000000" stroke-width="{label_border_px}"/>'
        )
        label_svgs.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{label_font_size_px}px" font-family="DejaVu Sans, Arial, sans-serif" '
            f'fill="{txt_hex}">{label}</text>'
        )

    d.FinishDrawing()
    svg = d.GetDrawingText()

    # Inject our label SVG just before closing </svg>
    insert = "\n".join(label_svgs) + "\n"
    if "</svg>" in svg:
        svg = svg.replace("</svg>", insert + "</svg>")
    else:
        # very defensive fallback
        svg += insert

    return svg