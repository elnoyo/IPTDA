"""Funciones de dibujo compartidas por los paneles."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from . import style


def split_by_dim(simplices):
    out = {0: [], 1: [], 2: [], 3: []}
    for s in simplices:
        k = len(s) - 1
        if k in out:
            out[k].append(s)
    return out


def setup_axes_for(ax, X, pad=0.12):
    X = np.asarray(X)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = max(float(np.max(hi - lo)), 1e-9)
    c = (lo + hi) / 2
    h = span / 2 * (1 + pad)
    if X.shape[1] == 2:
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_aspect("equal")
    else:
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        ax.set_zlim(c[2] - h, c[2] + h)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:  # pragma: no cover
            pass
        style_3d(ax)


def style_3d(ax):
    """Paneles y rejilla de los ejes 3D acordes con el tema."""
    from matplotlib.colors import to_rgba
    pane = to_rgba(style.T["surface2"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis._axinfo["grid"]["color"] = to_rgba(style.T["grid"])
        axis.line.set_color(style.T["border"])


def draw_complex(ax, X, simplices, *, balls: float | None = None, point_colors=None,
                 highlight=None, highlight_dim: int = 1, active_vertices=None,
                 point_size: float = 16, labels: bool = False, max_triangles: int = 20000,
                 cmap=None, vmin=None, vmax=None):
    """Dibuja un complejo simplicial sobre los puntos ``X`` (2D o 3D).

    ``highlight``: lista de símplices a resaltar (p. ej. un ciclo
    representante). ``active_vertices``: índices de los vértices que forman
    parte del complejo (el resto se dibuja atenuado).
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    S = split_by_dim(simplices)
    tris = S[2][:max_triangles]
    if d == 2:
        if balls is not None and balls > 0:
            verts = range(len(X)) if active_vertices is None else active_vertices
            for i in verts:
                ax.add_patch(Circle(X[i], balls, facecolor=style.BALL_COLOR, alpha=0.08,
                                    edgecolor=style.BALL_COLOR, linewidth=0.4))
        if tris:
            ax.add_collection(PolyCollection([X[list(t)] for t in tris], facecolor=style.TRIANGLE_COLOR,
                                             alpha=0.22, edgecolor="none"))
        if S[1]:
            ax.add_collection(LineCollection([X[list(e)] for e in S[1]], colors=style.EDGE_COLOR,
                                             linewidths=0.8, alpha=0.9))
    else:
        if tris:
            pc = Poly3DCollection([X[list(t)] for t in tris], alpha=0.16,
                                  facecolor=style.TRIANGLE_COLOR, edgecolor="none")
            ax.add_collection3d(pc)
        if S[1]:
            ax.add_collection3d(Line3DCollection([X[list(e)] for e in S[1]], colors=style.EDGE_COLOR,
                                                 linewidths=0.6, alpha=0.7))
    if highlight:
        H = split_by_dim(highlight)
        if highlight_dim == 2 and H[2]:
            if d == 2:
                ax.add_collection(PolyCollection([X[list(t)] for t in H[2]], facecolor=style.HIGHLIGHT,
                                                 alpha=0.45, edgecolor=style.HIGHLIGHT))
            else:
                ax.add_collection3d(Poly3DCollection([X[list(t)] for t in H[2]], alpha=0.5,
                                                     facecolor=style.HIGHLIGHT, edgecolor=style.HIGHLIGHT,
                                                     linewidths=0.4))
        if H[1]:
            segs = [X[list(e)] for e in H[1]]
            if d == 2:
                ax.add_collection(LineCollection(segs, colors=style.HIGHLIGHT, linewidths=3.0, zorder=4))
            else:
                ax.add_collection3d(Line3DCollection(segs, colors=style.HIGHLIGHT, linewidths=3.0))
    # puntos
    if point_colors is None:
        colors = np.array([style.POINT_COLOR] * len(X), dtype=object)
        if active_vertices is not None:
            colors[:] = style.MUTED
            colors[list(active_vertices)] = style.POINT_COLOR
        kw = dict(c=list(colors))
    else:
        kw = dict(c=point_colors, cmap=cmap or style.SEQ_POINTS, vmin=vmin, vmax=vmax)
    if d == 2:
        sc = ax.scatter(X[:, 0], X[:, 1], s=point_size, zorder=5, edgecolors="white",
                        linewidths=0.5, **kw)
    else:
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=point_size * 0.7, depthshade=True, **kw)
    if labels:
        for i, p in enumerate(X):
            if d == 2:
                ax.annotate(str(i), p, textcoords="offset points", xytext=(4, 4), fontsize=8,
                            color=style.TEXT2)
            else:
                ax.text(*p, str(i), fontsize=8, color=style.TEXT2)
    return sc


# ---------------------------------------------------------------------------
# Códigos de barras y diagramas
# ---------------------------------------------------------------------------

def visible_pairs(result, dims, include_zero=False):
    out = []
    for k in dims:
        out.extend(sorted(result.by_dim(k, include_zero), key=lambda p: (p.birth, -p.death)))
    return out


def draw_barcode(ax, pairs, xmax: float, t: float | None = None, selected=None,
                 xmin: float = 0.0, title="Código de barras", xlabel="r"):
    """Dibuja las barras (una fila por barra, agrupadas por dimensión).
    Las barras infinitas terminan en una flecha en ``xmax``."""
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    for row, p in enumerate(pairs):
        d = p.death if np.isfinite(p.death) else xmax
        lw = 3.2 if p is selected else 2.0
        color = style.HIGHLIGHT if p is selected else style.dim_color(p.dim)
        ax.plot([p.birth, d], [row, row], color=color, lw=lw, solid_capstyle="round")
        if not np.isfinite(p.death):
            ax.plot([xmax], [row], marker=">", color=color, markersize=5)
    if t is not None:
        ax.axvline(t, color=style.TEXT, lw=1.0, ls="--")
    ax.set_xlim(xmin, xmax * 1.03 if xmax > 0 else 1)
    ax.set_ylim(len(pairs) - 0.5 if pairs else 1, -0.5)
    ax.set_yticks([])
    # leyenda por dimensión
    dims = sorted({p.dim for p in pairs})
    for k in dims:
        ax.plot([], [], color=style.dim_color(k), lw=3, label=f"H{k}")
    if dims:
        ax.legend(loc="lower right")


def draw_diagram(ax, pairs, vmax: float, t: float | None = None, selected=None,
                 vmin: float = 0.0, title="Diagrama de persistencia", label_axes=("nacimiento", "muerte")):
    """Diagrama de persistencia. Los puntos esenciales se dibujan sobre la
    línea «∞». Si se da ``t``, se sombrea la región de las clases vivas en
    ``t`` (nacimiento <= t < muerte)."""
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(label_axes[0])
    ax.set_ylabel(label_axes[1])
    inf_y = vmin + (vmax - vmin) * 1.08
    lo = vmin - 0.03 * (vmax - vmin)
    hi = inf_y + 0.04 * (vmax - vmin)
    ax.plot([lo, hi], [lo, hi], color=style.MUTED, lw=1)
    ax.axhline(inf_y, color=style.MUTED, lw=0.8, ls=":")
    ax.text(lo, inf_y, " ∞", va="bottom", ha="left", color=style.TEXT2, fontsize=9)
    if t is not None:
        ax.add_patch(Rectangle((lo, t), t - lo, hi - t, facecolor=style.T["accent"], alpha=0.07,
                               edgecolor="none"))
        ax.axvline(t, color=style.TEXT, lw=0.8, ls="--")
        ax.axhline(t, color=style.TEXT, lw=0.8, ls="--")
    dims = sorted({p.dim for p in pairs})
    for k in dims:
        P = [p for p in pairs if p.dim == k]
        x = [p.birth for p in P]
        y = [p.death if np.isfinite(p.death) else inf_y for p in P]
        ax.scatter(x, y, s=30, color=style.dim_color(k), edgecolors="white", linewidths=0.8,
                   label=f"H{k}", zorder=3)
    if selected is not None:
        y = selected.death if np.isfinite(selected.death) else inf_y
        ax.scatter([selected.birth], [y], s=110, facecolors="none", edgecolors=style.HIGHLIGHT,
                   linewidths=2, zorder=4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    if dims:
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor=style.MUTED)
    return inf_y


def pick_diagram_point(event, ax, pairs, inf_y, tol_px=10):
    """Par más cercano a un clic sobre el diagrama (o ``None``)."""
    if event.inaxes is not ax or not pairs:
        return None
    pts = np.array([[p.birth, p.death if np.isfinite(p.death) else inf_y] for p in pairs])
    disp = ax.transData.transform(pts)
    dist = np.hypot(disp[:, 0] - event.x, disp[:, 1] - event.y)
    i = int(np.argmin(dist))
    return pairs[i] if dist[i] <= tol_px else None


def pick_barcode_row(event, ax, pairs):
    if event.inaxes is not ax or not pairs or event.ydata is None:
        return None
    row = int(round(event.ydata))
    if 0 <= row < len(pairs):
        return pairs[row]
    return None
