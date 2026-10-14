"""Tests del núcleo matemático. Los que comparan con gudhi se omiten si no
está instalado (es solo una dependencia de desarrollo)."""

from itertools import combinations, permutations

import numpy as np
import pytest

from iptda.core import datasets as ds
from iptda.core.distances import bottleneck, wasserstein
from iptda.core.filtration import (alpha_filtration, cech_filtration, lower_star_path,
                                   rips_filtration, static_complex, witness_filtration)
from iptda.core.geometry import (circumspheres, hausdorff_distance, maxmin_subsample,
                                 min_enclosing_ball, min_enclosing_radii)
from iptda.core.homology import homology
from iptda.core.mapper import lens_values, mapper, uniform_cover
from iptda.core.multipers import (codensity, fibered_barcode, hilbert_function, rank_invariant,
                                  rips_codensity_bifiltration)
from iptda.core.persistence import compute_persistence
from iptda.core.timeseries import make_signal, suggest_tau, takens_embedding
from iptda.core.vectorization import betti_curve, landscapes, persistence_image, persistent_entropy

try:
    import gudhi
except ImportError:      # pragma: no cover
    gudhi = None


def sorted_dgm(D, tol=1e-9):
    D = np.asarray(D, dtype=float).reshape(-1, 2)
    D = D[D[:, 1] - D[:, 0] > tol * np.maximum(1, np.abs(D[:, 0]))]
    return D[np.lexsort((D[:, 1], D[:, 0]))]


@pytest.fixture
def circle_pts():
    return ds.DATASETS["Circunferencia"].sample(40, 0.05, 3)


# ----------------------------------------------------------------- datasets
def test_datasets_shapes():
    for name, d in ds.DATASETS.items():
        X = d.sample(50, 0.01, 0)
        assert X.shape == (50, d.dim), name
        assert np.all(np.isfinite(X))


# ----------------------------------------------------------------- geometry
def test_min_enclosing_radius_matches_bruteforce():
    rng = np.random.default_rng(1)
    for d in (2, 3):
        X = rng.normal(size=(12, d))
        for k in (2, 3, 4):
            S = np.array(list(combinations(range(12), k)))[:200]
            r = min_enclosing_radii(X, S)
            r2 = np.array([min_enclosing_ball(X[s])[1] for s in S])
            assert np.allclose(r, r2)


def test_meb_triangle_closed_form():
    rng = np.random.default_rng(2)
    for _ in range(200):
        P = rng.normal(size=(3, 2))
        a, b, c = sorted([np.linalg.norm(P[1] - P[2]), np.linalg.norm(P[0] - P[2]),
                          np.linalg.norm(P[0] - P[1])])
        if c ** 2 >= a ** 2 + b ** 2:
            expected = c / 2
        else:
            u, v = P[1] - P[0], P[2] - P[0]
            expected = a * b * c / (2 * abs(u[0] * v[1] - u[1] * v[0]))
        assert np.isclose(min_enclosing_radii(P, np.array([[0, 1, 2]]))[0], expected)


def test_circumsphere_equidistant():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(20, 3, 3))
    c, r = circumspheres(P)
    for k in range(3):
        assert np.allclose(np.linalg.norm(P[:, k] - c, axis=1), r)


def test_maxmin_and_hausdorff():
    X = np.array([[0, 0], [1, 0], [10, 0], [0.5, 0]])
    assert list(maxmin_subsample(X, 2)) == [0, 2]
    assert hausdorff_distance(X, X) == 0
    assert np.isclose(hausdorff_distance(X[:2], X[:3]), 9.0)


# ---------------------------------------------------------------- filtrations
@pytest.mark.parametrize("builder", [
    lambda X: rips_filtration(X, 2, 0.8),
    lambda X: cech_filtration(X, 2, 0.8),
    lambda X: alpha_filtration(X),
    lambda X: witness_filtration(X, 15, 1, 2)[0],
])
def test_filtrations_are_valid(builder, circle_pts):
    F = builder(circle_pts)
    assert F.check()
    v = F.values
    assert np.all(np.diff(v) >= 0)


def test_cech_inside_rips(circle_pts):
    R = rips_filtration(circle_pts, 2)
    C = cech_filtration(circle_pts, 2)
    rv = dict(zip(R.simplices, R.values))
    for s, v in zip(C.simplices, C.values):
        # Rips(r) ⊇ Čech(r)  y  Rips(r) ⊆ Čech(√2 r)  (en el plano basta 2/√3)
        assert rv[s] <= v + 1e-12
        assert v <= np.sqrt(2) * rv[s] + 1e-12


def test_alpha_equals_cech_persistence(circle_pts):
    """Teorema: las filtraciones alfa y de Čech tienen los mismos diagramas."""
    X = circle_pts[:25]
    PC = compute_persistence(cech_filtration(X, 2))
    PA = compute_persistence(alpha_filtration(X))
    for d in (0, 1):
        assert np.allclose(sorted_dgm(PC.diagram(d)), sorted_dgm(PA.diagram(d)))


@pytest.mark.skipif(gudhi is None, reason="gudhi no instalado")
def test_rips_matches_gudhi():
    X = ds.DATASETS["Ocho (dos círculos tangentes)"].sample(45, 0.05, 7)
    P = compute_persistence(rips_filtration(X, 2))
    st = gudhi.RipsComplex(points=X).create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    for d in (0, 1):
        g = st.persistence_intervals_in_dimension(d) / 2
        assert np.allclose(sorted_dgm(g), sorted_dgm(P.diagram(d)))


@pytest.mark.skipif(gudhi is None, reason="gudhi no instalado")
@pytest.mark.parametrize("name", ["Dos circunferencias", "Esfera", "Toro"])
def test_alpha_matches_gudhi(name):
    X = ds.DATASETS[name].sample(120, 0.0, 5)
    P = compute_persistence(alpha_filtration(X))
    st = gudhi.AlphaComplex(points=X).create_simplex_tree()
    st.compute_persistence()
    for d in range(X.shape[1]):
        g = np.sqrt(st.persistence_intervals_in_dimension(d))
        assert np.allclose(sorted_dgm(g), sorted_dgm(P.diagram(d)), atol=1e-7)


def test_witness_landmark_labels(circle_pts):
    F, L = witness_filtration(circle_pts, 12, 1, 2)
    verts = {s[0] for s in F.simplices if len(s) == 1}
    assert verts == set(L.tolist())


# ---------------------------------------------------------------- persistence
def test_betti_at_matches_direct_homology(circle_pts):
    F = rips_filtration(circle_pts, 2, 0.9)
    P = compute_persistence(F)
    for t in (0.0, 0.05, 0.12, 0.3, 0.5, 0.8):
        K = F.simplices_at(t)
        H = homology(static_complex(K), "Z/2", with_generators=False)
        assert [P.betti_at(t, 0), P.betti_at(t, 1)] == (H.betti + [0])[:2], t


def test_representative_cycles_are_cycles(circle_pts):
    F = rips_filtration(circle_pts, 2, 0.9)
    P = compute_persistence(F, essential_cycles=True)
    for p in P.pairs:
        if p.dim == 1 and p.cycle:
            bnd = {}
            for e in p.cycle:
                for v in e:
                    bnd[v] = bnd.get(v, 0) ^ 1
            assert not any(bnd.values())
            # el ciclo existe cuando nace la clase
            idx = F.index
            assert max(F.values[idx[e]] for e in p.cycle) <= p.birth + 1e-12


def test_circle_has_one_long_h1(circle_pts):
    P = compute_persistence(rips_filtration(circle_pts, 2))
    pers = np.sort(np.diff(P.diagram(1), axis=1).ravel())[::-1]
    assert pers[0] > 0.4 and (len(pers) == 1 or pers[1] < 0.15)
    assert len([p for p in P.pairs if p.dim == 0 and p.is_essential]) == 1


@pytest.mark.parametrize("name", ["Circunferencia con atípicos", "Toro"])
def test_cohomology_equals_homology(name):
    """La reducción cofrontera (sin representantes) da los mismos pares."""
    X = ds.DATASETS[name].sample(45, 0.03, 2)
    F = rips_filtration(X, 3 if X.shape[1] == 3 else 2, 1.2)
    A = compute_persistence(F, cycles=False)
    B = compute_persistence(F, cycles=True)
    key = lambda P: sorted((p.dim, p.birth_index, p.death_index if p.death_index is not None else -1)
                           for p in P.pairs)
    assert key(A) == key(B)


def test_alpha_flat_cloud_in_3d():
    """Una curva plana en R³ (inmersión de un seno con d = 3) no admite
    Delaunay 3D; la filtración alfa se calcula en su envolvente afín."""
    _, x = make_signal("Seno", 300, 0.0, 0, period=40)
    E = takens_embedding(x, 3, 10, 2)
    P = compute_persistence(alpha_filtration(E))
    Q = compute_persistence(alpha_filtration(E @ np.linalg.qr(np.random.default_rng(0)
                                                               .normal(size=(3, 3)))[0]))
    assert np.allclose(sorted_dgm(P.diagram(1)), sorted_dgm(Q.diagram(1)))
    assert np.max(np.diff(P.diagram(1), axis=1)) > 0.5


def test_alpha_with_duplicate_points():
    X = ds.DATASETS["Circunferencia"].sample(30, 0.02, 4)
    Xd = np.vstack([X, X[:7], X[:3] + 1e-14])
    F = alpha_filtration(Xd)
    assert F.check()
    assert {s[0] for s in F.simplices if len(s) == 1} == set(range(len(Xd)))
    P, Q = compute_persistence(alpha_filtration(X)), compute_persistence(F)
    for d in (0, 1):
        assert np.allclose(sorted_dgm(P.diagram(d)), sorted_dgm(Q.diagram(d)))


def _dim_filtration(entry):
    from iptda.core.filtration import Filtration
    F0 = entry.filtration()
    return Filtration(list(F0.simplices), np.array([len(s) - 1 for s in F0.simplices], float),
                      reliable_dim=2)


@pytest.mark.parametrize("cycles", [False, True])
def test_persistence_over_fields(cycles):
    """RP² y la botella de Klein: los diagramas dependen del cuerpo."""
    from iptda.core.complexes_library import LIBRARY
    rp2 = _dim_filtration(LIBRARY["Plano proyectivo RP² (polígono fundamental)"])
    klein = _dim_filtration(LIBRARY["Botella de Klein (polígono fundamental)"])
    expected = {("rp2", 2): [1, 1, 1], ("rp2", 3): [1, 0, 0], ("rp2", 5): [1, 0, 0],
                ("klein", 2): [1, 2, 1], ("klein", 3): [1, 1, 0], ("klein", 7): [1, 1, 0]}
    for (name, p), betti in expected.items():
        F = rp2 if name == "rp2" else klein
        P = compute_persistence(F, field=p, cycles=cycles, essential_cycles=cycles)
        assert [P.betti_at(2.0, k) for k in range(3)] == betti, (name, p)
        assert P.field == p


def test_persistence_mod_p_matches_homology_ranks():
    """β_k(K_t) con coeficientes en Z/3 igual a la homología directa."""
    from iptda.core.complexes_library import LIBRARY
    from iptda.core.filtration import Filtration
    F0 = LIBRARY["Plano proyectivo RP² (polígono fundamental)"].filtration()
    rng = np.random.default_rng(0)
    vals = {}
    for s in sorted(F0.simplices, key=len):          # valores crecientes por caras
        lo = max((vals[s[:i] + s[i + 1:]] for i in range(len(s))), default=0) if len(s) > 1 else 0
        vals[s] = lo + rng.uniform(0, 1)
    F = Filtration(list(vals), np.array(list(vals.values())), reliable_dim=2)
    P = compute_persistence(F, field=3)
    for t in np.linspace(0, F.values.max(), 9):
        K = F.simplices_at(t)
        H = homology(static_complex(K), "Z/3", with_generators=False)
        assert [P.betti_at(t, k) for k in range(len(H.betti))] == H.betti


def test_mod_p_cycles_are_cycles():
    X = ds.DATASETS["Ocho (dos círculos tangentes)"].sample(35, 0.04, 3)
    P = compute_persistence(rips_filtration(X, 2), field=5, essential_cycles=True)
    for q in P.pairs:
        if q.dim == 1 and q.cycle:
            bnd = {}
            for (a, b), c in zip(q.cycle, q.coefficients):
                bnd[b] = (bnd.get(b, 0) + c) % 5          # ∂[a, b] = b − a
                bnd[a] = (bnd.get(a, 0) - c) % 5
            assert not any(bnd.values())


@pytest.mark.skipif(gudhi is None, reason="gudhi no instalado")
def test_mod_p_matches_gudhi():
    X = ds.DATASETS["Dos circunferencias"].sample(40, 0.05, 2)
    F = rips_filtration(X, 2)
    for p in (3, 5):
        P = compute_persistence(F, field=p, cycles=False)
        st = gudhi.RipsComplex(points=X).create_simplex_tree(max_dimension=2)
        st.compute_persistence(homology_coeff_field=p)
        for d in (0, 1):
            g = st.persistence_intervals_in_dimension(d) / 2
            assert np.allclose(sorted_dgm(g), sorted_dgm(P.diagram(d)))


def test_non_prime_field_rejected():
    with pytest.raises(ValueError):
        compute_persistence(rips_filtration(np.eye(3), 2), field=4)


def test_lower_star_path():
    f = np.array([0.0, 2.0, 1.0, 3.0, -1.0])
    P = compute_persistence(lower_star_path(f), max_dim=0)
    D = sorted_dgm(P.diagram(0))
    assert np.allclose(D, [[-1, np.inf], [0, 3], [1, 2]])


# ---------------------------------------------------------------- homology
def _is_closed_surface(maximal):
    tris = [tuple(sorted(t)) for t in maximal]
    if len(set(tris)) != len(tris) or any(len(set(t)) != 3 for t in tris):
        return False
    verts = {v for t in tris for v in t}
    for v in verts:
        link = [tuple(x for x in t if x != v) for t in tris if v in t]
        deg = {}
        for a, b in link:
            deg[a] = deg.get(a, 0) + 1
            deg[b] = deg.get(b, 0) + 1
        if any(d != 2 for d in deg.values()):
            return False
        # el enlace debe ser un único ciclo
        adj = {a: set() for a in deg}
        for a, b in link:
            adj[a].add(b)
            adj[b].add(a)
        seen, stack = set(), [next(iter(adj))]
        while stack:
            a = stack.pop()
            if a not in seen:
                seen.add(a)
                stack.extend(adj[a])
        if seen != set(adj):
            return False
    return True


def test_library_betti_numbers():
    from iptda.core.complexes_library import LIBRARY
    for name, c in LIBRARY.items():
        F = c.filtration()
        for field, expected in c.expected.items():
            H = homology(F, field, with_generators=False)
            assert H.betti == expected, (name, field)
            assert H.dd_zero
            assert H.euler == sum((-1) ** k * b for k, b in enumerate(H.betti))


def test_library_closed_surfaces_are_valid():
    from iptda.core.complexes_library import LIBRARY
    for name in ["Toro (en R³)", "Toro (polígono fundamental)",
                 "Botella de Klein (polígono fundamental)",
                 "Plano proyectivo RP² (polígono fundamental)",
                 "Borde del tetraedro (esfera S²)", "Octaedro (esfera S²)"]:
        assert _is_closed_surface(LIBRARY[name].abstract_maximal()), name


def test_generators_count_z2():
    from iptda.core.complexes_library import LIBRARY
    H = homology(LIBRARY["Toro (polígono fundamental)"].filtration(), "Z/2")
    assert [len(g) for g in H.generators] == H.betti


def test_rank_over_q_exact():
    from iptda.core.homology import rank_rational, rank_mod_p
    M = np.array([[2, 4, 6], [1, 2, 3], [0, 0, 5]])
    assert rank_rational(M) == 2
    assert rank_mod_p(M, 2) == 2
    assert rank_mod_p(M, 5) == 1


# ---------------------------------------------------------------- distances
def _brute_bottleneck(A, B):
    """Fuerza bruta para diagramas diminutos."""
    A = [tuple(p) for p in A]
    B = [tuple(p) for p in B]
    a_ext = A + [None] * len(B)
    b_ext = B + [None] * len(A)
    best = np.inf
    for perm in permutations(range(len(b_ext))):
        cost = 0.0
        for i, j in enumerate(perm):
            x, y = a_ext[i], b_ext[j]
            if x is None and y is None:
                c = 0.0
            elif x is None:
                c = (y[1] - y[0]) / 2
            elif y is None:
                c = (x[1] - x[0]) / 2
            else:
                c = max(abs(x[0] - y[0]), abs(x[1] - y[1]))
            cost = max(cost, c)
        best = min(best, cost)
    return best


def test_bottleneck_bruteforce():
    rng = np.random.default_rng(4)
    for _ in range(20):
        A = rng.uniform(0, 1, (3, 2)); A[:, 1] = A[:, 0] + rng.uniform(0, 1, 3)
        B = rng.uniform(0, 1, (2, 2)); B[:, 1] = B[:, 0] + rng.uniform(0, 1, 2)
        assert np.isclose(bottleneck(A, B).value, _brute_bottleneck(A, B))


def test_distances_properties():
    A = np.array([[0, 1], [0.2, 0.5], [0, np.inf]])
    B = np.array([[0.1, 1.1], [0.3, np.inf]])
    assert bottleneck(A, A).value == 0
    assert np.isclose(bottleneck(A, B).value, max(0.1, 0.15, 0.3))
    assert np.isinf(bottleneck(A, B[:1]).value)
    assert np.isclose(wasserstein(A, B, 1).value, 0.1 + 0.15 + 0.3)


@pytest.mark.skipif(gudhi is None, reason="gudhi no instalado")
def test_bottleneck_matches_gudhi():
    rng = np.random.default_rng(5)
    A = rng.uniform(0, 1, (30, 2)); A[:, 1] = A[:, 0] + rng.uniform(0, 0.6, 30)
    B = rng.uniform(0, 1, (25, 2)); B[:, 1] = B[:, 0] + rng.uniform(0, 0.6, 25)
    assert np.isclose(bottleneck(A, B).value, gudhi.bottleneck_distance(A, B))


def test_stability_rips():
    """d_B ≤ ‖f − g‖∞ ≤ máximo desplazamiento (convención de radio)."""
    rng = np.random.default_rng(6)
    X = ds.DATASETS["Circunferencia"].sample(35, 0.0, 1)
    delta = 0.05
    Y = X + rng.uniform(-delta, delta, X.shape) / np.sqrt(2)
    disp = np.max(np.linalg.norm(X - Y, axis=1))
    PX = compute_persistence(rips_filtration(X, 2))
    PY = compute_persistence(rips_filtration(Y, 2))
    for d in (0, 1):
        assert bottleneck(PX.diagram(d), PY.diagram(d)).value <= disp + 1e-12


# ---------------------------------------------------------------- vectorizations
def test_vectorizations():
    D = np.array([[0.0, 1.0], [0.5, 0.7]])
    ts = np.array([0.0, 0.25, 0.5, 0.6, 1.0])
    assert list(betti_curve(D, ts)) == [1, 1, 2, 2, 0]
    L = landscapes(D, ts, 2)
    assert np.allclose(L[0], [0, 0.25, 0.5, 0.4, 0])
    assert np.allclose(L[1], [0, 0, 0, 0.1, 0])
    img = persistence_image(D, (-1, 2), (0, 2), 30, 0.05)
    assert np.isclose(img.sum(), 1.0 + 0.2, atol=1e-3)
    assert np.isclose(persistent_entropy(np.array([[0, 1], [0, 1]])), np.log(2))


# ---------------------------------------------------------------- time series
def test_takens_sine_periodic():
    _, x = make_signal("Seno", 400, 0.0, 0, period=40)
    tau = suggest_tau(x)
    assert tau in (10, 11)      # cuarto de periodo
    E = takens_embedding(x, 2, tau, 4)
    assert np.allclose(E[0], [x[0], x[tau]])
    P = compute_persistence(rips_filtration(E, 2))
    assert np.max(np.diff(P.diagram(1), axis=1)) > 0.5


# ---------------------------------------------------------------- mapper
def test_cover():
    I = uniform_cover(np.array([0.0, 10.0]), 4, 0.25)
    L = I[0][1] - I[0][0]
    assert np.isclose(I[0][0], 0) and np.isclose(I[-1][1], 10)
    assert np.isclose(I[1][0], I[0][0] + 0.75 * L)


def test_mapper_circle_has_a_cycle():
    X = ds.DATASETS["Circunferencia"].sample(300, 0.02, 0)
    G = mapper(X, lens_values(X, "Coordenada x"), 6, 0.4, "Enlace simple", 0.2)
    assert G.betti == (1, 1)
    Y = ds.DATASETS["Letra Y (árbol)"].sample(300, 0.01, 0)
    G2 = mapper(Y, lens_values(Y, "Coordenada y"), 8, 0.3, "DBSCAN", 0.15, 3)
    assert G2.betti[1] == 0


# ---------------------------------------------------------------- multiparameter
def test_mapper_2d_lens():
    X = ds.DATASETS["Esfera"].sample(800, 0.0, 0)
    f = np.column_stack([lens_values(X, "Coordenada x"), lens_values(X, "Coordenada y")])
    G = mapper(X, f, 4, 0.5, "Enlace simple", 0.35)
    assert G.lens_dim == 2 and len(G.cells) == 16
    # cada nodo está contenido en su celda
    for pts, k in zip(G.nodes, G.node_cell):
        assert set(pts.tolist()) <= set(G.cell_members(k).tolist())
    # los triángulos son triples de nodos con un punto común
    sets = [set(m.tolist()) for m in G.nodes]
    for a, b, c in G.triangles:
        assert sets[a] & sets[b] & sets[c]
    assert G.triangles
    nb = G.nerve_betti()
    assert nb[0] == G.betti[0] and nb[1] <= G.betti[1]


def test_bifiltration_invariants():
    X = ds.DATASETS["Circunferencia con atípicos"].sample(45, 0.02, 1)
    cod = codensity(X, "Distancia al k-ésimo vecino", 4)
    B = rips_codensity_bifiltration(X, cod, 2, 0.9)
    rg = np.linspace(0, 0.9, 10)
    cg = np.linspace(cod.min(), cod.max(), 8)
    HF = hilbert_function(B, rg, cg)
    # Comparación directa con la homología de K(r, c) en algunos puntos
    for i, j in [(2, 3), (5, 7), (9, 0), (4, 4)]:
        K = B.complex_at(rg[i], cg[j])
        if K:
            H = homology(static_complex(K), "Z/2", with_generators=False)
            b = H.betti + [0, 0]
            assert HF[0][j, i] == b[0] and HF[1][j, i] == b[1]
    # código fibrado horizontal = Rips del subconjunto con codensidad <= c
    c0 = cg[5]
    P = fibered_barcode(B, (0.0, c0), (1.0, 0.0), 1)
    keep = np.nonzero(cod <= c0)[0]
    Q = compute_persistence(rips_filtration(X[keep], 2, 0.9))
    for d in (0, 1):
        assert np.allclose(sorted_dgm(P.diagram(d)), sorted_dgm(Q.diagram(d)))
    # invariante de rango: rk(a→a) = función de Hilbert en a
    a = (rg[6], cg[6])
    assert rank_invariant(B, a, a, 1) == HF[1][6, 6]
    assert rank_invariant(B, a, a, 0) == HF[0][6, 6]
    # y rk(a→b) <= min(HF(a), HF(b))
    b = (rg[8], cg[7])
    assert rank_invariant(B, a, b, 1) <= min(HF[1][6, 6], HF[1][7, 8])
