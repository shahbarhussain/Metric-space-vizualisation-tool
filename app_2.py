import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Metric Space Visualizer", layout="wide")

# Core math — distance functions and ball-boundary generators

def minkowski_distance(p1, p2, p):
    """d_p(x, y) for two points in R^n. Use p=np.inf for the Chebyshev distance."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    diff = np.abs(p1 - p2)
    if p == np.inf:
        return diff.max()
    return (diff ** p).sum() ** (1 / p)


def minkowski_ball_2d(center, R, p, n_points=300):
    """Boundary (x, y) of B(center, R) under d_p in R^2, traced by angle."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    if p == np.inf:
        norm = np.maximum(np.abs(cos_t), np.abs(sin_t))
    else:
        norm = (np.abs(cos_t) ** p + np.abs(sin_t) ** p) ** (1 / p)
    norm = np.where(norm == 0, 1e-12, norm)
    scale = R / norm
    x = center[0] + scale * cos_t
    y = center[1] + scale * sin_t
    return x, y


def minkowski_ball_3d(center, R, p, n_theta=60, n_phi=60):
    """Surface (x, y, z) grids of B(center, R) under d_p in R^3, via spherical params."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta, phi = np.meshgrid(theta, phi)

    dx = np.sin(theta) * np.cos(phi)
    dy = np.sin(theta) * np.sin(phi)
    dz = np.cos(theta)

    if p == np.inf:
        norm = np.maximum(np.maximum(np.abs(dx), np.abs(dy)), np.abs(dz))
    else:
        norm = (np.abs(dx) ** p + np.abs(dy) ** p + np.abs(dz) ** p) ** (1 / p)
    norm = np.where(norm == 0, 1e-12, norm)
    scale = R / norm

    x = center[0] + scale * dx
    y = center[1] + scale * dy
    z = center[2] + scale * dz
    return x, y, z


P_LABELS = {1: "d1 (Manhattan / L1)", 2: "d2 (Euclidean / L2)", np.inf: "d∞ (Chebyshev / max)"}


def p_label(p):
    return P_LABELS.get(p, f"p = {p}")


def to_p(p_choice):
    return np.inf if p_choice == "∞" else p_choice


def dp_metric_latex(p, dim=2):
    """General equation for d_p(x, y) in R^dim, e.g. d1, d2, d3(=dinf label), or d_p."""
    xs = [f"x_{i + 1}" for i in range(dim)]
    ys = [f"y_{i + 1}" for i in range(dim)]
    if p == np.inf:
        parts = ", ".join(f"|{x}-{y}|" for x, y in zip(xs, ys))
        return rf"d_\infty(x,y) = \max\left\{{{parts}\right\}}"
    if p == 1:
        parts = " + ".join(f"|{x}-{y}|" for x, y in zip(xs, ys))
        return rf"d_1(x,y) = {parts}"
    if p == 2:
        parts = " + ".join(f"({x}-{y})^2" for x, y in zip(xs, ys))
        return rf"d_2(x,y) = \sqrt{{{parts}}}"
    parts = " + ".join(f"|{x}-{y}|^{{{p}}}" for x, y in zip(xs, ys))
    return rf"d_{{{p}}}(x,y) = \left({parts}\right)^{{1/{p}}}"


def ball_metric_latex(p):
    """General set-builder definition of B(a, R) under d_p, e.g. B(a,R) = {x : d_p(x,a) < R}."""
    if p == 1:
        d_sym = "d_1"
    elif p == 2:
        d_sym = "d_2"
    elif p == np.inf:
        d_sym = r"d_\infty"
    else:
        d_sym = f"d_{{{p}}}"
    return rf"B(a,R) = \{{\, x : {d_sym}(x,a) < R \,\}}"


def shape_facts(p, R, dim=2):
    if dim == 2:
        if p == 1:
            return [
                ("Shape", "Rhombus (diagonals along the axes)"),
                ("Horizontal diagonal", f"2R = {2*R:.3f}"),
                ("Vertical diagonal", f"2R = {2*R:.3f}"),
                ("Sum of diagonals", f"4R = {4*R:.3f}"),
                ("Side length", f"R√2 = {R*np.sqrt(2):.3f}"),
                ("Perimeter", f"4√2·R = {4*np.sqrt(2)*R:.3f}"),
            ]
        elif p == 2:
            return [
                ("Shape", "Circle"),
                ("Diameter", f"2R = {2*R:.3f}"),
                ("Circumference", f"2πR = {2*np.pi*R:.3f}"),
            ]
        elif p == np.inf:
            return [
                ("Shape", "Square (sides parallel to axes)"),
                ("Side length", f"2R = {2*R:.3f}"),
                ("Perimeter", f"8R = {8*R:.3f}"),
                ("Each diagonal", f"2√2·R = {2*np.sqrt(2)*R:.3f}"),
                ("Sum of diagonals", f"4√2·R = {4*np.sqrt(2)*R:.3f}"),
            ]
        else:
            x, y = minkowski_ball_2d((0, 0), R, p, n_points=2000)
            seg = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            return [
                ("Shape", f"Lp unit-ball boundary, p = {p}"),
                ("Perimeter (numerical)", f"{seg.sum():.3f}"),
            ]
    else:  # dim == 3
        if p == 1:
            return [
                ("Shape", "Regular octahedron (vertices on the axes)"),
                ("Each axis diagonal", f"2R = {2*R:.3f}"),
                ("Sum of diagonals", f"6R = {6*R:.3f}"),
                ("Edge length", f"R√2 = {R*np.sqrt(2):.3f}"),
                ("Volume", f"(4/3)R³ = {(4/3)*R**3:.3f}"),
            ]
        elif p == 2:
            return [
                ("Shape", "Sphere"),
                ("Diameter", f"2R = {2*R:.3f}"),
                ("Surface area", f"4πR² = {4*np.pi*R**2:.3f}"),
                ("Volume", f"(4/3)πR³ = {(4/3)*np.pi*R**3:.3f}"),
            ]
        elif p == np.inf:
            return [
                ("Shape", "Cube (sides parallel to axes)"),
                ("Side length", f"2R = {2*R:.3f}"),
                ("Surface area", f"24R² = {24*R**2:.3f}"),
                ("Volume", f"8R³ = {8*R**3:.3f}"),
                ("Each space diagonal", f"2√3·R = {2*np.sqrt(3)*R:.3f}"),
            ]
        else:
            return [
                ("Shape", f"Lp unit-ball surface, p = {p}"),
                ("Note", "No closed form in your notes for general p in 3D"),
            ]


def show_shape_facts(p, R, dim=2):
    st.markdown("**Shape geometry:**")
    for label, value in shape_facts(p, R, dim):
        st.markdown(f"- **{label}:** {value}")


def add_shape_annotations_2d(fig, center, R, p, color=None):
    a1, a2 = center
    line_color = color if color else "gray"
    dash_style = dict(color=line_color, dash="dot", width=2)

    if p == 1:
        # horizontal diagonal: length 2R
        fig.add_trace(go.Scatter(
            x=[a1 - R, a1 + R], y=[a2, a2], mode="lines+text",
            line=dash_style, text=["", "2R"], textposition="top center",
            showlegend=False,
        ))
        # vertical diagonal: length 2R -> together, sum of diagonals = 4R
        fig.add_trace(go.Scatter(
            x=[a1, a1], y=[a2 - R, a2 + R], mode="lines+text",
            line=dash_style, text=["", "2R"], textposition="middle right",
            showlegend=False,
        ))
    elif p == 2:
        # radius line from center to boundary: length R
        fig.add_trace(go.Scatter(
            x=[a1, a1 + R], y=[a2, a2], mode="lines+text",
            line=dash_style, text=["", "R"], textposition="top center",
            showlegend=False,
        ))
    elif p == np.inf:
        # main diagonal of the square: length 2*sqrt(2)*R
        fig.add_trace(go.Scatter(
            x=[a1 - R, a1 + R], y=[a2 - R, a2 + R], mode="lines+text",
            line=dash_style, text=["", "2√2·R"], textposition="top center",
            showlegend=False,
        ))
        # top side of the square: length 2R
        fig.add_trace(go.Scatter(
            x=[a1 - R, a1 + R], y=[a2 + R, a2 + R], mode="lines+text",
            line=dict(color=line_color, dash="dash", width=2),
            text=["", "2R"], textposition="top center",
            showlegend=False,
        ))
    else:
        # no closed form for general p -> just show the radius to one boundary
        # point (theta=0), labeled with its numerical length
        bx, by = minkowski_ball_2d(center, R, p, n_points=2)
        fig.add_trace(go.Scatter(
            x=[a1, bx[0]], y=[a2, by[0]], mode="lines+text",
            line=dash_style, text=["", f"R = {R:.2f}"], textposition="top center",
            showlegend=False,
        ))


def add_shape_annotations_3d(fig, center, R, p):
    """3D analogue of add_shape_annotations_2d — axis diagonals, radius, or cube diagonal."""
    a1, a2, a3 = center
    dash_style = dict(color="gray", dash="dot", width=4)

    if p == 1:
        # three axis diagonals of the octahedron, each length 2R (sum = 6R)
        fig.add_trace(go.Scatter3d(x=[a1 - R, a1 + R], y=[a2, a2], z=[a3, a3],
                                    mode="lines+text", line=dash_style,
                                    text=["", "2R"], showlegend=False))
        fig.add_trace(go.Scatter3d(x=[a1, a1], y=[a2 - R, a2 + R], z=[a3, a3],
                                    mode="lines+text", line=dash_style,
                                    text=["", "2R"], showlegend=False))
        fig.add_trace(go.Scatter3d(x=[a1, a1], y=[a2, a2], z=[a3 - R, a3 + R],
                                    mode="lines+text", line=dash_style,
                                    text=["", "2R"], showlegend=False))
    elif p == 2:
        # radius line from center to boundary: length R
        fig.add_trace(go.Scatter3d(x=[a1, a1 + R], y=[a2, a2], z=[a3, a3],
                                    mode="lines+text", line=dash_style,
                                    text=["", "R"], showlegend=False))
    elif p == np.inf:
        # main space diagonal of the cube: length 2*sqrt(3)*R
        fig.add_trace(go.Scatter3d(x=[a1 - R, a1 + R], y=[a2 - R, a2 + R], z=[a3 - R, a3 + R],
                                    mode="lines+text", line=dash_style,
                                    text=["", "2√3·R"], showlegend=False))
        # one edge of the cube: length 2R
        fig.add_trace(go.Scatter3d(x=[a1 - R, a1 + R], y=[a2 + R, a2 + R], z=[a3 + R, a3 + R],
                                    mode="lines+text", line=dict(color="darkgreen", dash="dot", width=4),
                                    text=["", "2R"], showlegend=False))
    else:
        fig.add_trace(go.Scatter3d(x=[a1, a1 + R], y=[a2, a2], z=[a3, a3],
                                    mode="lines+text", line=dash_style,
                                    text=["", f"R = {R:.2f}"], showlegend=False))

# App layout

st.title(" Metric Space Visualizer")
st.caption("Interactive companion to unit balls B(a, R) under different metrics on Rⁿ")

mode = st.sidebar.radio(
    "Choose a view",
    ["2D Ball Explorer", "2D Containment Overlay", "3D Ball Explorer", "Distance Calculator"],
)

# ------------------------------------------------------------
if mode == "2D Ball Explorer":
    st.sidebar.subheader("Ball parameters")
    a1 = st.sidebar.slider("Center a₁", -5.0, 5.0, 0.0, 0.1)
    a2 = st.sidebar.slider("Center a₂", -5.0, 5.0, 0.0, 0.1)
    R = st.sidebar.slider("Radius R", 0.1, 5.0, 1.0, 0.1)
    p_choice = st.sidebar.select_slider(
        "p (Minkowski order)",
        options=[1, 1.5, 2, 3, 4, 6, 10, 20, "∞"],
        value=2,
    )
    p = to_p(p_choice)

    x, y = minkowski_ball_2d((a1, a2), R, p)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, fill="toself", mode="lines",
        line=dict(color="royalblue"), fillcolor="rgba(65,105,225,0.3)",
        name=p_label(p),
    ))
    fig.add_trace(go.Scatter(
        x=[a1], y=[a2], mode="markers",
        marker=dict(color="black", size=8), name="center a",
    ))
    add_shape_annotations_2d(fig, (a1, a2), R, p)
    fig.update_layout(
        title=f"B(a, R) under {p_label(p)}",
        xaxis_title="x₁", yaxis_title="x₂",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=700, height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.latex(dp_metric_latex(p, dim=2))
    st.latex(ball_metric_latex(p))
    show_shape_facts(p, R, dim=2)

    st.markdown(f"""
**What you're looking at:** the set `{{x : d_p(x, a) < R}}` for `a = ({a1}, {a2})`, `R = {R}`.
- p = 1 → diamond (rhombus)
- p = 2 → circle
- p → ∞ → square
""")

# ------------------------------------------------------------
elif mode == "2D Containment Overlay":
    st.sidebar.subheader("Overlay parameters")
    a1 = st.sidebar.slider("Center a₁", -5.0, 5.0, 0.0, 0.1, key="ov_a1")
    a2 = st.sidebar.slider("Center a₂", -5.0, 5.0, 0.0, 0.1, key="ov_a2")
    R = st.sidebar.slider("Radius R", 0.1, 5.0, 1.0, 0.1, key="ov_R")
    p_options = st.sidebar.multiselect(
        "p values to overlay",
        options=[1, 1.5, 2, 3, 5, 10, "∞"],
        default=[1, 2, "∞"],
    )
    show_ref_lines = st.sidebar.checkbox("Show reference lines (diagonals/sides)", value=False)

    fig = go.Figure()
    colors = ["crimson", "royalblue", "seagreen", "darkorange", "purple", "brown", "black"]
    for i, p_choice in enumerate(p_options):
        p = to_p(p_choice)
        x, y = minkowski_ball_2d((a1, a2), R, p)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
            name=p_label(p),
        ))
        if show_ref_lines:
            add_shape_annotations_2d(fig, (a1, a2), R, p, color=colors[i % len(colors)])
    fig.add_trace(go.Scatter(
        x=[a1], y=[a2], mode="markers",
        marker=dict(color="black", size=8), name="center a",
    ))
    fig.update_layout(
        title="Containment: B∞ ⊇ B₂ ⊇ B₁ (same R, same center)",
        xaxis_title="x₁", yaxis_title="x₂",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=750, height=750,
    )
    st.plotly_chart(fig, use_container_width=True)

    for p_choice in p_options:
        p = to_p(p_choice)
        st.latex(dp_metric_latex(p, dim=2))
        show_shape_facts(p, R, dim=2)

# ------------------------------------------------------------
elif mode == "3D Ball Explorer":
    st.sidebar.subheader("Ball parameters")
    a1 = st.sidebar.slider("Center a₁", -5.0, 5.0, 0.0, 0.1, key="3d_a1")
    a2 = st.sidebar.slider("Center a₂", -5.0, 5.0, 0.0, 0.1, key="3d_a2")
    a3 = st.sidebar.slider("Center a₃", -5.0, 5.0, 0.0, 0.1, key="3d_a3")
    R = st.sidebar.slider("Radius R", 0.1, 5.0, 1.0, 0.1, key="3d_R")
    p_choice = st.sidebar.select_slider(
        "p (Minkowski order)",
        options=[1, 1.5, 2, 3, 5, 10, "∞"],
        value=1, key="3d_p",
    )
    p = to_p(p_choice)

    x, y, z = minkowski_ball_3d((a1, a2, a3), R, p)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.85, showscale=False)])
    add_shape_annotations_3d(fig, (a1, a2, a3), R, p)
    fig.update_layout(
        title=f"B(a, R) in R³ under {p_label(p)}",
        scene=dict(
            xaxis_title="x₁", yaxis_title="x₂", zaxis_title="x₃",
            aspectmode="cube",
        ),
        width=800, height=800,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.latex(dp_metric_latex(p, dim=3))
    st.latex(ball_metric_latex(p))
    show_shape_facts(p, R, dim=3)

    st.markdown("""
- p = 1 → octahedron
- p = 2 → sphere
- p → ∞ → cuboid (cube if R is the same in all directions)
""")

# ------------------------------------------------------------
elif mode == "Distance Calculator":
    st.sidebar.subheader("Two points in Rⁿ")
    n = st.sidebar.selectbox("Dimension n", [2, 3], index=0)
    st.write(f"Enter coordinates for two points in R^{n}")
    cols = st.columns(2)
    point1, point2 = [], []
    with cols[0]:
        st.markdown("**Point x**")
        for i in range(n):
            point1.append(st.number_input(f"x{i + 1}", value=0.0, key=f"x{i}"))
    with cols[1]:
        st.markdown("**Point y**")
        for i in range(n):
            point2.append(st.number_input(f"y{i + 1}", value=1.0, key=f"y{i}"))

    rows = []
    for p, label in [(1, "d1 (Manhattan)"), (2, "d2 (Euclidean)"), (np.inf, "d∞ (Chebyshev)")]:
        rows.append({"Metric": label, "Distance": round(minkowski_distance(point1, point2, p), 4)})
    st.table(rows)

    st.markdown("**General equations used above:**")
    for p in [1, 2, np.inf]:
        st.latex(dp_metric_latex(p, dim=n))

    if n == 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[point1[0], point2[0]], y=[point1[1], point2[1]],
            mode="markers+text", text=["x", "y"], textposition="top center",
            marker=dict(size=10, color=["crimson", "royalblue"]),
        ))
        fig.update_layout(
            xaxis_title="x₁", yaxis_title="x₂",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=500, height=500,
        )
        st.plotly_chart(fig, use_container_width=True)