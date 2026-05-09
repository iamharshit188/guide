import numpy as np


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    section("1. VECTOR BASICS")

    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])

    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"2 * v1  = {2 * v1}")
    print(f"v1 - v2 = {v1 - v2}")

    section("2. NORMS")

    l1 = np.sum(np.abs(v1))
    l2 = np.linalg.norm(v1)
    linf = np.max(np.abs(v1))

    print(f"v1 = {v1}")
    print(f"L1 norm  ||v1||_1 = {l1:.4f}  (sum of abs values)")
    print(f"L2 norm  ||v1||_2 = {l2:.4f}  (Euclidean length)")
    print(f"L∞ norm  ||v1||_∞ = {linf:.4f}  (max abs value)")

    unit_v1 = v1 / l2
    print(f"Unit vector v1/||v1|| = {unit_v1.round(4)}  (||.||_2 = {np.linalg.norm(unit_v1):.4f})")

    section("3. DOT PRODUCT")

    dot = np.dot(v1, v2)
    dot_manual = sum(a * b for a, b in zip(v1, v2))

    print(f"v1 · v2 (np.dot)  = {dot:.4f}")
    print(f"v1 · v2 (manual)  = {dot_manual:.4f}")
    print(f"Geometric: ||v1|| * ||v2|| * cos(θ) = {np.linalg.norm(v1) * np.linalg.norm(v2):.4f} * cos(θ)")

    theta_rad = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    print(f"Angle θ between v1, v2 = {np.degrees(theta_rad):.2f}°")

    # Orthogonal vectors
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    print(f"\nOrthogonal check: [1,0] · [0,1] = {np.dot(a, b):.4f}  (should be 0)")

    section("4. COSINE SIMILARITY")

    def cosine_similarity(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    sim_same = cosine_similarity(v1, v1)
    sim_v1v2 = cosine_similarity(v1, v2)
    sim_opp = cosine_similarity(v1, -v1)

    print(f"cos_sim(v1, v1)  = {sim_same:.4f}  (identical → 1.0)")
    print(f"cos_sim(v1, v2)  = {sim_v1v2:.4f}  (similar direction)")
    print(f"cos_sim(v1, -v1) = {sim_opp:.4f}  (opposite → -1.0)")

    # Embedding analogy: same semantic meaning → high cosine similarity
    doc_a = np.array([0.9, 0.1, 0.8, 0.05])  # "ML embedding"
    doc_b = np.array([0.85, 0.15, 0.75, 0.1])  # similar doc
    doc_c = np.array([0.05, 0.95, 0.1, 0.9])  # different topic

    print(f"\nEmbedding similarity demo:")
    print(f"  doc_a vs doc_b (similar): {cosine_similarity(doc_a, doc_b):.4f}")
    print(f"  doc_a vs doc_c (different): {cosine_similarity(doc_a, doc_c):.4f}")

    section("5. VECTOR PROJECTION")

    # proj_v2(v1) = (v1·v2 / v2·v2) * v2
    proj_scalar = np.dot(v1, v2) / np.dot(v2, v2)
    proj_vector = proj_scalar * v2

    print(f"Projection of v1 onto v2:")
    print(f"  scalar component = {proj_scalar:.4f}")
    print(f"  projected vector = {proj_vector.round(4)}")

    # Verify: (v1 - projection) should be orthogonal to v2
    residual = v1 - proj_vector
    print(f"  residual · v2 = {np.dot(residual, v2):.10f}  (should be ~0)")

    section("6. CROSS PRODUCT (3D)")

    cross = np.cross(v1, v2)
    print(f"v1 × v2 = {cross}")
    print(f"  Orthogonal to v1: {np.dot(cross, v1):.10f}  (should be ~0)")
    print(f"  Orthogonal to v2: {np.dot(cross, v2):.10f}  (should be ~0)")
    print(f"  ||v1 × v2|| = {np.linalg.norm(cross):.4f}  (area of parallelogram)")

    area = np.linalg.norm(v1) * np.linalg.norm(v2) * np.sin(theta_rad)
    print(f"  ||v1||*||v2||*sin(θ) = {area:.4f}  (should match above)")

    section("7. LINEAR INDEPENDENCE CHECK")

    # Linearly independent: det of matrix formed by vectors ≠ 0
    A = np.column_stack([v1[:2], v2[:2]])  # 2D for simplicity
    det = np.linalg.det(A)
    print(f"Matrix [v1[:2] | v2[:2]] det = {det:.4f}")
    print(f"Linearly independent: {abs(det) > 1e-10}")

    # Dependent case
    v3 = 2 * v1
    B = np.column_stack([v1[:2], v3[:2]])
    print(f"Matrix [v1 | 2*v1] det = {np.linalg.det(B):.4f}  (dependent → 0)")


if __name__ == "__main__":
    main()
