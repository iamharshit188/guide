"""
SQL fundamentals using SQLite (built-in — no extra install needed).
Covers: schema design, joins, indexes, aggregations, window functions,
        CTEs, EXPLAIN QUERY PLAN, and query optimization patterns.
"""

import sqlite3
import time
import random
import string


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ── Setup ─────────────────────────────────────────────────────

def setup_db(conn):
    cur = conn.cursor()
    cur.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL UNIQUE,
            email      TEXT NOT NULL UNIQUE,
            role       TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user','guest')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            is_active  INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL UNIQUE,
            parent_id     INTEGER REFERENCES categories(category_id)
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT NOT NULL,
            price         REAL NOT NULL CHECK (price >= 0),
            stock         INTEGER NOT NULL DEFAULT 0,
            category_id   INTEGER REFERENCES categories(category_id)
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(user_id),
            product_id  INTEGER NOT NULL REFERENCES products(product_id),
            quantity    INTEGER NOT NULL CHECK (quantity > 0),
            amount      REAL NOT NULL,
            order_date  TEXT NOT NULL DEFAULT (datetime('now')),
            status      TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','shipped','delivered','cancelled'))
        );
    """)
    conn.commit()


def seed_data(conn, n_users=100, n_orders=500):
    cur = conn.cursor()
    rng = random.Random(42)

    # Categories (hierarchical)
    parent_cats = ["Electronics", "Books", "Clothing", "Sports"]
    child_cats = {
        "Electronics": ["Laptops", "Phones", "Accessories"],
        "Books": ["Fiction", "Technical", "Self-Help"],
        "Clothing": ["Men", "Women", "Kids"],
        "Sports": ["Outdoor", "Gym", "Team Sports"],
    }
    cur.executemany("INSERT OR IGNORE INTO categories (name) VALUES (?)",
                    [(c,) for c in parent_cats])
    cat_ids = {r[1]: r[0] for r in cur.execute("SELECT category_id, name FROM categories")}
    for parent, children in child_cats.items():
        for child in children:
            cur.execute("INSERT OR IGNORE INTO categories (name, parent_id) VALUES (?,?)",
                        (child, cat_ids[parent]))
    conn.commit()

    # Users
    for i in range(n_users):
        role = rng.choice(["admin", "user", "user", "user", "guest"])
        cur.execute(
            "INSERT OR IGNORE INTO users (username, email, role, created_at) VALUES (?,?,?,?)",
            (f"user_{i:04d}", f"user_{i}@example.com", role,
             f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}")
        )
    conn.commit()

    # Products
    all_cats = [r[0] for r in cur.execute("SELECT category_id FROM categories")]
    products = [(f"Product_{i}", round(rng.uniform(5, 500), 2), rng.randint(0, 200),
                 rng.choice(all_cats)) for i in range(50)]
    cur.executemany("INSERT OR IGNORE INTO products (name, price, stock, category_id) VALUES (?,?,?,?)",
                    products)
    conn.commit()

    # Orders
    user_ids = [r[0] for r in cur.execute("SELECT user_id FROM users")]
    prod_ids = [r[0] for r in cur.execute("SELECT product_id FROM products")]
    for _ in range(n_orders):
        uid = rng.choice(user_ids)
        pid = rng.choice(prod_ids)
        qty = rng.randint(1, 5)
        price = cur.execute("SELECT price FROM products WHERE product_id=?", (pid,)).fetchone()[0]
        cur.execute(
            "INSERT INTO orders (user_id, product_id, quantity, amount, order_date, status) "
            "VALUES (?,?,?,?,?,?)",
            (uid, pid, qty, round(price * qty, 2),
             f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
             rng.choice(["pending","shipped","delivered","cancelled"]))
        )
    conn.commit()


def run_query(conn, sql, params=(), label=None):
    cur = conn.cursor()
    t0 = time.perf_counter()
    rows = cur.execute(sql, params).fetchall()
    elapsed = (time.perf_counter() - t0) * 1000
    if label:
        print(f"\n  [{label}]  ({elapsed:.2f}ms, {len(rows)} rows)")
    return rows, elapsed


# ── Main ──────────────────────────────────────────────────────

def main():
    conn = sqlite3.connect(":memory:")
    setup_db(conn)
    seed_data(conn, n_users=100, n_orders=500)

    section("1. BASIC SELECT, WHERE, ORDER BY, LIMIT")
    rows, _ = run_query(conn,
        "SELECT user_id, username, role, created_at FROM users "
        "WHERE role != 'guest' ORDER BY created_at DESC LIMIT 5",
        label="Recent non-guest users")
    for r in rows:
        print(f"  {r}")

    section("2. JOIN TYPES")
    # INNER JOIN — only users who placed orders
    rows, _ = run_query(conn, """
        SELECT u.username, COUNT(o.order_id) AS n_orders, ROUND(SUM(o.amount),2) AS total_spent
        FROM users u
        INNER JOIN orders o ON u.user_id = o.user_id
        GROUP BY u.user_id
        ORDER BY total_spent DESC
        LIMIT 5
    """, label="INNER JOIN: top 5 spenders")
    print(f"  {'Username':15s}  {'Orders':>8}  {'Total Spent':>12}")
    for r in rows:
        print(f"  {r[0]:15s}  {r[1]:>8}  {r[2]:>12.2f}")

    # LEFT JOIN — all users including those with no orders
    rows, _ = run_query(conn, """
        SELECT u.username, COALESCE(COUNT(o.order_id), 0) AS n_orders
        FROM users u
        LEFT JOIN orders o ON u.user_id = o.user_id
        GROUP BY u.user_id
        HAVING n_orders = 0
        LIMIT 5
    """, label="LEFT JOIN: users with 0 orders")
    print(f"  Users with no orders: {[r[0] for r in rows]}")

    # Self-join — categories with their parent names
    rows, _ = run_query(conn, """
        SELECT c.name AS subcategory, p.name AS parent_category
        FROM categories c
        LEFT JOIN categories p ON c.parent_id = p.category_id
        ORDER BY p.name, c.name
    """, label="SELF JOIN: category hierarchy")
    print(f"  {'Subcategory':20s}  {'Parent':20s}")
    for r in rows[:8]:
        print(f"  {r[0]:20s}  {str(r[1]):20s}")

    section("3. AGGREGATION & GROUP BY")
    rows, _ = run_query(conn, """
        SELECT
            c.name AS category,
            COUNT(DISTINCT p.product_id) AS n_products,
            ROUND(AVG(p.price), 2) AS avg_price,
            ROUND(MIN(p.price), 2) AS min_price,
            ROUND(MAX(p.price), 2) AS max_price,
            SUM(p.stock) AS total_stock
        FROM categories c
        JOIN products p ON c.category_id = p.category_id
        WHERE c.parent_id IS NULL
        GROUP BY c.category_id
        HAVING n_products > 0
        ORDER BY avg_price DESC
    """, label="Category summary")
    print(f"  {'Category':14s}  {'N Prods':>8}  {'Avg $':>8}  {'Min $':>8}  {'Max $':>8}  {'Stock':>8}")
    for r in rows:
        print(f"  {r[0]:14s}  {r[1]:>8}  {r[2]:>8.2f}  {r[3]:>8.2f}  {r[4]:>8.2f}  {r[5]:>8}")

    section("4. WINDOW FUNCTIONS")
    rows, _ = run_query(conn, """
        SELECT
            o.order_id,
            u.username,
            o.amount,
            o.order_date,
            SUM(o.amount) OVER (
                PARTITION BY o.user_id
                ORDER BY o.order_date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS running_total,
            RANK() OVER (PARTITION BY o.user_id ORDER BY o.amount DESC) AS rank_by_amount,
            LAG(o.amount, 1) OVER (PARTITION BY o.user_id ORDER BY o.order_date) AS prev_amount
        FROM orders o
        JOIN users u ON o.user_id = u.user_id
        WHERE o.user_id = (
            SELECT user_id FROM orders GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 1
        )
        ORDER BY o.order_date
        LIMIT 6
    """, label="Window functions: running total, rank, lag")
    print(f"  {'ID':>4}  {'User':12s}  {'Amount':>8}  {'RunTotal':>10}  {'Rank':>6}  {'PrevAmt':>9}")
    for r in rows:
        print(f"  {r[0]:>4}  {r[1]:12s}  {r[2]:>8.2f}  {r[4]:>10.2f}  {r[5]:>6}  {str(r[6]):>9}")

    section("5. CTE — COMMON TABLE EXPRESSIONS")
    rows, _ = run_query(conn, """
        WITH user_stats AS (
            SELECT
                u.user_id,
                u.username,
                u.role,
                COUNT(o.order_id)   AS n_orders,
                COALESCE(SUM(o.amount), 0) AS total_spent,
                COALESCE(AVG(o.amount), 0) AS avg_order
            FROM users u
            LEFT JOIN orders o ON u.user_id = o.user_id
            GROUP BY u.user_id
        ),
        ranked AS (
            SELECT *,
                   NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
            FROM user_stats
            WHERE n_orders > 0
        )
        SELECT quartile,
               COUNT(*) AS n_users,
               ROUND(AVG(total_spent), 2) AS avg_spent,
               ROUND(AVG(n_orders), 1) AS avg_orders
        FROM ranked
        GROUP BY quartile
        ORDER BY quartile
    """, label="CTE: user spend quartiles")
    print(f"  {'Quartile':>10}  {'Users':>8}  {'Avg Spent':>12}  {'Avg Orders':>12}")
    for r in rows:
        print(f"  {'Q' + str(r[0]):>10}  {r[1]:>8}  {r[2]:>12.2f}  {r[3]:>12.1f}")

    section("6. INDEXES & EXPLAIN QUERY PLAN")
    cur = conn.cursor()

    # Query without index
    explain_no_idx = cur.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM orders WHERE status='delivered'"
    ).fetchall()
    print("\nNo index on status:")
    for row in explain_no_idx:
        print(f"  {row}")

    # Create index
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_user_date ON orders(user_id, order_date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_amount ON orders(amount DESC)")
    conn.commit()

    explain_idx = cur.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM orders WHERE status='delivered'"
    ).fetchall()
    print("\nWith index on status:")
    for row in explain_idx:
        print(f"  {row}")

    explain_composite = cur.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM orders WHERE user_id=1 AND order_date > '2024-06-01'"
    ).fetchall()
    print("\nComposite index (user_id, order_date):")
    for row in explain_composite:
        print(f"  {row}")

    # Benchmark: indexed vs non-indexed
    print("\nBenchmark: 1000 repeated queries")
    for use_index in [False, True]:
        col = "status" if use_index else "amount"
        idx_sql = "DROP INDEX IF EXISTS idx_orders_amount"
        if not use_index:
            cur.execute(idx_sql)
            conn.commit()

        t0 = time.perf_counter()
        for _ in range(1000):
            cur.execute("SELECT * FROM orders WHERE status=?", ("delivered",)).fetchall()
        elapsed = (time.perf_counter() - t0) * 1000
        idx_label = "WITH index" if use_index else "WITHOUT index"
        print(f"  {idx_label}: {elapsed:.1f}ms total  ({elapsed/1000:.3f}ms/query)")

    section("7. SUBQUERIES")
    rows, _ = run_query(conn, """
        SELECT product_id, name, price
        FROM products
        WHERE price > (SELECT AVG(price) FROM products)
        ORDER BY price DESC
        LIMIT 5
    """, label="Products above average price")
    print(f"  Avg price baseline: {cur.execute('SELECT ROUND(AVG(price),2) FROM products').fetchone()[0]}")
    for r in rows:
        print(f"  {r}")

    # Correlated subquery
    rows, _ = run_query(conn, """
        SELECT u.username,
               (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.user_id
                AND o.status = 'delivered') AS delivered_count
        FROM users u
        ORDER BY delivered_count DESC
        LIMIT 5
    """, label="Correlated subquery: delivered orders per user")
    for r in rows:
        print(f"  {r}")

    section("8. ORDER STATUS ANALYTICS — COMPLEX QUERY")
    rows, _ = run_query(conn, """
        WITH order_summary AS (
            SELECT
                strftime('%Y-%m', order_date) AS month,
                status,
                COUNT(*) AS cnt,
                ROUND(SUM(amount), 2) AS revenue
            FROM orders
            GROUP BY month, status
        ),
        monthly_total AS (
            SELECT month, SUM(cnt) AS total_orders, SUM(revenue) AS total_revenue
            FROM order_summary GROUP BY month
        )
        SELECT
            os.month,
            os.status,
            os.cnt,
            ROUND(os.cnt * 100.0 / mt.total_orders, 1) AS pct_of_orders,
            os.revenue
        FROM order_summary os
        JOIN monthly_total mt USING (month)
        ORDER BY os.month DESC, os.cnt DESC
        LIMIT 12
    """, label="Monthly order status breakdown")
    print(f"  {'Month':>8}  {'Status':>12}  {'Count':>7}  {'% Total':>8}  {'Revenue':>10}")
    for r in rows:
        print(f"  {r[0]:>8}  {r[1]:>12}  {r[2]:>7}  {r[3]:>7.1f}%  {r[4]:>10.2f}")

    conn.close()
    print("\n  [Done] All SQLite queries executed in-memory.")


if __name__ == "__main__":
    main()
