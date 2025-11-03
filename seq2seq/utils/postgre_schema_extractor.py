import psycopg2

def extract_table_schema_from_postgres(db=None, f=None, user="sm3", password="secretserver", host="localhost", port=5432):
    conn = psycopg2.connect(
        dbname="sm3", user=user, password=password, host=host, port=port
    )
    cur = conn.cursor()
    
    table = f
    if table is None:
        raise NotImplementedError("For SM3, include table name as a db name.")

    schema = {
        "db_id": table,
        "table_names_original": [table],
        "table_names": [table.lower().replace("_", " ")],
        "column_names_original": [],
        "column_names": [],
        "column_types": [],
        "primary_keys": [],
        "foreign_keys": [],
    }

    # Column info
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = 'public';
    """, (table,))
    columns = cur.fetchall()

    schema["column_names_original"].append((-1, "*"))
    schema["column_names"].append((-1, "*"))
    schema["column_types"].append("text")

    for idx, (col_name, col_type) in enumerate(columns):
        schema["column_names_original"].append((0, col_name))
        schema["column_names"].append((0, col_name.lower().replace("_", " ")))

        col_type = col_type.lower()
        if any(x in col_type for x in ["char", "text"]):
            schema["column_types"].append("text")
        elif any(x in col_type for x in ["int", "numeric", "decimal", "real", "double", "float"]):
            schema["column_types"].append("number")
        elif any(x in col_type for x in ["date", "time", "timestamp"]):
            schema["column_types"].append("time")
        elif "boolean" in col_type:
            schema["column_types"].append("boolean")
        else:
            schema["column_types"].append("others")

    # Primary keys
    cur.execute("""
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass AND i.indisprimary;
    """, (table,))
    pk_cols = [row[0] for row in cur.fetchall()]
    for pk in pk_cols:
        for i, (_, col_name) in enumerate(schema["column_names_original"]):
            if col_name == pk:
                schema["primary_keys"].append(i)

    # Foreign keys
    cur.execute("""
        SELECT
            kcu.column_name,
            ccu.table_name,
            ccu.column_name
        FROM
            information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
              ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
    """, (table,))
    fks = cur.fetchall()
    for from_col, to_table, to_col in fks:
        from_idx = next(i for i, (_, col) in enumerate(schema["column_names_original"]) if col == from_col)
        # we assume referenced table is loaded independently too → both sides are (0, idx)
        schema["foreign_keys"].append([(0, from_idx), (0, -1)])  # leave -1 placeholder for now

    cur.close()
    conn.close()
    return schema
