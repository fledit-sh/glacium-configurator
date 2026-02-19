# app.py
import os
import sqlite3
from pathlib import Path
from datetime import datetime

import streamlit as st

DATA_DIR = Path("data")
FILES_DIR = DATA_DIR / "files"
DB_PATH = DATA_DIR / "trace.db"

FILES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="UBIQ Traceability",
    page_icon="ubiq.png",
    layout="wide",
)

st.image("ubiq.png", width=180)

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        serial TEXT NOT NULL,
        doc_type TEXT NOT NULL,
        doc_version TEXT NOT NULL,
        filename TEXT NOT NULL,
        uploaded_at TEXT NOT NULL
    )
    """)
    return conn

def list_docs(conn, serial: str):
    cur = conn.execute(
        "SELECT doc_type, doc_version, filename, uploaded_at FROM docs WHERE serial=? ORDER BY uploaded_at DESC",
        (serial,),
    )
    return cur.fetchall()

def save_doc(conn, serial: str, doc_type: str, doc_version: str, file_bytes: bytes, original_name: str):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_name = f"{serial}__{doc_type}__v{doc_version}__{ts}__{original_name}".replace("/", "_")
    out_path = FILES_DIR / safe_name
    out_path.write_bytes(file_bytes)

    conn.execute(
        "INSERT INTO docs(serial, doc_type, doc_version, filename, uploaded_at) VALUES(?,?,?,?,?)",
        (serial, doc_type, doc_version, safe_name, ts),
    )
    conn.commit()

def is_admin_logged_in() -> bool:
    return st.session_state.get("admin_ok", False)

def admin_login():
    st.sidebar.subheader("Admin Login")
    pw = st.sidebar.text_input("Passwort", type="password")
    if st.sidebar.button("Login"):
        if pw == os.getenv("ADMIN_PASSWORD", "changeme"):
            st.session_state["admin_ok"] = True
            st.sidebar.success("Eingeloggt")
        else:
            st.sidebar.error("Falsches Passwort")

def device_page(conn, serial: str):
    st.title(f"Device: {serial}")
    st.caption("Technical Documentation & Traceability")

    docs = list_docs(conn, serial)
    if not docs:
        st.info("Noch keine Dokumente hinterlegt.")
    else:
        for doc_type, doc_version, filename, uploaded_at in docs:
            st.markdown(f"**{doc_type}** — Version: `{doc_version}` — Upload: `{uploaded_at}`")
            file_path = FILES_DIR / filename
            if file_path.exists():
                st.download_button(
                    label=f"Download: {doc_type}",
                    data=file_path.read_bytes(),
                    file_name=filename.split("__", maxsplit=4)[-1],
                    mime="application/pdf",
                    key=f"{serial}:{filename}",
                )
            st.divider()

    if is_admin_logged_in():
        st.subheader("Admin: Upload document")
        doc_type = st.text_input("Documenttype (i.e. DoC, Manual, TestReport)")
        doc_version = st.text_input("Version (z.B. 1.0.2)")
        up = st.file_uploader("Upload pdf", type="pdf")
        if st.button("Save") and up and doc_type and doc_version:
            save_doc(conn, serial, doc_type, doc_version, up.getvalue(), up.name)
            st.success("Saved.")
            st.rerun()

def main():
    conn = db()
    admin_login()

    # "Routing": Serial aus Query-Param lesen: ?sn=KM1-000123
    q = st.query_params
    serial = q.get("sn", None)

    st.sidebar.subheader("Navigation")
    sn_input = st.sidebar.text_input("Open serial number", value=serial or "")
    if st.sidebar.button("Öffnen") and sn_input:
        st.query_params["sn"] = sn_input
        st.rerun()

    if not serial:
        st.title("Traceability Portal (MVP)")
        st.write("Öffne eine Seriennummer über die Sidebar oder per QR-Code URL:")
        st.code("https://dein-app.streamlit.app/?sn=KM1-000123")
        st.stop()

    device_page(conn, serial)

if __name__ == "__main__":
    main()
