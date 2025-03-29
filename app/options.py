import streamlit as st
from db import SessionLocal
from models import User, Reception

def get_distinct_values(column, model):
    """
    指定されたモデルとカラムから重複を除いた値のリストを取得する共通関数。
    """
    session = SessionLocal()
    try:
        st.write(f"DEBUG: Getting distinct values for {model.__tablename__}.{column.name}...")
        results = session.query(column).distinct().all()
        st.write("DEBUG: Raw distinct results:", results)
        return [r[0] for r in results if r[0] is not None]
    except Exception as e:
        st.error(f"データ取得中にエラーが発生しました: {e}")
        return []
    finally:
        session.close()

def get_store_options():
    """
    Userテーブルのstore_idをget_distinct_valuesで取得し、ソートして返す。
    """
    store_list = get_distinct_values(User.store_id, User)
    store_list = sorted(store_list)
    return store_list

def get_category_options():
    """
    Receptionテーブルのcategory_idをget_distinct_valuesで取得し、ソートして返す。
    """
    category_list = get_distinct_values(Reception.category_id, Reception)
    category_list = sorted(category_list)
    return category_list
