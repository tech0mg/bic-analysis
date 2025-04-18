import streamlit as st
from db import SessionLocal
from models import User, Reception, Store, Category

def get_distinct_values(column, model):
    """
    指定されたモデルとカラムから重複を除いた値のリストを取得する共通関数。
    """
    session = SessionLocal()
    try:
        results = session.query(column).distinct().all()
        return [r[0] for r in results if r[0] is not None]
    except Exception as e:
        st.error(f"データ取得中にエラーが発生しました: {e}")
        return []
    finally:
        session.close()

def get_store_options():
    """
    Storeテーブルから店舗名とIDを取得し、名称->IDの辞書を返す。
    """
    session = SessionLocal()
    try:
        results = session.query(Store.id, Store.name).filter(Store.is_available == True).all()
        # 結果を辞書にマッピング: name -> id
        return {name: id for id, name in results}
    except Exception as e:
        st.error(f"店舗情報取得中にエラーが発生しました: {e}")
        return {}
    finally:
        session.close()

def get_category_options():
    """
    Categoryテーブルからカテゴリ名とIDを取得し、名称->IDの辞書を返す。
    """
    session = SessionLocal()
    try:
        results = session.query(Category.id, Category.name).all()
        # 結果を辞書にマッピング: name -> id
        return {name: id for id, name in results}
    except Exception as e:
        st.error(f"カテゴリ情報取得中にエラーが発生しました: {e}")
        return {}
    finally:
        session.close()
