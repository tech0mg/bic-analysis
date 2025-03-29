# auth_utils.py
import bcrypt
import streamlit as st
from db import SessionLocal
from models import Store

def hash_password(plain_password: str) -> str:
    # bcryptでパスワードをハッシュ化
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    # 平文パスワードをbcryptハッシュと比較
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

# ログイン機能
def login_page():
    st.title("店舗ログイン")
    store_name = st.text_input("店舗名 「大宮」で入れます")
    plain_password = st.text_input("パスワード 「omiya」で入れます", type="password")

    if st.button("ログイン"):
        session = SessionLocal()
        try:
            store_obj = session.query(Store).filter_by(name=store_name).first()
            if not store_obj:
                st.error("店舗が見つかりません。")
            else:
                if verify_password(plain_password, store_obj.password):
                    st.session_state["logged_in"] = True
                    st.rerun()  # スクリプトを再実行し、ログイン済みの画面に切り替え
                else:
                    st.error("パスワードが間違っています。")
        except Exception as e:
            st.error(f"ログイン処理中にエラーが発生しました: {e}")
        finally:
            session.close()

# 店舗登録機能
def register_store():
    st.title("店舗登録")
    store_name = st.text_input("店舗名（新規登録）")
    plain_password = st.text_input("パスワード", type="password")
    prefectures = [
        "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
        "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
        "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
        "岐阜県", "静岡県", "愛知県", "三重県",
        "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
        "鳥取県", "島根県", "岡山県", "広島県", "山口県",
        "徳島県", "香川県", "愛媛県", "高知県",
        "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県",
        "沖縄県"
    ]
    prefecture = st.selectbox("都道府県", prefectures)

    if st.button("店舗登録"):
        session = SessionLocal()
        try:
            # 既に同じ店舗名が存在しないかチェック
            existing = session.query(Store).filter_by(name=store_name).first()
            if existing:
                st.error("この店舗名は既に使用されています。")
            else:
                hashed_pass = hash_password(plain_password)
                new_store = Store(name=store_name, password=hashed_pass, prefecture=prefecture)
                session.add(new_store)
                session.commit()
                st.success("店舗登録に成功しました。")
        except Exception as e:
            session.rollback()
            st.error(f"店舗登録中にエラーが発生しました: {e}")
        finally:
            session.close()