# app/models.py
from sqlalchemy import Column, Integer, String, TIMESTAMP, func, text, ForeignKey, Boolean
from sqlalchemy.types import Enum
from db import Base

# userテーブル
class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    store_id = Column(Integer, ForeignKey("store.id"), nullable=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(Enum("male", "female", "other"), nullable=True)
    household = Column(Integer, nullable=True)
    time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=True)

# receptionテーブル
class Reception(Base):
    __tablename__ = "reception"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=True, index=True)
    category_id = Column(Integer, ForeignKey("question.id", ondelete="SET NULL"), nullable=True)
    time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"),nullable=True)

# answer_infoテーブル
class Answer_info(Base):
    __tablename__ = "answer_info"
    id = Column(Integer, primary_key=True, index=True)
    reception_id = Column(Integer, ForeignKey("reception.id"), nullable=False)
    question_id = Column(Integer, ForeignKey("question.id"), nullable=False)
    answer = Column(Integer, nullable=False)

# salse_callテーブル
class Sales_call(Base):
    __tablename__ ="sales_call"
    id = Column(Integer, primary_key=True, autoincrement=True)
    reception_id = Column(Integer, ForeignKey("reception.id", ondelete="CASCADE"), nullable=False, index=True)
    #category_id = Column(Integer, ForeignKey("question.id", ondelete="SET NULL"), nullable=True)
    #store_id = Column(Integer, ForeignKey("store.id"), nullable=True, index=True)
    time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=True)


# storeテーブル
class Store(Base):
    __tablename__ = "store"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)  # bcryptでハッシュ化済みの文字列を格納
    prefecture = Column(String(100), nullable=True)
    is_available = Column(Boolean, default=True)
    time = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # question テーブル（設問情報）
class Question(Base):
    __tablename__ = "question"
    id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("category.id"))
    question_text = Column(String)

# question_option テーブル（設問の選択肢情報）
class QuestionOption(Base):
    __tablename__ = "question_option"
    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("question.id"))
    label = Column(String)
    value = Column(Integer)

# カテゴリテーブルモデル
class Category(Base):
    __tablename__ = "category"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)