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
    __tablename__ ="answer_info"
    id = Column(Integer, primary_key=True, autoincrement=True)
    reception_id = Column(Integer, ForeignKey("reception.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("question.id", ondelete="CASCADE"), nullable=False)
    answer_numeric = Column(Integer, nullable=True)
    answer_boolean = Column(Boolean, nullable=True)
    answer_categorical = Column(String(255), nullable=True)


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