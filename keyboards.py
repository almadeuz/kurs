"""
Клавиатуры для бота
"""

from aiogram.utils.keyboard import InlineKeyboardBuilder

def menu_kb():
    """Основная клавиатура"""
    kb = InlineKeyboardBuilder()
    kb.button(text="Начать анализ", callback_data="predict")
    return kb.as_markup()

def cancel_kb():
    """Клавиатура отмены"""
    kb = InlineKeyboardBuilder()
    kb.button(text="Отмена", callback_data="cancel")
    return kb.as_markup()