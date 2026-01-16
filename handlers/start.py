"""
Обработчики главного меню
"""

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import CommandStart
from keyboards import menu_kb

router: Router = Router()

@router.message(CommandStart())
async def start_command(message: Message):
    await message.answer("Бот для предсказания диабета.\n\n", reply_markup=menu_kb())