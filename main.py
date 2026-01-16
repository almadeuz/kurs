"""
Точка входа в бота
"""

import asyncio
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage

async def main():
    load_dotenv()
    key:        str = os.getenv("key")
    bot:        Bot = Bot(token=key)
    storage:    MemoryStorage = MemoryStorage()
    dp:         Dispatcher = Dispatcher(storage=storage)
    
    import handlers as hs
    dp.include_router(hs.start)
    dp.include_router(hs.predict)
    
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())