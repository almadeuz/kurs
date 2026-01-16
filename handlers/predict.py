from aiogram import Router, F
from aiogram.types import CallbackQuery, Message
from keyboards import menu_kb, cancel_kb
from model import predict, load_model
from typing import Dict

router:     Router = Router()
users:      Dict[int, Dict[str, object]] = {}

load_model()

steps = [
    ("Pregnancies", "Введите количество беременностей:"),
    ("Glucose", "Введите уровень глюкозы:"),
    ("BloodPressure", "Введите давление:"),
    ("SkinThickness", "Введите толщину складки:"),
    ("Insulin", "Введите уровень инсулина:"),
    ("BMI", "Введите BMI:"),
    ("DiabetesPedigreeFunction", "Введите функцию родословной:"),
    ("Age", "Введите возраст:")
]

@router.callback_query(F.data == "predict")
async def predict_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    await callback.answer()
    
    users[user_id] = {"step": 0, "features": {}}
    await show_next(user_id, callback)

@router.callback_query(F.data == "cancel")
async def cancel_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    await callback.answer()
    del users[user_id]
    
    await callback.message.edit_text("Бот для предсказания диабета.\n\n")
    await callback.message.edit_reply_markup(reply_markup=menu_kb())

@router.message(F.text)
async def process_input(message: Message):
    user_id = message.from_user.id
    
    user_data = users[user_id]
    step = user_data.get("step")
    
    if step >= len(steps):
        if user_id in users:
            del users[user_id]
        return
    
    field, _ = steps[step]
    
    value = float(message.text)
    
    if value == 0 and field in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        from model import medians
        value = medians.get(field)
    
    user_data["features"][field] = value
    user_data["step"] = step + 1
    
    await show_next(user_id, message)
        

async def show_next(user_id, msg_or_cb):    
    user_data = users[user_id]
    step = user_data.get("step")
    
    if step < len(steps):
        _, prompt = steps[step]
        text = f"{prompt}\n\nВведите 0 если неизвестно"
        
        if isinstance(msg_or_cb, Message):
            await msg_or_cb.answer(text, reply_markup=cancel_kb())
        else:
            await msg_or_cb.message.edit_text(text)
            await msg_or_cb.message.edit_reply_markup(reply_markup=cancel_kb())
    else:
        features = user_data.get("features")
        result = predict(features)
        
        advice = ""
        if result["prob"] > 0.5:
            advice = "Нужно срочно обратиться в больницу!"
        else:
            advice = "В больницу можно не обращаться, но провериться никогда не помешает."
        
        text = (
            f"Результат: {result["label"]}\n"
            f"Вероятность: {result["prob_formatted"]}\n"
            f"{advice}"
        )
        
        if user_id in users:
            del users[user_id]
        
        if isinstance(msg_or_cb, Message):
            await msg_or_cb.answer(text, reply_markup=menu_kb())
        else:
            await msg_or_cb.message.edit_text(text)
            await msg_or_cb.message.edit_reply_markup(reply_markup=menu_kb())