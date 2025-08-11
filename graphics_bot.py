from aiogram import Bot, Dispatcher
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from dotenv import load_dotenv
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify, lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr
import asyncio

load_dotenv('tokens/BOT_TOKEN.env') # получает токен
TOKEN = os.getenv('TOKEN')

if not TOKEN: # проверяет наличие токена
    raise ValueError('TOKEN not found')

bot = Bot(token=TOKEN)
dp = Dispatcher()

class Keyboards(): # Кнопки главного меню
    def inline_keyboard_menu(self):
        inline_keyboard_menu = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text='График по функции', callback_data='function')],
                             [InlineKeyboardButton(text='График по уравнению', callback_data='equation')],
                             [InlineKeyboardButton(text='График системы ур-ий', callback_data='system_equation')]]
        )
        
        return inline_keyboard_menu
    
class infoFunction(StatesGroup): 
    func = State()
    
class infoEquation(StatesGroup):
    equat = State()
    
class infoSystemEquation(StatesGroup):
    waiting_for_count = State()
    waiting_for_equations = State()    
    
keyboard = Keyboards()

async def build_graph(message: Message, equations: list, state: FSMContext): # создаёт и отправляет график по уравнениям
    try:
        plt.figure(figsize=(10, 8))
        x = symbols('x')
        x_vals = np.linspace(-10, 10, 400)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, eq_text in enumerate(equations):
            try:
                expr = sympify(eq_text)
                f_lam = lambdify(x, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'tan': np.tan}])
                y_vals = f_lam(x_vals)
                
                valid_indices = np.isfinite(y_vals)
                x_valid = x_vals[valid_indices]
                y_valid = y_vals[valid_indices]
                
                color = colors[i % len(colors)]
                plt.plot(x_valid, y_valid, label=f'Уравнение {i+1}: {eq_text}', color=color)
            except Exception as e:
                await message.answer(f"Ошибка при построении уравнения {i+1} ({eq_text}): {str(e)}")
                continue
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Графики функций')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xlim(-10, 10)
        time_stamp = time.time()
        
        file_path = f'system_graphic_{time_stamp}.png'
        plt.savefig(file_path)
        plt.close()
        
        image = FSInputFile(file_path)
        await message.answer_photo(photo=image, caption="Вот график ваших уравнений")
        
        # Отправляем список уравнений
        equations_list = "\n".join([f"{i+1}. <b>{eq}</b>" for i, eq in enumerate(equations)])
        await message.answer(f"Построены графики для уравнений:\n{equations_list}", parse_mode='HTML')
        await state.clear()
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        await message.answer('Не удалось сгенерировать график')
        await state.clear()

@dp.message(lambda message: message.text == '/start') # приветственная фраза
async def start(message: Message):
    await message.answer('Привет, я бот для создания графиков. Выбери какой график ты хочешь создать', reply_markup=keyboard.inline_keyboard_menu())
    
@dp.callback_query() # активация кнопок
async def inline_buttons(callback: CallbackQuery, state: FSMContext):
    await callback.answer()
    if callback.data == 'function':
        await state.set_state(infoFunction.func)
        await callback.message.answer('Напиши функцию (например x**2 + 3 - 8)')
        
    elif callback.data == 'equation':
        await state.set_state(infoEquation.equat)
        await callback.message.answer('Напиши уравнение (например 20 - 3 * x). y = ')
        
    elif callback.data == 'system_equation':
        await state.set_state(infoSystemEquation.waiting_for_count)
        await callback.message.answer('Напиши сколько уравнений надо сгенерировать')

@dp.message(infoFunction.func) # принимает функцию, решает её и отправляет график пользователю
async def graphic_function(message: Message, state: FSMContext):
    await state.update_data(func=message.text)
    data = await state.get_data()
    
    func_x = data['func']
    
    try:
        time_stamp = time.time()
        expr = sympify(func_x)
        x_sym = symbols('x')
        
        f = lambdify(x_sym, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'tan': np.tan}])
        
        x_vals = np.linspace(-10, 10, 400)
        y_vals = f(x_vals)
        file_path = f'{time_stamp}_function.png'
        
        plt.plot(x_vals, y_vals)
        plt.title(f"График функции: {data['func']}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.savefig(file_path)
        
        graphic = FSInputFile(file_path)
        await message.answer_photo(photo=graphic, caption='Вот ваш график функции')
        await state.clear()
    except Exception as e:
        await message.answer('Не удалось сгенерировать график')
        print(f'Ошибка: {e}')
        await state.clear()
        
@dp.message(infoEquation.equat) # принимает уравнение, решает его и отправляет график пользователю
async def graphic_equation(message: Message, state: FSMContext):
    await state.update_data(equat=message.text)
    data = await state.get_data()
    
    try:
        time_stamp = time.time()
        x_sym = symbols('x')
        expr = parse_expr(data['equat'])
        func = lambdify(x_sym, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'tan': np.tan}])
        
        x_vals = np.linspace(-10, 10, 400)
        y_vals = func(x_vals)
        
        plt.figure(figsize=(12, 7))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f"y = {expr}")
        
        plt.title(f'График уравнения: y = {expr}', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        file_path = f"{time_stamp}_equation.png"
        plt.savefig(file_path)
        
        file = FSInputFile(file_path)        
        await message.answer_photo(photo=file, caption='Вот ваш график уравнения')
        await state.clear()
    except Exception as e:
        await message.answer('Не удалось сгенерировать график')
        print(f'Ошибка: {e}')
        await state.clear()
        
@dp.message(infoSystemEquation.waiting_for_count) # запрашивает уравнения у пользователя
async def counter_function(message: Message, state: FSMContext):
    try:
        count = int(message.text)
        
        await state.update_data(equation_count=count, current_equation=1, equations=[])
        await message.answer(f"Введите уравнение 1 (например: x**2 + 2*x + 1):")
        await state.set_state(infoSystemEquation.waiting_for_equations)
                
    except Exception as e:
        await message.answer('Введите число (например 1)')
        print(f'Ошибка: {e}')
        
@dp.message(infoSystemEquation.waiting_for_equations) # проверяет валидность уравнения и запрашивает другие уравнения (если они остались)
async def equation_system(message: Message, state: FSMContext):
    user_data = await state.get_data()
    current_equation = user_data['current_equation']
    equations = user_data.get('equations', [])
    equation_count = user_data['equation_count']
    
    try:
        x = symbols('x')
        expr = sympify(message.text)
        lambdify(x, expr, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'tan': np.tan}])
        equations.append(message.text)
    except Exception as e:
        print(f'Ошибка: {str(e)}')
        await message.answer("Некорректное уравнение")
        return
    
    await state.update_data(equations=equations)
    
    if current_equation < equation_count:
        next_number = current_equation + 1
        await state.update_data(current_equation=next_number)
        await message.answer(f"Введите уравнение {next_number}:")
    else:
        await build_graph(message, equations, state)

async def main(): # главный цикл бота
    while True:
        try:
            print('Start Bot')
            await dp.start_polling(bot, polling_timeout=10)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Ошибка: {e}')
            print('Restart Bot')
            await asyncio.sleep(3)
            
if __name__ == "__main__":
    asyncio.run(main())