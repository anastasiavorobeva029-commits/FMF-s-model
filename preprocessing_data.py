import pandas as pd

def fix_value(val):
    if pd.isna(val) or val == '':
        return 0.0

    s = str(val).replace(',', '').replace('.', '').strip()

    if not s:
        return 0.0

    if len(s) > 2 and 10 <= int(s[:2]) <= 50:
        return float(s[:2] + '.' + s[2:])

    return float(s[0] + '.' + s[1:])


def process_demographics():
    files = {
        'birth_rate_full_1950_2125.csv': 'Predicted_birth_rate',
        'death_rate_full_1950_2125.csv': 'Predicted_death_rate',
        'fertility_rate_full_1950_2125.csv': 'Predicted_fertility_rate'
    }

    for file_name, col_name in files.items():
        print(f"Обработка {file_name}...")


        df = pd.read_csv(file_name, sep=';')


        df[col_name] = df[col_name].apply(fix_value)


        df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(float)


        new_name = 'fixed_' + file_name
        df.to_csv(new_name, sep=';', index=False)
        print(f"✅ Готово! Исправленный файл: {new_name}")


files = [
    'birth_rate_full_1950_2125.csv',
    'death_rate_full_1950_2125.csv',
    'fertility_rate_full_1950_2125.csv',
    'age_structure_1950.csv',  # замените на реальное имя файла с percentage
    'age_fertility_dist.csv'  # замените на реальное имя файла с fertility_rate
]

for file in files:
    # Загружаем файл (учитываем ваш разделитель ;)
    df = pd.read_csv(file, sep=';')

    # Находим целевой столбец (обычно это первый или второй)
    # В ваших файлах это Predicted_..., percentage или fertility_rate
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'str':
            # 1. Убираем лишние пробелы по краям
            df[col] = df[col].astype(str).str.strip()
            # 2. Заменяем запятую на точку (если она там есть)
            df[col] = df[col].str.replace(',', '.')
            # 3. Превращаем в число. errors='coerce' превратит ошибки в NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"--- Файл {file} обработан ---")
    print(df.dtypes)
    print("-" * 30)

    # Чтобы сохранить изменения обратно в файлы, раскомментируйте строку ниже:
    df.to_csv(file, sep=';', index=False)

df = pd.read_csv('birth_rate_full_1950_2125.csv', sep=';')
df_1 = pd.read_csv('death_rate_full_1950_2125.csv', sep=';')
df_2 = pd.read_csv('fertility_rate_full_1950_2125.csv', sep=';')
df_3 = pd.read_csv('age_structure_1950.csv', sep=';')
df_4 = pd.read_csv('age_fertility_dist.csv', sep=';')

# Выводим общую информацию
df.info()
df_1.info()
df_2.info()
df_3.info()
df_4.info()
# if __name__ == "__main__":
#     process_demographics()