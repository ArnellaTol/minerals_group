import streamlit as st
import pandas as pd
import pickle 

from sklearn.preprocessing import MultiLabelBinarizer

st.set_page_config(
    page_title="Mineral group classification",
)



# Создание переменных session state
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()

if 'answer' not in st.session_state:
    st.session_state['answer'] = 0

if 'probability' not in st.session_state:
    st.session_state['probability'] = []

def reset_session_state():
    st.session_state['df'] = pd.DataFrame()
    st.session_state['answer'] = 0
    st.session_state['probability'] = []


# ML section start
other_column = ['cleavage']
columns_binarize = ['color', 'streak', 'lustre', 'transparency', 'fracture']
columns_range = ['mohs_hardness', 'density']

colors = {
    'Белый': 1,
    'Черный': 2,
    'Серый': 3,
    'Коричневый': 4,
    'Красный': 5,
    'Оранжевый': 6,
    'Желтый': 7,
    'Зеленый': 8,
    'Голубой': 9,
    'Синий': 10,
    'Фиолетовый': 11,
    'Розовый': 12,
    'Бесцветный': 13
}

lustres = {
    'Металлический': 1,
    'Стеклянный': 2,
    'Алмазный': 3,
    'Перламутровый': 4,
    'Шелковистый': 5,
    'Жирный': 6,
    'Восковый': 7,
    'Матовый': 8
}

transparency = {
    'Прозрачный': 1,
    'Полупрозрачный': 2,
    'Непрозрачный': 3
}

fractures = {
    'Ровный': 1,
    'Неровный': 2,
    'Ступенчатый': 3,
    'Занозистый': 4,
    'Игольчатый': 5,
    'Раковистый': 6,
    'Зернистый': 7,
    'Землистый': 8
}

cleavage = {
    'Весьма совершенная': 1,
    'Совершенная': 2,
    'Средняя': 3,
    'Несовершенная': 4,
    'Весьма несовершенная': 5,
    'Отсутствует': 6
}

groups = {
    1: 'Силикаты',
    2: 'Фосфаты',
    3: 'Карбонаты',
    4: 'Окислы (оксиды)',
    5: 'Сульфиды (дисульфиды)',
    6: 'Сульфаты',
    7: 'Галогениды',
    8: 'Самородные элементы'
}

model_file_path = 'lr_model_mineral_group_prediction.sav'
model = pickle.load(open(model_file_path, 'rb'))


class NumericRange:
    def __init__(self, start, end=None):
        self.start = float(start)
        self.end = float(end) if end is not None else self.start
        self.mean = (self.start + self.end) / 2
        self.range = self.end - self.start

    def __repr__(self):
        return f'{self.start}-{self.end}' if self.start != self.end else str(self.start)

def parse_numeric_range(value):
    parts = value.split('-')
    if len(parts) == 2:
        return NumericRange(parts[0], parts[1])
    else:
        return NumericRange(parts[0])
    

def apply_mlb(df, column, all_values):
    mlb = MultiLabelBinarizer(classes=all_values)
    transformed_data = mlb.fit_transform(df[column])
    transformed_df = pd.DataFrame(transformed_data, columns=[f"{column}_{cls}" for cls in mlb.classes_])
    return transformed_df

@st.cache_data
def predict_group(df):
    df_colors = apply_mlb(df, 'color', [1,2,3,4,5,6,7,8,9,10,11,12,13])
    df_streaks = apply_mlb(df, 'streak', [1,2,3,4,5,6,7,8,9,10,11,12,13])
    df_lustres = apply_mlb(df, 'lustre', [1,2,3,4,5,6,7,8])
    df_transparencies = apply_mlb(df, 'transparency', [1,2,3])
    df_fractures = apply_mlb(df, 'fracture', [1,2,3,4,5,6,7,8])
    df = pd.concat([df, df_colors, df_streaks, df_lustres, df_transparencies, df_fractures], axis=1)
    df.drop(['color', 'streak', 'lustre', 'transparency', 'fracture'], axis=1, inplace=True)

    df['mohs_hardness'] = df['mohs_hardness'].apply(parse_numeric_range)

    df['mohs_hardness_start'] = df['mohs_hardness'].apply(lambda x: x.start)
    df['mohs_hardness_end'] = df['mohs_hardness'].apply(lambda x: x.end)
    df['mohs_hardness_mean'] = df['mohs_hardness'].apply(lambda x: x.mean)
    df['mohs_hardness_range'] = df['mohs_hardness'].apply(lambda x: x.range)

    df.drop('mohs_hardness', axis=1, inplace=True)

    df['density'] = df['density'].apply(parse_numeric_range)

    df['density_start'] = df['density'].apply(lambda x: x.start)
    df['density_end'] = df['density'].apply(lambda x: x.end)
    df['density_mean'] = df['density'].apply(lambda x: x.mean)
    df['density_range'] = df['density'].apply(lambda x: x.range)

    df.drop('density', axis=1, inplace=True)

    answer = model.predict(df)
    probability = model.predict_proba(df)

    return answer, probability[0]


st.title('Определение группы минерала по его признакам на основе ИИ')
st.subheader('🗂 Ввод данных')
selected_text_colors = st.multiselect('Цвет(-а) минерала:', list(colors.keys()))
selected_text_streak = st.multiselect('Цвет(-а) черты:', list(colors.keys()))
selected_text_lustre = st.multiselect('Блеск:', list(lustres.keys()))
selected_text_transparency = st.multiselect('Прозрачность:', list(transparency.keys()))
selected_text_fracture = st.multiselect('Излом:', list(fractures.keys()))
selected_text_cleavage = st.selectbox('Спайность:', list(cleavage.keys()))

mohs_hardness_input = st.text_input('Твердость по шкале Мооса:', placeholder='1.5-2.0', help='Ввод БЕЗ пробелов!!\nДля диапазона использовать знак "-"\nДля десятичной дроби использовать знак точки "." НЕ запятой ","!!')
density_input = st.text_input('Плотность в г/см³:', placeholder='5.67-5.76', help='Ввод БЕЗ пробелов!!\nДля диапазона использовать знак "-"\nДля десятичной дроби использовать знак точки "." НЕ запятой ","!!')



selected_colors = [colors[i] for i in selected_text_colors]
selected_streak = [colors[i] for i in selected_text_streak]
selected_lustre = [lustres[i] for i in selected_text_lustre]
selected_transparency = [transparency[i] for i in selected_text_transparency]
selected_fracture = [fractures[i] for i in selected_text_fracture]
selected_cleavage = int(cleavage[selected_text_cleavage])



prediction_button = st.button('Определить группу', type='primary', use_container_width=True, key='button')
            
if prediction_button:

    st.session_state['df'] = pd.DataFrame({
        'color': [selected_colors], 
        'streak': [selected_streak], 
        'lustre': [selected_lustre], 
        'transparency': [selected_transparency], 
        'cleavage': int(selected_cleavage), 
        'mohs_hardness': str(mohs_hardness_input), 
        'density': str(density_input), 
        'fracture': [selected_fracture]
    }, index=[0])

    st.session_state['answer'], st.session_state['probability'] = predict_group(st.session_state['df'])

    st.subheader('Ответ:')
    st.write(groups[int(st.session_state['answer'])])

    st.subheader('Вероятности:')
    for i in range(0,8):
        st.write(f"{groups[i+1]} - {float(round(st.session_state['probability'][i], 5))*100}%")
