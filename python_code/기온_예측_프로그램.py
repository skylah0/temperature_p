import tkinter as tk
from tkinter import ttk
import os,sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

# 옷차림 데이터 (온도 범위에 따른 추천)
CLOTHING_RECOMMENDATIONS = [
    {"temp_range": (28, float("inf")), "outerwear": "-", "top": "민소매, 반팔 티셔츠", "bottom": "반바지(핫팬츠), 짧은 치마", "others": "민소매 원피스, 린넨 재질 옷"},
    {"temp_range": (23, 27), "outerwear": "-", "top": "반팔 티셔츠, 얇은 셔츠, 얇은 긴팔 티셔츠", "bottom": "반바지, 면바지", "others": "-"},
    {"temp_range": (20, 22), "outerwear": "얇은 가디건", "top": "긴팔 티셔츠, 셔츠, 블라우스, 후드티", "bottom": "면바지, 슬랙스, 7부 바지, 청바지", "others": "-"},
    {"temp_range": (17, 19), "outerwear": "얇은 니트, 얇은 가디건, 얇은 재킷, 바람막이", "top": "후드티, 맨투맨", "bottom": "긴바지, 청바지, 슬랙스, 스키니진", "others": "-"},
    {"temp_range": (12, 16), "outerwear": "재킷, 가디건, 야상", "top": "스웨트 셔츠(맨투맨), 셔츠, 기모 후드티", "bottom": "청바지, 면바지", "others": "스타킹, 니트"},
    {"temp_range": (9, 11), "outerwear": "재킷, 야상, 점퍼, 트렌치 코트", "top": "-", "bottom": "청바지, 면바지, 검은색 스타킹, 기모 바지, 레이어드", "others": "니트"},
    {"temp_range": (5, 8), "outerwear": "(울)코트, 가죽 재킷", "top": "-", "bottom": "레깅스, 청바지, 두꺼운 바지, 기모", "others": "스카프, 플리스, 내복, 니트"},
    {"temp_range": (-float("inf"), 4), "outerwear": "패딩, 두꺼운 코트", "top": "-", "bottom": "-", "others": "누빔, 내복, 목도리, 장갑, 기모, 방한용품"}
]

# 옷차림 추천 함수
def recommend_clothing(temperature):
    for recommendation in CLOTHING_RECOMMENDATIONS:
        min_temp, max_temp = recommendation["temp_range"]
        if min_temp <= temperature <= max_temp:
            return recommendation
    return {"outerwear": "-", "top": "-", "bottom": "-", "others": "-"}

# CSV 파일 로드 함수
def load_csv_files(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    seoul_data = []
    busan_data = []
    for file in all_files:
        data = pd.read_csv(file)
        if "서울" in file:
            seoul_data.append(data)
        elif "부산" in file:
            busan_data.append(data)
    seoul_combined = pd.concat(seoul_data, ignore_index=True) if seoul_data else pd.DataFrame()
    busan_combined = pd.concat(busan_data, ignore_index=True) if busan_data else pd.DataFrame()
    return seoul_combined, busan_combined

# 결측값 처리
def handle_missing_values_temp_and_feel(data):
    return data.dropna(subset=['기온(°C)', '체감온도(°C)'])

# 데이터 전처리
def preprocess_data(data):
    X = data[['기온(°C)']].values
    y = data['체감온도(°C)'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 모델 학습 및 저장
def train_and_save_model(X_train, y_train, scaler, city):
    model = LinearRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, f"weather_model_{city}.pkl")
    joblib.dump(scaler, f"scaler_{city}.pkl")

# 미래 예측 함수
def predict_future_with_month_data(data, start_date, num_days, city):
    model = joblib.load(f"weather_model_{city}.pkl")
    scaler = joblib.load(f"scaler_{city}.pkl")
    start_date = datetime.strptime(start_date, "%Y%m%d")
    input_month = start_date.month
    data['일자'] = pd.to_datetime(data['일자'], errors='coerce')
    month_data = data[data['일자'].dt.month == input_month]
    if month_data.empty:
        raise ValueError(f"{city} 데이터가 없습니다.")
    min_temp = month_data['기온(°C)'].min()
    max_temp = month_data['기온(°C)'].max()
    future_temperatures = np.random.uniform(min_temp, max_temp, num_days).reshape(-1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    future_temperatures_scaled = scaler.transform(future_temperatures)
    predicted_feel_temp = model.predict(future_temperatures_scaled)
    return pd.DataFrame({
        "날짜": [date.strftime("%Y-%m-%d") for date in dates],
        "미래기온": future_temperatures.flatten(),
        "예측체감온도": predicted_feel_temp
    })

# GUI 함수
def open_gui(seoul_data, busan_data):
    def on_submit():
        try:
            start_date = date_entry.get()
            num_days = int(days_entry.get())
            city = city_var.get()
            if city == "서울":
                predictions = predict_future_with_month_data(seoul_data, start_date, num_days, "서울")
            elif city == "부산":
                predictions = predict_future_with_month_data(busan_data, start_date, num_days, "부산")
            else:
                raise ValueError("도시를 선택하세요.")
            for row in result_table.get_children():
                result_table.delete(row)
            for _, row in predictions.iterrows():
                result_table.insert("", "end", values=(row["날짜"], f"{row['미래기온']:.2f}", f"{row['예측체감온도']:.2f}"))
        except Exception:
            result_table.delete(*result_table.get_children())
            result_table.insert("", "end", values=("", "", ""))

    def on_item_click(event):
        selected_item = result_table.selection()
        if not selected_item:
            return
        selected_values = result_table.item(selected_item[0], "values")
        if selected_values[1]:
            temperature = float(selected_values[1])
            recommendations = recommend_clothing(temperature)
            msg = (
                f"기온: {temperature}℃\n"
                f"외투: {recommendations['outerwear']}\n"
                f"상의: {recommendations['top']}\n"
                f"하의: {recommendations['bottom']}\n"
                f"기타: {recommendations['others']}"
            )
            recommendation_window = tk.Toplevel(root)
            recommendation_window.configure(background="light yellow", padx=20, pady=20)
            recommendation_window.title("옷차림 추천")

            # 창을 화면 중앙에 배치
            rec_width = 300
            rec_height = 200
            rec_x = (recommendation_window.winfo_screenwidth() // 2) - (rec_width // 2)
            rec_y = (recommendation_window.winfo_screenheight() // 2) - (rec_height // 2)
            recommendation_window.geometry(f"{rec_width}x{rec_height}+{rec_x}+{rec_y}")

            # 아이콘 설정
            recommendation_window.iconbitmap(icon_path)

            tk.Label(recommendation_window, text=msg, bg="light yellow", fg="black", justify="center").pack(expand=True)

    root = tk.Tk()
    root.title("기온 예측 프로그램")
    root.configure(bg="light yellow")

    # 실행 파일 경로에 아이콘 파일이 있는 경우, 상대 경로로 설정
    global icon_path
    icon_path = os.path.join(os.path.dirname(sys.argv[0]), "icon.ico")
    root.iconbitmap(icon_path)


    # 창을 화면 중앙에 배치
    window_width = 635
    window_height = 420
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = (screen_width // 2) - (window_width // 2)
    y_position = (screen_height // 2) - (window_height // 2)
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    frame = tk.Frame(root, padx=10, pady=10, bg="light yellow")
    frame.pack(fill="both", expand=True)

    # 스타일 설정
    style = ttk.Style()
    style.configure("TCombobox", fieldbackground="white", bordercolor="orange", borderwidth=2, relief="solid")
    style.configure("Custom.Treeview", background="gold", fieldbackground="gold", bordercolor="orange", borderwidth=2, relief="solid", foreground="black")
    style.map("Custom.Treeview", background=[('selected', 'orange')], foreground=[('selected', 'white')])

    # 도시 선택
    tk.Label(frame, text="도시 선택:", bg="light yellow", fg="orange").grid(row=0, column=0, sticky="w", padx=5, pady=5)
    city_var = tk.StringVar(value="")
    city_menu = ttk.Combobox(frame, textvariable=city_var, values=["서울", "부산"], state="readonly", style="TCombobox")
    city_menu.grid(row=0, column=1, padx=5, pady=5)

    # 시작 날짜 입력
    tk.Label(frame, text="시작 날짜 (YYYYMMDD):", bg="light yellow", fg="orange").grid(row=1, column=0, sticky="w", padx=5, pady=5)
    date_entry = tk.Entry(frame, bg="white", highlightbackground="orange", highlightthickness=2, bd=0)
    date_entry.grid(row=1, column=1, padx=5, pady=5)

    # 예측 일수 입력
    tk.Label(frame, text="예측 일수:", bg="light yellow", fg="orange").grid(row=2, column=0, sticky="w", padx=5, pady=5)
    days_entry = tk.Entry(frame, bg="white", highlightbackground="orange", highlightthickness=2, bd=0)
    days_entry.grid(row=2, column=1, padx=5, pady=5)

    # 실행 버튼
    tk.Button(frame, text="예측 실행", command=on_submit, bg="orange", fg="black").grid(row=3, column=0, columnspan=2, pady=10)

    # 결과 테이블
    tk.Label(frame, text="예측 결과:", bg="light yellow", fg="orange").grid(row=4, column=0, columnspan=2, sticky="w")
    columns = ("날짜", "미래기온", "예측체감온도")
    result_table = ttk.Treeview(frame, columns=columns, show="headings", height=10, style="Custom.Treeview")
    result_table.heading("날짜", text="날짜")
    result_table.heading("미래기온", text="미래기온")
    result_table.heading("예측체감온도", text="예측체감온도")
    result_table.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

    for i in range(10):
        result_table.insert("", "end", values=("", "", ""))

    result_table.bind("<Double-1>", on_item_click)

    root.mainloop()

if __name__ == "__main__":
    folder_path = "./data"
    seoul_data, busan_data = load_csv_files(folder_path)
    seoul_data = handle_missing_values_temp_and_feel(seoul_data)
    busan_data = handle_missing_values_temp_and_feel(busan_data)
    seoul_X_train, seoul_X_test, seoul_y_train, seoul_y_test, seoul_scaler = preprocess_data(seoul_data)
    busan_X_train, busan_X_test, busan_y_train, busan_y_test, busan_scaler = preprocess_data(busan_data)
    train_and_save_model(seoul_X_train, seoul_y_train, seoul_scaler, "서울")
    train_and_save_model(busan_X_train, busan_y_train, busan_scaler, "부산")
    open_gui(seoul_data, busan_data)
