# dashboard.py
import tkinter as tk
from tkinter import ttk, messagebox

# import your functions from the notebook file
# assuming your notebook is converted to flood_functions.py
from flood import get_current_weather, prepare_data, train_model, predict_future

# dummy dataset for now (replace with your actual dataframe)
import pandas as pd
df = pd.read_csv("dataset/flood-prediction.csv")  # make sure this exists or pass your df

# prepare and train model on startup
X, y, le = prepare_data(df)
flood_model = train_model(X, y)

# --- GUI ---
root = tk.Tk()
root.title("🌧️ Flood Risk Dashboard")
root.geometry("500x400")

# city input
tk.Label(root, text="City:").pack(pady=5)
city_entry = tk.Entry(root)
city_entry.pack(pady=5)

# location input for flood prediction
tk.Label(root, text="Location (from dataset):").pack(pady=5)
location_entry = tk.Entry(root)
location_entry.pack(pady=5)

# predict button
def predict():
    city = city_entry.get()
    location = location_entry.get()
    
    if not city or not location:
        messagebox.showerror("Error", "Please enter both city and location.")
        return
    
    try:
        # weather data
        weather = get_current_weather(city)
        temp = weather['current_temp']
        
        # flood prediction
        if location not in le.classes_:
            messagebox.showerror("Error", f"{location} not in dataset locations!")
            return
        
        loc_encoded = le.transform([location])[0]
        # for simplicity, use temp as Rainfall_mm (or you can add more realistic values)
        row = [[1, 1, loc_encoded, temp, 0, 0, 0]]  # dummy Month/Day/Rainfall/etc
        pred = flood_model.predict(row)[0]
        
        messagebox.showinfo("Prediction", f"🌡️ Temp: {temp}°C\n🌊 Flood Risk: {pred}")
        
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Button(root, text="Predict Flood Risk", command=predict).pack(pady=20)

root.mainloop()