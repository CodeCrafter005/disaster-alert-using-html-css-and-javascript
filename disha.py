import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import requests

# 🔹 Force CPU to Avoid TensorFlow-Metal Issues on macOS
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_METAL_ENABLE"] = "0"

# ✅ Load Fire Detection Model
try:
    model_path = "/Users/ashutoshmahakhud/Desktop/fire_detection_model.keras"
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    print("🔹 Model expects input shape:", model.input_shape)  # Debugging
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# 🔹 Fix Image Preprocessing (Ensure Correct Input Shape)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(250, 250))  # Adjust to model input size if needed
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# 🔥 Run Fire Detection
def predict_fire(rgb_img_path):
    if not os.path.exists(rgb_img_path):
        print(f"❌ Error: RGB image not found at {rgb_img_path}")
        return "❌ Image Not Found"

    img_array = preprocess_image(rgb_img_path)
    try:
        prediction = model.predict(img_array)
        print("✅ Prediction completed! Raw Output:", prediction)  # Debugging
        return "🔥 Fire Detected" if prediction[0][0] > 0.5 else "✅ No Fire"
    except Exception as e:
        print("❌ Error during prediction:", e)
        return "❌ Prediction Failed"

# 📍 Extract GPS Coordinates
def get_exif_data(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == "GPSInfo":
                for t in value:
                    sub_tag = GPSTAGS.get(t, t)
                    gps_info[sub_tag] = value[t]
        return gps_info
    except Exception as e:
        print("❌ Error reading EXIF data:", e)
        return None

def convert_to_degrees(value):
    d, m, s = value
    return float(d) + (float(m) / 60.0) + (float(s) / 3600.0)

def get_image_location(image_path):
    gps_data = get_exif_data(image_path)
    if not gps_data:
        return None, None

    try:
        lat = convert_to_degrees(gps_data["GPSLatitude"])
        lon = convert_to_degrees(gps_data["GPSLongitude"])

        if gps_data.get("GPSLatitudeRef", "N") != "N":
            lat = -lat
        if gps_data.get("GPSLongitudeRef", "E") != "E":
            lon = -lon

        return lat, lon
    except KeyError:
        return None, None

# 🌍 Reverse Geocoding: Get Address from Coordinates
def get_address_from_coordinates(latitude, longitude):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={latitude}&lon={longitude}"
        headers = {"User-Agent": "fire-detection-app"}
        response = requests.get(url, headers=headers)
        data = response.json()

        if "display_name" in data:
            return data["display_name"]
        else:
            return "❌ No location found."
    except Exception as e:
        print("❌ Error getting location name:", e)
        return "❌ Failed to retrieve location."

# 🌡 Extract Temperature from Thermal Image
def extract_temperature_from_thermal(thermal_img_path):
    if not os.path.exists(thermal_img_path):
        print(f"❌ Error: Thermal image not found at {thermal_img_path}")
        return None

    try:
        thermal_img = cv2.imread(thermal_img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if thermal_img is None:
            print("❌ Error loading thermal image.")
            return None

        min_temp = 20.0  # Minimum temperature in the scale
        max_temp = 150.0  # Maximum temperature in the scale
        thermal_normalized = thermal_img / 255.0  # Normalize
        estimated_temp = min_temp + (thermal_normalized * (max_temp - min_temp))

        avg_temp = np.mean(estimated_temp)  # Compute average temperature
        return round(avg_temp, 2)
    except Exception as e:
        print("❌ Error processing thermal image:", e)
        return None

# 🔥🔥🔥 Predict Fire, Location, and Temperature
def predict_fire_and_temperature(rgb_img_path, thermal_img_path):
    print("\n🚀 Running Fire Detection on:", rgb_img_path)
    
    # 🔥 Fire Detection
    fire_prediction = predict_fire(rgb_img_path)

    # 📍 Get GPS Coordinates
    lat, lon = get_image_location(rgb_img_path)
    if lat is not None and lon is not None:
        location = get_address_from_coordinates(lat, lon)
        location_text = f"📍 {location}\n🌍 Lat: {lat:.6f}, Lon: {lon:.6f}"
    else:
        location_text = "❌ No Location Available"

    # 🌡 Get Temperature from Thermal Image
    estimated_temperature = extract_temperature_from_thermal(thermal_img_path)
    if estimated_temperature is not None:
        temperature_text = f"🌡 Estimated Temperature: {estimated_temperature}°C"
    else:
        temperature_text = "🌡 Temperature Not Available"

    return fire_prediction, location_text, temperature_text

# 🏁 Test Code
if __name__ == "__main__":
    rgb_img_path = "/Applications/TERABOX FOLDER/FLAME 3 CV Dataset (Sycan Marsh)-1/Fire/RGB/RAW/00001.JPG"  # Change to actual RGB image path
    thermal_img_path = "/Applications/TERABOX FOLDER/FLAME 3 CV Dataset (Sycan Marsh)-1/Fire/Thermal/RAW JPG/00001.JPG"  # Change to actual Thermal image path

    # 🔎 Check if images exist before running
    if not os.path.exists(rgb_img_path):
        print(f"❌ Error: RGB image not found at {rgb_img_path}")
        exit()

    if not os.path.exists(thermal_img_path):
        print(f"❌ Error: Thermal image not found at {thermal_img_path}")
        exit()

    print("🚀 Starting Fire Detection Script...")
    
    # 🔥 Predict Fire, Location, and Temperature
    result, location_text, temperature_text = predict_fire_and_temperature(rgb_img_path, thermal_img_path)
    
    print("\n🔥 Prediction:", result)
    print(location_text)
    print(temperature_text)