import google_generativeai as genai

genai.configure(api_key="AIzaSyDMqCY75DyTbK49kOvd73A7oc0p1ATLOzY")
for m in genai.list_models():
    print(m.name)