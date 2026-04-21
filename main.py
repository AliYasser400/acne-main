from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from gradio_client import Client as GradioClient, handle_file
from pydantic import BaseModel
from typing import Optional
import shutil
import os
import json
import ast
import sys
import io
from dotenv import load_dotenv

# Fix Windows console encoding issue for emojis
if sys.platform == 'win32':
    # Set UTF-8 encoding for stdout/stderr
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# تحميل متغيرات البيئة
load_dotenv()

app = FastAPI()

# ---------------------------------------------------------
# 1. تفعيل CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 2. إعدادات Supabase
# ---------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("[ERROR] Supabase keys are missing! Check .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------
# 3. إعدادات الموديل (Lazy Loading + Singleton Pattern) 🚀
# ---------------------------------------------------------
# متغير لتخزين الاتصال حتى لا نعيد الاتصال كل مرة
_model_client_instance = None

def get_model_client():
    global _model_client_instance
    # لو الاتصال غير موجود، قم بإنشائه (يحدث مرة واحدة فقط)
    if _model_client_instance is None:
        print("[INFO] Connecting to Hugging Face Model (First Time)...")
        try:
            _model_client_instance = GradioClient("m-taha6/monkeypox")
            print("[SUCCESS] Connected Successfully to Hugging Face!")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Hugging Face model: {e}")
            raise
    
    return _model_client_instance

# ---------------------------------------------------------
# 4. قاعدة البيانات الطبية (Medical Knowledge Base)
# ---------------------------------------------------------
MEDICAL_REPORT_DATA = {
    "Monkeypox": {
        "assessment": "The analysis indicates signs consistent with Monkeypox. Lesions typically progress from macules to papules, vesicles, pustules, and then scabs.",
        "key_features": [
            "Deep-seated, firm/hard lesions",
            "Well-defined borders with central umbilication",
            "Lesions often start on the face/mouth and spread",
            "Swollen lymph nodes (lymphadenopathy)"
        ],
        "recommendations": [
            "Isolate immediately to prevent spread.",
            "Wear a mask and cover lesions.",
            "Consult a healthcare provider for PCR testing.",
            "Monitor for fever and other systemic symptoms."
        ]
    },
    "Chickenpox": {
        "assessment": "The skin shows signs characteristic of Chickenpox (Varicella). This usually presents as an itchy, blister-like rash.",
        "key_features": [
            "Rash appears in crops (different stages visible)",
            "Superficial dew-drop on a rose petal appearance",
            "Intense itching",
            "Usually starts on the trunk and spreads"
        ],
        "recommendations": [
            "Stay at home until all blisters have crusted over.",
            "Use calamine lotion to soothe itching.",
            "Avoid scratching to prevent secondary infection.",
            "Avoid contact with pregnant women or immunocompromised individuals."
        ]
    },
    "Measles": {
        "assessment": "The analysis suggests Measles. This is a highly contagious viral infection appearing as a flat, red rash.",
        "key_features": [
            "Flat red rash starting at hairline/face",
            "Spreads downwards to neck, trunk, and limbs",
            "Associated with high fever, cough, and runny nose",
            "Tiny white spots inside the mouth (Koplik spots)"
        ],
        "recommendations": [
            "Seek medical attention immediately (highly contagious).",
            "Isolate from others, especially unvaccinated individuals.",
            "Rest and maintain hydration.",
            "Vitamin A supplements may be prescribed by a doctor."
        ]
    },
    "Normal": {
        "assessment": "The skin appears healthy with no signs of pathological rashes related to Monkeypox, Chickenpox, or Measles.",
        "key_features": [
            "Clear skin texture",
            "No suspicious lesions or blisters",
            "Normal pigmentation"
        ],
        "recommendations": [
            "Continue regular skin hygiene routine.",
            "Use sunscreen when exposed to the sun.",
            "Perform regular self-checks for any changes.",
            "Stay hydrated to maintain skin health."
        ]
    }
}

@app.get("/")
def home():
    return {"message": "Skin Disease Classification API is Running!"}

@app.get("/test-model")
def test_model_connection():
    """Test endpoint to check if the model connection is working"""
    try:
        print("[INFO] Testing model connection...")
        hf_client = get_model_client()
        print("[SUCCESS] Model client created successfully")
        
        # محاولة الحصول على معلومات عن الـ API
        try:
            api_info = hf_client.view_api()
            return {
                "status": "success",
                "message": "Model connection is working",
                "model": "m-taha6/monkeypox",
                "api_info": str(api_info)[:500] if api_info else "No API info available"
            }
        except Exception as api_error:
            return {
                "status": "partial_success",
                "message": "Model client created but API info unavailable",
                "model": "m-taha6/monkeypox",
                "error": str(api_error)
            }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": "Failed to connect to model",
            "error": str(e),
            "traceback": traceback.format_exc()
        }

# =========================================================
# 5. الـ Scan Endpoint
# =========================================================
@app.post("/scan")
async def scan_face(user_id: str = Form(...), file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # أ) حفظ الصورة مؤقتاً
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ب) استدعاء الموديل (باستخدام الدالة الذكية)
        print("[INFO] Analyzing Image...")
        hf_client = get_model_client()
        
        print(f"[INFO] Sending image to model: {temp_filename}")
        result = hf_client.predict(
            image=handle_file(temp_filename),
            api_name="/predict" 
        )
        
        # ج) استخراج النتيجة - تحسين معالجة النتائج
        print(f"[INFO] Raw model result type: {type(result)}")
        print(f"[INFO] Raw model result: {result}")
        
        predicted_diagnosis = None
        confidence = 0.0
        
        # محاولة استخراج النتيجة بطرق مختلفة
        try:
            # الحالة 1: إذا كانت النتيجة dictionary مباشرة
            if isinstance(result, dict):
                print("[INFO] Result is a dictionary")
                if 'label' in result:
                    predicted_diagnosis = result['label']
                elif 'prediction' in result:
                    predicted_diagnosis = result['prediction']
                elif len(result) > 0:
                    # أخذ أول مفتاح كتشخيص
                    predicted_diagnosis = list(result.keys())[0]
                
                # استخراج الثقة إذا كانت موجودة
                if 'confidence' in result:
                    confidence = float(result['confidence'])
                elif 'score' in result:
                    confidence = float(result['score'])
                    
            # الحالة 2: إذا كانت النتيجة string
            elif isinstance(result, str):
                print("[INFO] Result is a string")
                # محاولة تحويل string إلى dict
                try:
                    # محاولة JSON
                    res_dict = json.loads(result)
                    if isinstance(res_dict, dict):
                        if 'label' in res_dict:
                            predicted_diagnosis = res_dict['label']
                        elif 'prediction' in res_dict:
                            predicted_diagnosis = res_dict['prediction']
                except:
                    try:
                        # محاولة ast.literal_eval
                        res_dict = ast.literal_eval(result)
                        if isinstance(res_dict, dict):
                            if 'label' in res_dict:
                                predicted_diagnosis = res_dict['label']
                            elif 'prediction' in res_dict:
                                predicted_diagnosis = res_dict['prediction']
                    except:
                        # إذا فشل كل شيء، استخدم النتيجة كما هي
                        predicted_diagnosis = result.strip()
            
            # الحالة 3: إذا كانت النتيجة list
            elif isinstance(result, (list, tuple)):
                print("[INFO] Result is a list/tuple")
                if len(result) > 0:
                    predicted_diagnosis = str(result[0])
                    if len(result) > 1:
                        try:
                            confidence = float(result[1])
                        except:
                            pass
            
            # الحالة 4: أي نوع آخر
            else:
                print(f"[WARNING] Unexpected result type: {type(result)}")
                predicted_diagnosis = str(result)
                
        except Exception as parse_error:
            print(f"[ERROR] Error parsing result: {parse_error}")
            predicted_diagnosis = str(result)
        
        # إذا لم نتمكن من استخراج التشخيص
        if not predicted_diagnosis or predicted_diagnosis == "None":
            print("[WARNING] Could not extract diagnosis, using raw result")
            predicted_diagnosis = str(result) if result else "Unknown"
        
        # تنظيف التشخيص (إزالة علامات الاقتباس والمسافات)
        predicted_diagnosis = str(predicted_diagnosis).strip().strip('"').strip("'")
        
        # إذا كانت الثقة 0، ضع قيمة افتراضية
        if confidence == 0.0:
            confidence = 0.95
            
        print(f"[SUCCESS] Diagnosis Detected: {predicted_diagnosis}")
        print(f"[SUCCESS] Confidence: {confidence}")

        # د) جلب التقرير الطبي
        report_data = MEDICAL_REPORT_DATA.get(predicted_diagnosis, {
            "assessment": "Analysis completed. Diagnosis not specifically listed.",
            "key_features": [],
            "recommendations": ["Consult a doctor for further checkup."]
        })

        # هـ) رفع الصورة لـ Supabase
        BUCKET_NAME = "skin-diseases"
        print(f"[INFO] Uploading Image to {BUCKET_NAME}...")
        
        with open(temp_filename, "rb") as f:
            file_content = f.read()
        
        file_path = f"{user_id}/{file.filename}"
        
        supabase.storage.from_(BUCKET_NAME).upload(file_path, file_content, {"upsert": "true"})
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)

        # و) حفظ البيانات في الهيستوري
        print("[INFO] Saving to History...")
        data = {
            "user_id": user_id,
            "image_url": public_url,
            "diagnosis": predicted_diagnosis,
            "confidence": float(confidence),
            "medical_advice": json.dumps(report_data) 
        }
        
        supabase.table("scan_history").insert(data).execute()

        # ز) تنظيف
        os.remove(temp_filename)

        return {
            "status": "success",
            "diagnosis": predicted_diagnosis,
            "image_url": public_url,
            "report": report_data 
        }

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[ERROR] Error in scan endpoint: {str(e)}")
        print(f"[ERROR] Full traceback:\n{error_details}")
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
        return {
            "status": "error", 
            "message": str(e),
            "error_type": type(e).__name__,
            "details": error_details if "DEBUG" in os.environ else None
        }

# =========================================================
# 6. إدارة البروفايل (Profile Management) - كاملة وشاملة
# =========================================================
class ProfileUpdate(BaseModel):
    user_id: str
    full_name: Optional[str] = None
    username: Optional[str] = None
    website: Optional[str] = None
    # الحقول الجديدة
    age: Optional[int] = None
    gender: Optional[str] = None
    skin_type: Optional[str] = None
    role: Optional[str] = None
    phone: Optional[str] = None
    city: Optional[str] = None 

@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    try:
        response = supabase.table("profiles").select("*").eq("id", user_id).execute()
        if not response.data:
            return {"status": "error", "message": "Profile not found"}
        return {"status": "success", "data": response.data[0]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/profile/update")
def update_profile(profile: ProfileUpdate):
    try:
        # تصفية البيانات (إرسال القيم الموجودة فقط)
        data_to_update = {k: v for k, v in profile.dict().items() if v is not None and k != "user_id"}
        
        if not data_to_update:
            return {"status": "error", "message": "No data to update"}

        print(f"[INFO] Updating Profile for {profile.user_id}: {data_to_update}")

        response = supabase.table("profiles").update(data_to_update).eq("id", profile.user_id).execute()
        return {"status": "success", "data": response.data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =========================================================
# 7. جلب الهيستوري (History)
# =========================================================
@app.get("/history/{user_id}")
def get_user_history(user_id: str):
    try:
        response = supabase.table("scan_history").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        final_data = []
        for item in response.data:
            if item.get("medical_advice"):
                try:
                    item["medical_advice"] = json.loads(item["medical_advice"])
                except:
                    pass
            final_data.append(item)
            
        return {"status": "success", "data": final_data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

