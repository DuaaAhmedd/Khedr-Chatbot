from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import sqlite3
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from typing import List
import logging
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Khadr Chatbot API", description="API لتطبيق خضر لزراعة الأشجار بالعربية")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    text: str

class ChatHistory(BaseModel):
    text: str
    label: str
    timestamp: str

try:
    with open("train_embeddings.pkl", "rb") as f:
        train_embeddings = pickle.load(f)
    with open("training_data.pkl", "rb") as f:
        training_data = pickle.load(f)
except Exception as e:
    logger.error(f"خطأ في تحميل ملفات الإمبدنجز أو بيانات التدريب: {str(e)}")
    raise

def init_db():
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                label TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        conn.commit()
    except Exception as e:
        logger.error(f"خطأ في تهيئة قاعدة البيانات: {str(e)}")
        raise
    finally:
        conn.close()

def save_interaction(user_id: str, text: str, label: str):
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO chat_history (user_id, text, label, timestamp) VALUES (?, ?, ?, ?)",
                  (user_id, text, label, timestamp))
        conn.commit()
    except Exception as e:
        logger.error(f"خطأ في حفظ التفاعل: {str(e)}")
        raise
    finally:
        conn.close()

def get_user_history(user_id: str, limit: int = 5) -> List[ChatHistory]:
    try:
        conn = sqlite3.connect("chat_history.db")
        c = conn.cursor()
        c.execute("SELECT text, label, timestamp FROM chat_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                  (user_id, limit))
        history = c.fetchall()
        return [ChatHistory(text=row[0], label=row[1], timestamp=row[2]) for row in history]
    except Exception as e:
        logger.error(f"خطأ في استرجاع السجل: {str(e)}")
        raise
    finally:
        conn.close()

model_path = "./trained_model"
try:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"مجلد المودل {model_path} غير موجود.")
    if not os.path.exists("label_encoder.npy"):
        raise FileNotFoundError("label_encoder.npy غير موجود.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    label_encoder = np.load("label_encoder.npy", allow_pickle=True)
except Exception as e:
    logger.error(f"خطأ في تحميل المودل أو label encoder: {str(e)}")
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

try:
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None,
                         device=0 if torch.cuda.is_available() else -1)
except Exception as e:
    logger.error(f"خطأ في تهيئة pipeline التصنيف: {str(e)}")
    raise

def classify_intent_with_embeddings(user_input: str, user_id: str) -> str:
    try:
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.base_model(**inputs).last_hidden_state
            input_embedding = outputs[:, 0, :].cpu().numpy()
        similarities = cosine_similarity(input_embedding, train_embeddings)
        max_similarity_idx = np.argmax(similarities[0])
        intent = training_data[max_similarity_idx]["label"]
        save_interaction(user_id, user_input, intent)
        return intent
    except Exception as e:
        logger.error(f"خطأ في تصنيف النية باستخدام الإمبدنجز: {str(e)}")
        return "unknown"
    
def chatbot_response(user_input: str, user_id: str) -> str:
    try:
        user_input = user_input.strip().lower()
        intent = classify_intent_with_embeddings(user_input, user_id)
        save_interaction(user_id, user_input, intent)
        history = get_user_history(user_id)
        if intent == "greeting":
            base_response = ("وعليكم السلام! أنا خضر، هنا لمساعدتك في زراعة الأشجار. "
                             "ما الذي تريد مناقشته؟ (زراعة، أمراض، أنواع للشوارع، العناية والسماد)")
            contextual_message = base_response if not history else (
                f"آخر مرة تكلمنا عن {history[0].label}، دلوقتي عايز/ة تبدأ من جديد؟ {base_response}")
        elif intent == "planting":
            base_response = ("لزراعة شجرة: \n"
                             "1. اختر مكانًا به شمس كافية وتربة جيدة التصريف.\n"
                             "2. احفر حفرة بعمق وعرض ضعف حجم جذور الشجرة.\n"
                             "3. ضع الشجرة في الحفرة، ثم أضف التربة وادكها برفق.\n"
                             "4. اسقِ الشجرة جيدًا وأضف طبقة من المهاد (مثل القش) لحفظ الرطوبة.")
            contextual_message = base_response if not history or history[0].label != "planting" else (
                f"يبدو إنك مهتم/ة بـزراعة الأشجار! تحدثنا عنها قبل كده. إليك نصيحة جديدة: {base_response}")
        elif intent == "diseases":
            if "البياض الدقيقي" in user_input:
                base_response = "البياض الدقيقي: بقع بيضاء على الأوراق، العلاج: مبيدات فطرية."
            elif "العفن الجذري" in user_input:
                base_response = "العفن الجذري: بسبب الري الزائد، الحل: تحسين التصريف."
            elif "تسوس الجذع" in user_input:
                base_response = "تسوس الجذع: بسبب الفطريات، الحل: تقليم الأجزاء المصابة."
            else:
                base_response = ("من أشهر أمراض الأشجار:\n"
                                 "- العفن الجذري: بسبب الري الزائد، الحل: تحسين التصريف.\n"
                                 "- البياض الدقيقي: بقع بيضاء على الأوراق، العلاج: مبيدات فطرية.\n"
                                 "- تسوس الجذع: بسبب الفطريات، الحل: تقليم الأجزاء المصابة.\n"
                                 "هل تريد تفاصيل عن مرض معين؟")
            contextual_message = base_response if not history or history[0].label != "diseases" else (
                f"يبدو إنك مهتم/ة بأمراض الأشجار! تحدثنا عنها قبل كده. إليك معلومات إضافية: {base_response}")
        elif intent == "tree_types":
            base_response = ("أنواع مناسبة للشوارع:\n"
                             "- الزنزلخت: سريع النمو، ظل كثيف، يتحمل الجفاف.\n"
                             "- الكافور: أوراق عطرية، مناسب للمدن.\n"
                             "- النخيل: جمالي ويتحمل الحرارة.\n"
                             "- الأكاسيا: قوية ومناسبة للمناطق الجافة.\n"
                             "اختر نوعًا للشارع بناءً على مناخ منطقتك!")
            contextual_message = base_response if not history or history[0].label != "tree_types" else (
                f"يبدو إنك مهتم/ة بأنواع الأشجار! تحدثنا عنها قبل كده. إليك نصيحة جديدة: {base_response}")
        elif intent == "care":
            if "كيف أعتني" in user_input:
                base_response = "العناية بالأشجار تتطلب اهتمامًا دقيقًا ومستمرًا لضمان نموها بشكل صحي وقوي..."
            elif "متى أقلم" in user_input or "تقليم" in user_input:
                base_response = "التقليم عملية أساسية للحفاظ على صحة الشجرة وجمالها..."
            elif "من" in user_input or "كم مرة أسقي" in user_input:
                base_response = "الري المنتظم يعتمد على عمر الشجرة وحالة المناخ..."
            elif "تربة" in user_input:
                base_response = "الاهتمام بالتربة يشمل فحصها دوريًا..."
            elif "سماد" in user_input or "ما السماد" in user_input:
                base_response = "السماد المناسب يختلف حسب احتياجات الشجرة..."
            else:
                base_response = "العناية بالأشجار تشمل الري المنتظم، التقليم، والتسميد..."
            contextual_message = base_response if not history or history[0].label != "care" else (
                f"يبدو إنك مهتم/ة بالعناية بالأشجار! تحدثنا عنها قبل كده. إليك نصيحة جديدة: {base_response}")
        elif intent == "farewell":
            contextual_message = "مع السلامة! أتمنى أن أكون قد ساعدتك في زراعة أشجارك."
        else:
            contextual_message = ("آسف، لم أفهم. يمكنك السؤال عن (طرق الزراعة، أمراض الأشجار، أنواع للشوارع، العناية والسماد). "
                                  "ما الذي تريد مناقشته؟")
        return contextual_message
    except Exception as e:
        logger.error(f"خطأ في إنشاء الاستجابة: {str(e)}")
        raise
@app.post("/chat", response_model=dict)
async def chat(request: ChatRequest):
    try:
        response = chatbot_response(request.text, request.user_id)
        return {"response": response}
    except Exception as e:
        logger.error(f"خطأ في نقطة نهاية /chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في معالجة الطلب: {str(e)}")

@app.post("/classify_embedding", response_model=dict)
async def classify_embedding(request: ChatRequest):
    try:
        intent = classify_intent_with_embeddings(request.text, request.user_id)
        return {"intent": intent}
    except Exception as e:
        logger.error(f"خطأ في تصنيف النية باستخدام الإمبدنجز: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في تصنيف النية: {str(e)}")
@app.get("/history/{user_id}", response_model=List[ChatHistory])
async def get_history(user_id: str, limit: int = 1000):
    try:
        history = get_user_history(user_id, limit)
        return history
    except Exception as e:
        logger.error(f"خطأ في نقطة نهاية /history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"خطأ في استرجاع السجل: {str(e)}")
