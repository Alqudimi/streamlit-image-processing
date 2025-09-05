# دليل النشر على Streamlit Cloud

## متطلبات النشر

للنشر على استضافة Streamlit المجانية، تحتاج للملفات التالية:

### 1. ملفات التكوين المطلوبة

#### `pyproject.toml` (موجود)
يحتوي على متطلبات المشروع والمكتبات المطلوبة.

#### `packages.txt` (تم إنشاؤه)
```
libgl1-mesa-glx
libglib2.0-0
```
يحتوي على حزم النظام المطلوبة لـ OpenCV.

#### `runtime.txt` (تم إنشاؤه)
```
python-3.11
```
يحدد إصدار Python المطلوب.

#### `.streamlit/config.toml` (موجود)
إعدادات Streamlit للخادم.

### 2. ملفات بديلة للنشر

#### `streamlit_requirements.txt` (تم إنشاؤه)
ملف متطلبات منفصل يمكن استخدامه مع Streamlit Cloud:
```
streamlit>=1.28.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
```

#### `setup.sh` (تم إنشاؤه)
سكريبت إعداد للنشر على منصات أخرى.

#### `Procfile` (تم إنشاؤه)
ملف تكوين لـ Heroku إذا أردت النشر هناك.

## خطوات النشر على Streamlit Cloud

### 1. تحضير المستودع

1. **رفع الكود على GitHub**
   ```bash
   git add .
   git commit -m "تحضير المشروع للنشر"
   git push origin main
   ```

2. **التأكد من وجود الملفات المطلوبة**
   - ✅ `app.py` (الملف الرئيسي)
   - ✅ `pyproject.toml` (المتطلبات)
   - ✅ `packages.txt` (حزم النظام)
   - ✅ `runtime.txt` (إصدار Python)
   - ✅ `.streamlit/config.toml` (إعدادات Streamlit)

### 2. النشر على Streamlit Cloud

1. **الذهاب إلى [share.streamlit.io](https://share.streamlit.io)**

2. **تسجيل الدخول بحساب GitHub**

3. **إنشاء تطبيق جديد**
   - اختر المستودع: `Alqudimi/streamlit-image-processing`
   - اختر الفرع: `main`
   - الملف الرئيسي: `app.py`

4. **النقر على "Deploy"**

### 3. إعدادات متقدمة (اختيارية)

في إعدادات التطبيق على Streamlit Cloud:

```toml
[server]
headless = true
port = 8501

[browser]
gatherUsageStats = false
```

## حل المشاكل الشائعة

### مشكلة OpenCV
إذا واجهت مشاكل مع OpenCV، تأكد من:
- استخدام `opencv-python-headless` بدلاً من `opencv-python`
- وجود `packages.txt` مع المكتبات المطلوبة

### مشكلة الذاكرة
Streamlit Cloud لها حدود للذاكرة. إذا واجهت مشاكل:
- قلل حجم الصور الافتراضية
- استخدم تحسينات الذاكرة في معالجة الصور

### مشكلة المتطلبات
إذا فشل تثبيت المتطلبات:
- تأكد من صحة `pyproject.toml`
- جرب استخدام `streamlit_requirements.txt`

## بدائل أخرى للنشر

### 1. Heroku
استخدم `Procfile` و `setup.sh` للنشر على Heroku.

### 2. Railway
يدعم `pyproject.toml` مباشرة.

### 3. Render
يدعم `requirements.txt` و `pyproject.toml`.

## نصائح للنشر الناجح

1. **اختبر محلياً أولاً**
   ```bash
   streamlit run app.py
   ```

2. **تأكد من عمل جميع المحاضرات**

3. **اختبر رفع الصور**

4. **تحقق من عدم وجود أخطاء في الكونسول**

5. **استخدم صور افتراضية صغيرة الحجم**

المشروع جاهز الآن للنشر على استضافة Streamlit المجانية!