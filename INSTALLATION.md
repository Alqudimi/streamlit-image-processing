# دليل التثبيت

## المتطلبات الأساسية

### Python
يتطلب المشروع Python 3.8 أو أحدث. يمكنك تحميل Python من:
- [الموقع الرسمي لـ Python](https://www.python.org/downloads/)
- أو استخدام مدير بيئة مثل Anaconda/Miniconda

### التحقق من إصدار Python
```bash
python --version
# أو
python3 --version
```

## طرق التثبيت

### الطريقة 1: التثبيت المباشر

1. **استنساخ المستودع**
```bash
git clone https://github.com/Alqudimi/streamlit-image-processing.git
cd streamlit-image-processing
```

2. **إنشاء بيئة افتراضية (مستحسن)**
```bash
# باستخدام venv
python -m venv image_processing_env

# تفعيل البيئة
# على Windows:
image_processing_env\Scripts\activate
# على macOS/Linux:
source image_processing_env/bin/activate
```

3. **تثبيت المتطلبات**
```bash
pip install -r requirements.txt
```

4. **تشغيل التطبيق**
```bash
streamlit run app.py
```

### الطريقة 2: استخدام Conda

1. **استنساخ المستودع**
```bash
git clone https://github.com/Alqudimi/streamlit-image-processing.git
cd streamlit-image-processing
```

2. **إنشاء بيئة conda**
```bash
conda create -n image_processing python=3.9
conda activate image_processing
```

3. **تثبيت المتطلبات**
```bash
pip install -r requirements.txt
```

4. **تشغيل التطبيق**
```bash
streamlit run app.py
```

### الطريقة 3: استخدام Docker

1. **إنشاء Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **بناء وتشغيل الصورة**
```bash
docker build -t image-processing-app .
docker run -p 8501:8501 image-processing-app
```

## حل المشاكل الشائعة

### مشكلة تثبيت OpenCV
إذا واجهت مشاكل في تثبيت opencv-python:

```bash
# جرب هذا الأمر
pip install opencv-python-headless

# أو على بعض الأنظمة
pip install opencv-contrib-python
```

### مشكلة مع NumPy
```bash
pip install --upgrade numpy
```

### مشكلة مع Streamlit
```bash
pip install --upgrade streamlit
```

### مشاكل الأذونات على macOS/Linux
```bash
sudo pip install -r requirements.txt
# أو استخدم
pip install --user -r requirements.txt
```

## التحقق من التثبيت

بعد التثبيت، تحقق من عمل التطبيق:

1. **تشغيل التطبيق**
```bash
streamlit run app.py
```

2. **فتح المتصفح**
انتقل إلى `http://localhost:8501`

3. **اختبار المحاضرات**
- جرب تحميل صورة
- اختبر المحاضرات المختلفة
- تأكد من عمل التفاعلات

## التحديث

لتحديث المشروع:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## متطلبات النظام

### الحد الأدنى
- RAM: 2 GB
- مساحة القرص: 500 MB
- معالج: أي معالج حديث

### المستحسن
- RAM: 4 GB أو أكثر
- مساحة القرص: 1 GB
- معالج: متعدد النوى

## الدعم

إذا واجهت مشاكل في التثبيت:
1. تحقق من إصدار Python
2. تأكد من تفعيل البيئة الافتراضية
3. جرب إعادة تثبيت المتطلبات
4. افتح issue في GitHub مع تفاصيل المشكلة