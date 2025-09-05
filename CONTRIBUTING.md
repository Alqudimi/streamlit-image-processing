# المساهمة في مشروع Interactive Image Processing Lectures

شكراً لاهتمامك بالمساهمة في هذا المشروع! نحن نرحب بجميع أشكال المساهمات.

## كيفية المساهمة

### الإبلاغ عن الأخطاء

عند الإبلاغ عن خطأ، يرجى تضمين:
- وصف واضح للمشكلة
- خطوات إعادة إنتاج المشكلة
- النتيجة المتوقعة والنتيجة الفعلية
- معلومات البيئة (نظام التشغيل، إصدار Python، إلخ)

### اقتراح ميزات جديدة

لاقتراح ميزة جديدة:
- تأكد من عدم وجود طلب مماثل
- اشرح الميزة المقترحة بالتفصيل
- اذكر الفائدة التعليمية من الميزة
- قدم أمثلة عن كيفية استخدامها

### المساهمة بالكود

1. **Fork المشروع**
2. **أنشئ branch جديد**
   ```bash
   git checkout -b feature/new-feature
   ```
3. **اكتب التغييرات**
4. **تأكد من عمل الكود**
5. **اتبع معايير الكود**
6. **أرسل Pull Request**

## معايير الكود

### Python Code Style
- اتبع PEP 8
- استخدم أسماء متغيرات واضحة
- أضف تعليقات للكود المعقد
- استخدم type hints عند الإمكان

### Streamlit Components
- استخدم `use_container_width=True` بدلاً من `use_column_width`
- اتبع نمط المحاضرات الموجود
- تأكد من التوافق مع الواجهة العامة

### إضافة محاضرة جديدة

عند إضافة محاضرة جديدة:

1. **أنشئ ملف في `lectures/`**
   ```python
   def show():
       st.header("Lecture X: Title")
       
       # Theory section
       st.subheader("Theory")
       st.write("...")
       
       # Interactive demo
       st.subheader("Interactive Demo")
       # ... demo code
       
       # Key takeaways
       st.subheader("Key Takeaways")
       st.write("...")
   ```

2. **أضف استيراد في `app.py`**
3. **أضف المحاضرة لقائمة التنقل**
4. **اختبر التشغيل**

### إضافة عملية معالجة جديدة

1. **أضف الدالة في `utils/processing.py`**
2. **اتبع نمط الدوال الموجودة**
3. **أضف documentation string**
4. **اختبر الدالة**

## اختبار التغييرات

قبل إرسال Pull Request:

1. **اختبر التطبيق محلياً**
   ```bash
   streamlit run app.py
   ```

2. **تأكد من عمل جميع المحاضرات**
3. **اختبر رفع الصور**
4. **تأكد من عدم وجود أخطاء في الكونسول**

## أسلوب Commit Messages

استخدم رسائل commit واضحة:
- `feat: add new lecture on advanced filtering`
- `fix: resolve image upload issue in lecture 3`
- `docs: update README with new features`
- `style: improve UI layout in lecture 2`

## المراجعة

سيتم مراجعة جميع Pull Requests من قبل المشرفين. قد نطلب تعديلات أو توضيحات قبل الدمج.

## أسئلة؟

إذا كان لديك أي أسئلة، لا تتردد في:
- فتح issue للنقاش
- التواصل مع المطور

شكراً لمساهمتك!