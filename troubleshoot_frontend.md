# Frontend White Screen Troubleshooting

## **🚨 Quick Fixes for White Screen**

### **Step 1: Check Browser Console**
1. Open your browser (Chrome/Firefox/Safari)
2. Go to: http://localhost:3000
3. Press **F12** (or **Cmd+Option+I** on Mac)
4. Click **"Console"** tab
5. Look for red error messages

### **Step 2: Hard Refresh**
1. Press **Cmd+Shift+R** (Mac) or **Ctrl+Shift+R** (Windows)
2. This clears browser cache

### **Step 3: Try Different Browser**
- Chrome: http://localhost:3000
- Firefox: http://localhost:3000  
- Safari: http://localhost:3000

### **Step 4: Check Server Status**
The server is running! You should see:
```
✅ Frontend: http://localhost:3000
✅ Backend: http://localhost:8000
```

## **🔧 Manual Restart (If Still White Screen)**

### **Terminal 1 - Stop and Restart Frontend**
```bash
# 1. Stop any existing process (Ctrl+C if running)
cd omnibio-frontend

# 2. Clear cache and restart
rm -rf .next
npm run dev
```

### **Terminal 2 - Check Backend**
```bash
cd biomarker/api
conda activate metabo
python -m uvicorn main:app --reload --port 8000
```

## **🧪 Test the Fix**

1. **Open Browser**: http://localhost:3000
2. **Expected**: Should see OmniBio login page
3. **If still white**: Check browser console for errors

## **📱 Mobile/Responsive Issues**
- Try desktop browser first
- Check browser zoom level (should be 100%)
- Disable browser extensions temporarily

## **🔍 Common Error Messages & Fixes**

### **"Module not found"**
```bash
cd omnibio-frontend
npm install
npm run dev
```

### **"Port 3000 already in use"**
```bash
# Kill existing process
lsof -ti:3000 | xargs kill
npm run dev
```

### **"JavaScript error in console"**
- Check if you have ad blockers
- Try incognito/private mode
- Clear all browser data for localhost

## **✅ Success Check**
When working, you should see:
- OmniBio logo
- Login form with "API Key" tab
- No red errors in browser console

**Current Status**: Frontend server is running at http://localhost:3000
**Action**: Try the steps above and check your browser console! 