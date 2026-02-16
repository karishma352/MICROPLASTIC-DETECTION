from flask import Flask,render_template,request,flash
import joblib 
import numpy as np
import os

app=Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'

# load trained model with error handling

try:
  model = joblib.load("logreg.pkl")
except FileNotFoundError:
  print("warning:Model file not found.Using dummy predictions.")
  model = None

# mapping back numeric prediction to original stage
stage_map = {
  0: 'NORMAL',
  1: 'HYPERTENSION(Stage-1)',
  2: 'HYPERTENSION(Stage-2)',
  3: 'HYPERTENSIVE CRISIS'
}

# medical-grade color mappping for results
color_map = {
  0:'#10B981',
  0:'#F59E0B',
  0:'#F97316',
  0:'#EF4444'
}

# Detailed medical recommendations

recommendations = {
  0:{
    'title' : 'Normal Blood Pressure',
    'description' : 'Your cadiovascular risk assessment indicates normal bllod pressure levels',
    'actions':[
      'Maintain current healthy lifestyle',
      'Regular physical activity (150 minutes/week)',
      'Continue balanced, low-sodium diet',
      'Regular health check-ups'
    ],
    'priority' : 'Low Risk'
  },
  1:{
    'title' : 'Stage 1 Hypertension',
    'description':'Mild elevation detected requiring lifestyle modifications and medical consultations',
    'actions':[
      'Scedule appoinment with healthcare provider',
      'Implement DASH diet plan',
      'Increase physical activity gradually',
      'Monitor blood pressure bi-weekly',
      'Reduced sodium intake(<2300mg/day)',
      'Consider stress management techniques'
    ],
    'priority':'Moderate Risk'
  },
  2:{
    'title' : 'Stage 2 Hypertension',
    'description':'Signifcant hypertension requiring immediate medical intervention and treatment.',
    'actions':[
      'URGENT : Consult physician within 1-2 days',
      'Likely medication therapy required',
      'Comprehensive cardiovascular assessment',
      'Daily blood pressure monitoring',
      'Strict dietary sodium restrictions',
      'Lifestyle modification counseling'
    ],
    'priority':'High Risk'
  },
  3:{
    'title' : 'Hypertensive crisis',
    'description':'CRITICAL : Dangerously elevated blood pressure requiring emergency medical care.',
    'actions':[
      'EMERGENCY : Seek immediate medical attention',
      'Call 108 if experiencing symptoms',
      'Do not delay treatment',
      'Monitor for stroke/heart attack signs',
      'Prepare current medication list',
      'Avoid physical exertion'
    ],
    'priority':'EMERGENCY'
  },
}
@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
  try:
    if request.method == 'POST':
      # collect user inputs from the form with validation
      required_fields = ['Gender','Age','History','Patient','TakeMedication','severity','BreathShortness','VisualChanges','NoseBleeding','Whendiagonsed','Systolic','Diasystolic','ControlledDiet']

      form_data = {}
      for field in required_fields:
        value =request.form.get(field)
        if not value or value =="":
          flash(f"Please complete all required fields :{field.replace('_',' ')}","error")
          return render_template('index.html')
        form_data[field] = value

        # encode inputs with error handling

      try:
        encoded = {
          0 if form_data['Gender']=='Male' else 1,
          {'18-34':1,'35-50':2,'51-64':3,'65+':4}[form_data['Age']],
          1 if form_data['History'] == 'Yes' else 0,
          1 if form_data['Patient'] == 'Yes' else 0,
          1 if form_data['TakeMedition'] == 'Yes' else 0,
          {'Mild':0,'Moderate':1,'Severe':2}[form_data['Severity']],
          1 if form_data['BreatheShortness'] == 'Yes' else 0,
          1 if form_data['VisualChanges'] == 'Yes' else 0,
          1 if form_data['NoseBleeding'] == 'Yes' else 0,
            {'<1 Year':0,'1-5 Years':1,'>5 Years':2}[form_data['Whendiagonoused']],
            {'100-110':0,'111 - 120':1,'121-130':2,'130+':3}[form_data['Systolic']],
             {'70-80':0,'81-90':1,'91-100':2,'100+':3}[form_data['Diastolic']],
             1 if form_data['ControlledDiet'] == 'Yes' else 0
        }

      except KeyError as e:
        flash(f"Invlaid selection detected:{str(e)}","error")
        return render_template('index.html')
      

      # manual scaling of ordinal features
      scaled_encoded = encoded.copy()
      scaled_encoded[1]=(encoded[1]-1)/(4-1)  #Age
      scaled_encoded[5]=(encoded[5])/2        #severity
      scaled_encoded[9]=(encoded[9]-1)/(3-1)  #when diagnosed
      scaled_encoded[10]=(encoded[10])/3      #systolic
      scaled_encoded[11]=(encoded[11])/3      #diasystolic

      input_array = np.array(scaled_encoded).reshape(1,-1)

      # predict with model or use dummy prediction
      if model is not None:
        prediction = model.predict(input_array)[0]
        try:
          confidence = max(model.predict_proba(input_array)[0])*100
        except:
          confidence = 85.0
      else:
        # dummy prediction  for testing when model is not available
        import random
        prediction = random.randint(0,3)
        confidence = 87.5
        flash("Demo Mode: Using simulated AI prediction for demostration","info")

      result_text = stage_map[prediction]
      result_color = color_map[prediction]
      result_recommendation = recommendations[prediction]

      return render_template('index.html',
                             prediction_text=result_text,
                             result_color=result_color,
                             confidence=confidence,
                             recommendation=result_recommendation,
                             form_data=form_data)
  except Exception as e:
    flash({"System error occured. Please try again or contact technical support.","error"})
    return render_template('index.html')
  
if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0',port=5000)




