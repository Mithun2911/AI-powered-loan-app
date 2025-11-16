from flask import Flask, request, render_template, session, jsonify
from markupsafe import escape 
import pickle
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'  # Change this in production
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path,'rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/form')
def form():
    return render_template("index.html")

@app.route('/chatbot')
def chatbot():
    # Initialize session for chatbot
    if 'chatbot_started' not in session:
        session['chatbot_started'] = False
        session['chatbot_step'] = -1
        session['chatbot_responses'] = []
    return render_template("chatbot.html")

@app.route('/test')
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running!'})

def preprocess_chatbot_data(gender, married, dependents, education, employed, credit, area, 
                   ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term):
    try:
        male = 1 if gender.lower() == "male" else 0
        married_yes = 1 if married.lower() == "yes" else 0
        if dependents == '1':
            dependents_1, dependents_2, dependents_3 = 1, 0, 0
        elif dependents == '2':
            dependents_1, dependents_2, dependents_3 = 0, 1, 0
        elif dependents == "3+":
            dependents_1, dependents_2, dependents_3 = 0, 0, 1
        else:
            dependents_1, dependents_2, dependents_3 = 0, 0, 0

        not_graduate = 1 if education.lower() == "not graduate" else 0
        employed_yes = 1 if employed.lower() == "yes" else 0
        semiurban = 1 if area.lower() == "semiurban" else 0
        urban = 1 if area.lower() == "urban" else 0

        ApplicantIncomelog = np.log(float(ApplicantIncome))
        totalincomelog = np.log(float(ApplicantIncome) + float(CoapplicantIncome))
        LoanAmountlog = np.log(float(LoanAmount))
        Loan_Amount_Termlog = np.log(float(Loan_Amount_Term))
        if float(credit) >= 850 and float(credit) <= 1000:
            credit = 1
        else:
            credit = 0

        return [
            credit, ApplicantIncomelog, LoanAmountlog, Loan_Amount_Termlog, totalincomelog,
            male, married_yes, dependents_1, dependents_2, dependents_3, not_graduate, employed_yes, semiurban, urban
        ]
    except Exception as e:
        return None

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return jsonify({
            'status': 'ok',
            'message': 'Chat endpoint is working. Use POST to send messages.',
            'chatbot_started': session.get('chatbot_started', False),
            'current_step': session.get('chatbot_step', -1)
        })
    
    try:
        if 'chatbot_started' not in session:
            session['chatbot_started'] = False
            session['chatbot_step'] = -1
            session['chatbot_responses'] = []
        
        if not request.is_json:
            return jsonify({
                'response': "Invalid request format. Please try again.",
                'completed': False,
                'error': True
            }), 400
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                'response': "Please provide a message.",
                'completed': False,
                'error': True
            }), 400
        
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'response': "Please enter a message.",
                'completed': False,
                'error': True
            }), 400
        
        questions = [
            "What is your gender? (Male/Female)",
            "Are you married? (Yes/No)",
            "How many dependents do you have? (0/1/2/3+)",
            "What is your education level? (Graduate/Not Graduate)",
            "Are you self-employed? (Yes/No)",
            "What is your monthly applicant income?",
            "What is your monthly co-applicant income?",
            "What is the loan amount you are requesting?",
            "What is the loan term in days?",
            "What is your credit history score? (0-1000)",
            "What is the property area? (Urban/Semiurban/Rural)"
        ]
        
        # Handle initial greeting
        if not session['chatbot_started']:
            if user_message.lower() == 'yes':
                session['chatbot_started'] = True
                session['chatbot_step'] = 0
                return jsonify({
                    'response': questions[0],
                    'completed': False
                })
            else:
                return jsonify({
                    'response': "No problem! Let me know when you're ready to begin by typing 'Yes'.",
                    'completed': False
                })
        
        # Handle questionnaire responses
        current_step = session['chatbot_step']
        
        if current_step < len(questions):
            # Validate input
            valid_input = True
            error_message = ""
            
            if current_step in [5, 6, 7, 8]:  # Numeric inputs
                try:
                    float(user_message)
                except ValueError:
                    valid_input = False
                    error_message = "Please enter a valid number."
            elif current_step == 9:  # Credit score
                try:
                    score = float(user_message)
                    if not (0 <= score <= 1000):
                        valid_input = False
                        error_message = "Credit score must be between 0 and 1000."
                except ValueError:
                    valid_input = False
                    error_message = "Please enter a valid credit score."
            
            if not valid_input:
                return jsonify({
                    'response': error_message,
                    'completed': False
                })
            
            # Store response
            responses = session.get('chatbot_responses', [])
            if len(responses) <= current_step:
                responses.extend([None] * (current_step + 1 - len(responses)))
            responses[current_step] = user_message
            session['chatbot_responses'] = responses
            session['chatbot_step'] = current_step + 1
            
            # Check if all questions answered
            if current_step + 1 < len(questions):
                return jsonify({
                    'response': questions[current_step + 1],
                    'completed': False
                })
            else:
                # All questions answered, process prediction
                try:
                    responses = session.get('chatbot_responses', [])
                    if len(responses) < len(questions):
                        return jsonify({
                            'response': "Incomplete information received. Please try again.",
                            'completed': False,
                            'error': True
                        })

                    gender = responses[0]
                    married = responses[1]
                    dependents = responses[2]
                    education = responses[3]
                    self_employed = responses[4]
                    applicant_income = responses[5]
                    coapplicant_income = responses[6]
                    loan_amount = responses[7]
                    loan_amount_term = responses[8]
                    credit_history = responses[9]
                    property_area = responses[10]
                    
                    # Preprocess and predict
                    features = preprocess_chatbot_data(
                        gender, married, dependents, education, self_employed, 
                        credit_history, property_area, applicant_income, 
                        coapplicant_income, loan_amount, loan_amount_term
                    )
                    
                    if features:
                        prediction = model.predict([features])
                        if prediction[0] == "N":
                            result = "No"
                            message = "⚠️ Based on your information, you are **not eligible** for the loan at this time. Consider improving your credit score, increasing your income, or reducing the loan amount."
                        else:
                            result = "Yes"
                            message = "🎉 Congratulations! Based on your information, you are **eligible** for the loan!"
                        
                        summary = f"""
Here is a summary of your information:
• Gender: {gender}
• Marital Status: {married}
• Dependents: {dependents}
• Education: {education}
• Self-Employed: {self_employed}
• Applicant Income: {applicant_income}
• Coapplicant Income: {coapplicant_income}
• Loan Amount: {loan_amount}
• Loan Term: {loan_amount_term} days
• Credit History: {credit_history}
• Property Area: {property_area}

**Loan Eligibility Result: {result}**

{message}
                        """
                        
                        # Reset session for new conversation
                        session.pop('chatbot_started', None)
                        session.pop('chatbot_step', None)
                        session.pop('chatbot_responses', None)
                        
                        return jsonify({
                            'response': summary,
                            'completed': True
                        })
                    else:
                        return jsonify({
                            'response': "Sorry, there was an error processing your information. Please try again.",
                            'completed': False
                        })
                except Exception as e:
                    return jsonify({
                        'response': f"Sorry, an error occurred: {str(e)}. Please try starting a new conversation.",
                        'completed': False,
                        'error': True
                    })
        
        return jsonify({
            'response': "I'm not sure how to respond to that. Please refresh the page to start a new conversation.",
            'completed': False
        })
    except Exception as e:
        import traceback
        print(f"Error in chat route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'response': f"Sorry, an unexpected error occurred: {str(e)}. Please try again.",
            'completed': False,
            'error': True
        }), 500

@app.route('/predict', methods = ["GET","POST"]) #get - typically used to show a blank prediction page or result page. #post-used to submit the form with input values that the server uses to make a prediction.
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        married = request.form['married']
        dependents = request.form['dependents']
        education = request.form['education']
        employed = request.form['employed']
        credit  = float(request.form['credit'])
        area = request.form['area']
        ApplicantIncome = float(request.form['ApplicantIncome']) #25000-> 0,1
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])


        #gender
        if (gender == "Male"):
            male = 1
        else:
            male = 0
        
        #married
        if (married == "Yes"):
            married_yes = 1
        else:
            married_yes = 0
        
        #dependents
        if ( dependents == '1'):
            dependents_1 = 1
            dependents_2 = 0
            dependents_3 = 0
        elif dependents == '2':
            dependents_1 = 0
            dependents_2 = 1
            dependents_3 = 0
        elif dependents == '3+':
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 1
        else:
            dependents_1 = 0
            dependents_2 = 0
            dependents_3 = 0

        #education 
        if education =="Not Graduate":
            not_graduate = 1
        else:
            not_graduate = 0

        #employed
        if (employed == "Yes"):
            employed_yes = 1
        else:
            employed_yes = 0
        
        #property area
        if area == "Semiurban":
            semiurban = 1
            urban = 0
        elif area == "Urban":
            semiurban = 0
            urban = 1
        else:
            semiurban = 0
            urban = 0

        ApplicantIncomeLog = np.log(ApplicantIncome)
        totalincomelog = np.log(ApplicantIncome+CoapplicantIncome)
        LoanAmountLog = np.log(LoanAmount)
        Loan_Amount_Termlog = np.log(Loan_Amount_Term)

        prediction = model.predict([[credit,ApplicantIncomeLog,LoanAmountLog,Loan_Amount_Termlog,totalincomelog,male,married_yes,dependents_1,dependents_2,dependents_3,not_graduate,employed_yes,semiurban,urban]])
        
        #print(prediction)
        if(prediction=="N"):
            prediction = "No"
        else:
            prediction = "Yes"
        return render_template("prediction.html",prediction_text="loan status is {}".format(prediction))
    else:
        return render_template("prediction.html")
if __name__ == "__main__":
    app.run(debug=True)
    
        