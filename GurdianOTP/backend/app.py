import os
from flask import Flask, request
from pymongo import MongoClient
from balance import check_balance, add_credit, deduct_debit

app = Flask(__name__)

# Initialize MongoDB client
client = MongoClient('mongodb://localhost:27017/')
db = client['GurdianOTP']
collection = db['otp']

phone_number = "1234567890"

@app.route("/", methods=['POST'])
def ussd():
    # Read the variables sent via POST from our API
    session_id = request.values.get("sessionId", None)
    service_code = request.values.get("serviceCode", None)
    phone_number = request.values.get("phoneNumber", None)
    text = request.values.get("text", "default")

    if text == '':
        # This is the first request. Note how we start the response with CON
        response = "CON Welcome to USSD Menu Please select an option: \n"
        response += "1. Check Balance \n"
        response += "2. Deposit Money \n"
        response += "3. Withdraw Money \n"
        response += "4. Bank Transaction\n"
        response += "5. Support and Call"
       
        
    elif text.startswith('1'):
        # Business logic for balance inquiry
        balance, message = check_balance(phone_number)
        response = f'Your current account balance is {balance}' if message is None else message

    elif text == '2':
        # Business logic for deposit
        response = "CON Enter amount to deposit:"

    elif text.startswith('2*'):
        # Process deposit
        amount = float(text.split('*')[1])
        balance, message = add_credit(phone_number, amount)
        response = f"You have successfully deposited Ksh. {amount} into your account." if message is None else message

    elif text == '3':
        # Business logic for withdrawal
        available_balance, message = check_balance(phone_number)
        response = "Please enter the amount you would like to withdraw:" if message is None and available_balance > 0 else message

    elif text.startswith('3*'):
        # Process withdrawal
        amount = float(text.split('*')[1])
        balance, message = deduct_debit(phone_number, amount)
        response = "Withdrawal successful. Thank you!" if message is None else message

    elif text == '4':
        # Money Transaction
        response = "CON Enter recipient's phone number:"
    
    elif len(text) == 12:
        # Validate and process the transaction
        sender = db.users.find_one({"phone_number": phone_number})["name"]
        receiver = db.users.find_one({"phone_number": text})["name"]
        response = f"END Successfully sent to {receiver}."
        transfer(sender, text, float(text.split('*')[1]))     
        return response
        
    elif text == '5':
        # This is a terminal request. Note how we start the response with END
        response = "END For support, call our helpline at 123456789."

    else:
        response = "END Invalid choice"

    # Send the response back to the API
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
