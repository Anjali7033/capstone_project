from flask import Flask, render_template, request
from functions import load_model, predict_spam

app = Flask(__name__)


@app.route("/")
def index():
    """Renders the main HTML page for email input."""
    return render_template("ecommerce.html", spam_prediction=None, error=None)


@app.route("/features")
def features():
    """Renders the main HTML page for email input."""
    return render_template("features.html", spam_prediction=None, error=None)


@app.route("/teams")
def teams():
    """Renders the main HTML page for email input."""
    return render_template("team.html", spam_prediction=None, error=None)


@app.route("/creditcard")
def creditcard():
    """Renders the main HTML page for email input."""
    return render_template("creditcard.html", spam_prediction=None, error=None)


@app.route("/password")
def password():
    """Renders the main HTML page for email input."""
    return render_template("password.html", spam_prediction=None, error=None)


@app.route("/email")
def email():
    """Renders the main HTML page for email input."""
    return render_template("emailspam.html", spam_prediction=None, error=None)


@app.route("/emailpredict", methods=["POST"])
def predict():
    """Handles POST requests for email content prediction."""
    if request.method == "POST":
        email_content = request.form["email_content"]
        try:
            model, vectorizer = load_model()
            prediction = predict_spam(email_content, model, vectorizer)
            return render_template("emailspam.html", spam_prediction=prediction, error=None)
        except Exception as e:
            return render_template("emailspam.html", spam_prediction=None, error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
